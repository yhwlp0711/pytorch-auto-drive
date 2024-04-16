import os
import torch
import time
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast, GradScaler
else:
    from ..torch_amp_dummy import autocast, GradScaler

from ..common import save_checkpoint
from ..ddp_utils import reduce_dict, is_main_process
from .lane_det_tester import LaneDetTester
from .base import BaseTrainer, DATASETS, TRANSFORMS


class LaneDetTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        # Should be the same as segmentation, given customized loss classes
        # 设置模型为训练模式。
        # 初始化epoch为0。
        # 初始化running_loss为None，用于记录每个loss的运行平均值。
        # 设置loss_num_steps，用于确定多少步后记录一次loss。
        self.model.train()
        epoch = 0
        running_loss = None  # Dict logging for every loss (too many losses in this task)
        loss_num_steps = int(len(self.dataloader) / 10) if len(self.dataloader) > 10 else 1
        # 如果配置中启用了混合精度训练(mixed_precision)，则初始化一个GradScaler对象。
        if self._cfg['mixed_precision']:
            scaler = GradScaler()

        # Training
        best_validation = 0
        # 循环每个epoch，直到达到预定的num_epochs
        while epoch < self._cfg['num_epochs']:
            self.model.train()
            # 函数就是用来设置每个epoch的随机种子的，这样确保每个训练节点使用的数据顺序是不同的，从而增加模型的泛化能力
            if self._cfg['distributed']:
                self.train_sampler.set_epoch(epoch)
            time_now = time.time()
            # for batch size
            for i, data in enumerate(self.dataloader, 0):
                # 这段代码检查配置文件中的'seg'键是否为True，如果是的话，它期望data包含三个部分：inputs、labels和existence
                if self._cfg['seg']:
                    # inputs: 输入数据，通常是图像或其他类型的特征
                    # labels: 标签，用于监督学习的目标输出
                    # existence: 一个存在性标志，可能表示某个实体（如车道线或物体）是否存在
                    inputs, labels, existence = data
                    inputs, labels, existence = inputs.to(self.device), labels.to(self.device), existence.to(self.device)
                # 假设data只包含两部分：inputs和labels
                else:
                    # inputs: 输入数据，通常是图像或其他类型的特征
                    # labels: 标签，用于监督学习的目标输出
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    if self._cfg['collate_fn'] is None:
                        # labels直接被移到设备上
                        labels = labels.to(self.device)
                    else:
                        # 它会被应用于每一个label，并且每一个键值对（k: v）都会被移到设备上。
                        # 这种处理可能会比较慢，因为它涉及到了更多的数据移动和复制操作
                        labels = [{k: v.to(self.device) for k, v in label.items()} for label in labels]  # Seems slow
                # 每个训练步骤之前清零优化器的梯度
                self.optimizer.zero_grad()

                with autocast(self._cfg['mixed_precision']):
                    # To support intermediate losses for SAD
                    # 是否为分割任务
                    # self.criterion = LOSSES.from_dict(cfg['loss'])
                    if self._cfg['seg']:
                        loss, log_dict = self.criterion(inputs, labels, existence,
                                                        self.model, self._cfg['input_size'])
                    else:
                        loss, log_dict = self.criterion(inputs, labels,
                                                        self.model)

                if self._cfg['mixed_precision']:
                    # 对损失进行缩放并进行反向传播。这里使用的缩放因子是通过 GradScaler 自动选择的
                    scaler.scale(loss).backward()
                    # 使用缩放后的梯度来更新模型参数。这里使用的是优化器，它会根据缩放后的梯度来更新模型
                    scaler.step(self.optimizer)
                    # 更新 GradScaler 的缩放因子，准备下一次迭代
                    scaler.update()
                else:
                    # 计算损失的梯度
                    loss.backward()
                    # 使用计算得到的梯度来更新模型参数
                    self.optimizer.step()
                # 更新学习率
                self.lr_scheduler.step()

                # 同步和汇总指标值，从而确保在整个训练过程中能够准确地记录和监控模型的性能
                log_dict = reduce_dict(log_dict)
                if running_loss is None:  # Because different methods may have different values to log
                    # 第一次更新 running_loss，需要初始化它
                    # 为了与 log_dict 中的所有指标名称对应，使用 log_dict.keys() 创建一个新的字典，其中所有指标的值都被初始化为 0.0
                    running_loss = {k: 0.0 for k in log_dict.keys()}
                # 这段代码是在累积 log_dict 中的各个损失或指标到 running_loss 字典中
                for k in log_dict.keys():
                    running_loss[k] += log_dict[k]
                # 这行代码计算当前训练的总步数 current_step_num。在多次迭代中，这个值用于跟踪训练的总进度
                current_step_num = int(epoch * len(self.dataloader) + i + 1)

                # Record losses
                # 这部分代码用于记录和打印损失值
                # 这样做是为了每完成 loss_num_steps 步就记录一次损失值
                if current_step_num % loss_num_steps == (loss_num_steps - 1):
                    # 遍历 running_loss 字典中的所有键（损失的名称或类型）
                    for k in running_loss.keys():
                        print('[%d, %d] %s: %.4f' % (epoch + 1, i + 1, k, running_loss[k] / loss_num_steps))
                        # Logging only once
                        # 如果当前进程是主进程（is_main_process() 返回 True），则将损失值写入到Tensorboard或其他可视化工具
                        if is_main_process():
                            self.writer.add_scalar(k, running_loss[k] / loss_num_steps, current_step_num)
                        running_loss[k] = 0.0

                # Record checkpoints
                # 记录模型检查点（checkpoints）和在验证集上评估模型
                if self._cfg['validation']:
                    assert self._cfg['seg'], 'Only segmentation based methods can be fast evaluated!'
                    if current_step_num % self._cfg['val_num_steps'] == (self._cfg['val_num_steps'] - 1) or \
                            current_step_num == self._cfg['num_epochs'] * len(self.dataloader):
                        test_pixel_accuracy, test_mIoU = LaneDetTester.fast_evaluate(
                            loader=self.validation_loader,
                            device=self.device,
                            net=self.model,
                            num_classes=self._cfg['num_classes'],
                            output_size=self._cfg['input_size'],
                            mixed_precision=self._cfg['mixed_precision'])
                        if is_main_process():
                            self.writer.add_scalar('test pixel accuracy',
                                                   test_pixel_accuracy,
                                                   current_step_num)
                            self.writer.add_scalar('test mIoU',
                                                   test_mIoU,
                                                   current_step_num)
                        self.model.train()

                        # Record best model (straight to disk)
                        if test_mIoU > best_validation:
                            best_validation = test_mIoU
                            save_checkpoint(net=self.model.module if self._cfg['distributed'] else self.model,
                                            optimizer=None,
                                            lr_scheduler=None,
                                            filename=os.path.join(self._cfg['exp_dir'], 'model.pt'))

            epoch += 1
            print('Epoch time: %.2fs' % (time.time() - time_now))

        # For no-evaluation mode
        # 如果训练不需要验证，那么在最后保存模型
        if not self._cfg['validation']:
            save_checkpoint(net=self.model.module if self._cfg['distributed'] else self.model,
                            optimizer=None,
                            lr_scheduler=None,
                            filename=os.path.join(self._cfg['exp_dir'], 'model.pt'))

    def get_validation_dataset(self, cfg):
        # 获取验证集的数据集对象
        if not self._cfg['validation']:
            return None
        validation_transforms = TRANSFORMS.from_dict(cfg['test_augmentation'])
        validation_set = DATASETS.from_dict(cfg['validation_dataset'],
                                            transforms=validation_transforms)
        return validation_set
