# Copied and modified from facebookresearch/detr
# Refactored and added comments


import torch
import torch.nn.functional
import torch.distributed
from scipy.optimize import linear_sum_assignment

from ..ddp_utils import is_dist_avail_and_initialized, get_world_size
from ..curve_utils import BezierSampler, cubic_bezier_curve_segment, get_valid_points
from ._utils import WeightedLoss
from .hungarian_loss import HungarianLoss
from .builder import LOSSES

# 匈牙利算法

# TODO: Speed-up Hungarian on GPU with tensors
class _HungarianMatcher(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    POTO matching, which maximizes the cost matrix.
    """
    """这个类计算网络目标和预测之间的分配关系。

    出于效率的考虑，目标不包括无对象。因此，一般来说，预测的数量会比目标多。在这种情况下，我们对最佳的预测进行一对一的匹配，而其他的则未匹配（因此被视为非对象）。
    使用POTO匹配方法，该方法最大化成本矩阵。
    """

    def __init__(self, alpha=0.8, bezier_order=3, num_sample_points=100, k=7):
        super().__init__()
        # alpha: 一个浮点数，默认值为0.8。
        # bezier_order: 贝塞尔曲线的阶数，默认值为3。
        # num_sample_points: 采样点的数量，默认值为100。
        # k: 一个整数，控制某些操作的参数，默认值为7。
        # bezier_sampler: 一个BezierSampler对象，用于贝塞尔曲线的采样，在贝塞尔曲线上生成均匀的采样点
        self.k = k
        self.alpha = alpha
        self.num_sample_points = num_sample_points
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points, order=bezier_order)

    @torch.no_grad()
    def forward(self, outputs, targets):
        # Compute the matrices for an entire batch (computation is all pairs, in a way includes the real loss function)
        # targets: each target: ['keypoints': L x N x 2]
        # B: batch size; Q: max lanes per-pred, G: total num ground-truth-lanes
        # targets: 每个目标都有一个 'keypoints' 键，对应一个形状为 L x N x 2 的数组。
        # 这里，L 表示每个目标的车道数，N 表示每条车道的关键点数，2 表示 (x, y) 坐标。
        # B: 批次大小，表示一个批次中的样本数。
        # Q: 每个预测的最大车道数。
        # G: 总的地面实况车道数。
        B, Q = outputs["logits"].shape
        target_keypoints = torch.cat([i['keypoints'] for i in targets], dim=0)  # G x N x 2
        target_sample_points = torch.cat([i['sample_points'] for i in targets], dim=0)  # G x num_sample_points x 2

        # Valid bezier segments
        # 对目标关键点target_keypoints进行分段处理，使其成为有效的贝塞尔曲线段
        # 获取采样点
        target_keypoints = cubic_bezier_curve_segment(target_keypoints, target_sample_points)
        target_sample_points = self.bezier_sampler.get_sample_points(target_keypoints)

        # target_valid_points = get_valid_points(target_sample_points)  # G x num_sample_points
        # 获取目标关键点target_keypoints的形状，其中G表示总的车道数，N表示每条车道的关键点数
        G, N = target_keypoints.shape[:2]
        # 对网络的输出logits进行sigmoid激活，得到每个车道存在的概率
        # 其中B是批次大小，Q是每个样本的最大车道数
        out_prob = outputs["logits"].sigmoid()  # B x Q
        out_lane = outputs['curves']  # B x Q x N x 2
        sizes = [target['keypoints'].shape[0] for target in targets]

        # 1. Local maxima prior
        # 进行局部最大值池化，以寻找每个车道线的局部最大值
        # max_indices是一个形状为(B, 1, Q)的张量
        # 其中每个元素代表对应车道线的局部最大值在原始概率值中的位置索引
        _, max_indices = torch.nn.functional.max_pool1d(out_prob.unsqueeze(1),
                                                        kernel_size=self.k, stride=1,
                                                        padding=(self.k - 1) // 2, return_indices=True)
        max_indices = max_indices.squeeze(1)  # B x Q
        # 形状为(BQ, G)，其中每个元素表示对应车道线的局部最大值是否在max_indices中
        indices = torch.arange(0, Q, dtype=out_prob.dtype, device=out_prob.device).unsqueeze(0).expand_as(max_indices)
        local_maxima = (max_indices == indices).flatten().unsqueeze(-1).expand(-1, G)  # BQ x G

        # Safe reshape
        out_prob = out_prob.flatten()  # BQ
        out_lane = out_lane.flatten(end_dim=1)  # BQ x N x 2

        # 2. Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # Then 1 can be omitted due to it is only a constant.
        # For binary classification, it is just prob (understand this prob as objectiveness in OD)
        # 其中每个元素是对应车道线的分类损失
        cost_label = out_prob.unsqueeze(-1).expand(-1, G)  # BQ x G

        # 3. Compute the curve sampling cost
        # torch.cdist计算两个点集之间的距离，使用范数p=1
        cost_curve = 1 - torch.cdist(self.bezier_sampler.get_sample_points(out_lane).flatten(start_dim=-2),
                                     target_sample_points.flatten(start_dim=-2),
                                     p=1) / self.num_sample_points  # BQ x G

        # Bound the cost to [0, 1]
        cost_curve = cost_curve.clamp(min=0, max=1)

        # Final cost matrix (scipy uses min instead of max)
        # 计算最终的成本矩阵C。这里使用了POTO匹配的成本函数
        # cost_label代表分类成本，cost_curve代表曲线采样成本。
        # self.alpha是一个权重参数，用于平衡两者之间的权重
        C = local_maxima * cost_label ** (1 - self.alpha) * cost_curve ** self.alpha
        # 将成本矩阵C重塑为(B, Q, G)的形状，并取负值（因为linear_sum_assignment使用最小化问题）
        C = -C.view(B, Q, -1).cpu()

        # Hungarian (weighted) on each image
        # 对于每个图像，使用匈牙利算法来找到最佳匹配的索引
        # 其中每个元素都是一个二元组，分别包含了匹配的行索引和列索引，代表了目标和预测之间的匹配关系
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # Return (pred_indices, target_indices) for each image
        # 中每个元素都是一个包含两个张量的元组 (pred_indices, target_indices)。
        # 这两个张量分别表示预测和目标之间的匹配索引
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@LOSSES.register()
class HungarianBezierLoss(WeightedLoss):
    def __init__(self, curve_weight=1, label_weight=0.1, seg_weight=0.75, alpha=0.8,
                 num_sample_points=100, bezier_order=3, weight=None, size_average=None, reduce=None, reduction='mean',
                 ignore_index=-100, weight_seg=None, k=9):
        super().__init__(weight, size_average, reduce, reduction)
        # curve_weight: 用于曲线采样点L1距离误差的权重。
        # label_weight: 用于分类误差的权重。
        # seg_weight: 用于二进制分割辅助任务误差的权重。
        # weight_seg: BCE损失的权重。
        # ignore_index: 忽略的索引，用于遮罩不考虑的类别或特定的标签。
        # bezier_sampler: 贝塞尔采样器，用于生成贝塞尔曲线上的采样点。
        # matcher: 匈牙利匹配器，用于为预测的车道线和目标车道线匹配最优索引。
        self.curve_weight = curve_weight  # Weight for sampled points' L1 distance error between curves
        self.label_weight = label_weight  # Weight for classification error
        self.seg_weight = seg_weight  # Weight for binary segmentation auxiliary task
        self.weight_seg = weight_seg  # BCE loss weight
        self.ignore_index = ignore_index
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points, order=bezier_order)
        self.matcher = _HungarianMatcher(alpha=alpha, num_sample_points=num_sample_points, bezier_order=bezier_order,
                                         k=k)
        if self.weight is not None and not isinstance(self.weight, torch.Tensor):
            self.weight = torch.tensor(self.weight).cuda()
        if self.weight_seg is not None and not isinstance(self.weight_seg, torch.Tensor):
            self.weight_seg = torch.tensor(self.weight_seg).cuda()
        self.register_buffer('pos_weight', self.weight[1] / self.weight[0])
        self.register_buffer('pos_weight_seg', self.weight_seg[1] / self.weight_seg[0])

    def forward(self, inputs, targets, net):
        # 通过网络net计算得到预测输出outputs，包括curves、logits和segmentations
        outputs = net(inputs)
        output_curves = outputs['curves']
        # target_labels: 初始化为与logits相同形状的全零张量。
        # target_segmentations: 从targets中提取segmentation_mask并堆叠为一个张量
        target_labels = torch.zeros_like(outputs['logits'])
        target_segmentations = torch.stack([target['segmentation_mask'] for target in targets])

        total_targets = 0
        for i in targets:
            total_targets += i['keypoints'].numel()

        # CULane actually can produce a whole batch of no-lane images,
        # in which case, we just calculate the classification loss
        if total_targets > 0:
            # Match
            # 通过匈牙利匹配器matcher进行预测与目标的匹配
            indices = self.matcher(outputs=outputs, targets=targets)
            # @staticmethod
            # def get_src_permutation_idx(indices):
            #     # Permute predictions following indices
            #     # 2-dim indices: (dim0 indices, dim1 indices)
            #     batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            #     image_idx = torch.cat([src for (src, _) in indices])

            #     return batch_idx, image_idx
            idx = HungarianLoss.get_src_permutation_idx(indices)
            output_curves = output_curves[idx]

            # Targets (rearrange each lane in the whole batch)
            # B x N x ... -> BN x ...
            # 根据匹配结果重新排序output_curves、target_keypoints和target_sample_points
            target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_sample_points = torch.cat([t['sample_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            # Valid bezier segments
            # 对目标曲线target_keypoints进行贝塞尔曲线分段处理。
            # 获取处理后的采样点target_sample_points
            target_keypoints = cubic_bezier_curve_segment(target_keypoints, target_sample_points)
            target_sample_points = self.bezier_sampler.get_sample_points(target_keypoints)

            target_labels[idx] = 1  # Any matched lane has the same label 1

        else:
            # For DDP
            target_sample_points = torch.tensor([], dtype=torch.float32, device=output_curves.device)

        target_valid_points = get_valid_points(target_sample_points)
        # Loss
        # point_loss: 计算曲线采样点的L1距离误差。
        # classification_loss: 计算分类误差。
        # binary_seg_loss: 计算二进制分割的辅助任务误差。
        loss_curve = self.point_loss(self.bezier_sampler.get_sample_points(output_curves),
                                     target_sample_points)
        loss_label = self.classification_loss(inputs=outputs['logits'], targets=target_labels)
        loss_seg = self.binary_seg_loss(inputs=outputs['segmentations'], targets=target_segmentations)

        loss = self.label_weight * loss_label + self.curve_weight * loss_curve + self.seg_weight * loss_seg

        return loss, {'training loss': loss, 'loss label': loss_label, 'loss curve': loss_curve,
                      'loss seg': loss_seg,
                      'valid portion': target_valid_points.float().mean()}

    def point_loss(self, inputs, targets, valid_points=None):
        # L1 loss on sample points
        # inputs/targets: L x N x 2
        # valid points: L x N
        # 计算两组采样点（inputs和targets）之间的L1损失。
        # 这种损失计算方式是对每个点的误差的绝对值之和。
        if targets.numel() == 0:
            targets = inputs.clone().detach()
        # 使用torch.nn.functional.l1_loss函数计算inputs和targets之间的L1损失。
        # 这会给出每个采样点的误差。
        loss = torch.nn.functional.l1_loss(inputs, targets, reduction='none')
        # 如果valid_points（形状为L x N的布尔掩码）被提供，那么将损失乘以这个掩码。
        # 这样可以忽略那些在某些条件下不重要或无效的点
        # 根据valid_points或targets的形状，计算正常化因子。
        # 这个因子用于将总损失归一化为每个有效点或总点数。
        if valid_points is not None:
            loss *= valid_points.unsqueeze(-1)
            normalizer = valid_points.sum()
        else:
            normalizer = targets.shape[0] * targets.shape[1]
            normalizer = torch.as_tensor([normalizer], dtype=inputs.dtype, device=inputs.device)
        if self.reduction == 'mean':
            if is_dist_avail_and_initialized():  # Global normalizer should be same across devices
                torch.distributed.all_reduce(normalizer)
            normalizer = torch.clamp(normalizer / get_world_size(), min=1).item()
            loss = loss.sum() / normalizer
        elif self.reduction == 'sum':  # Usually not needed, but let's have it anyway
            loss = loss.sum()
        # 根据self.reduction参数（可能是'mean'或'sum'）返回归一化或总损失。
        return loss

    def classification_loss(self, inputs, targets):
        # Typical classification loss (cross entropy)
        # No need for permutation, assume target is matched to inputs

        # Negative weight as positive weight
        # 计算二分类交叉熵损失
        return torch.nn.functional.binary_cross_entropy_with_logits(inputs.unsqueeze(1), targets.unsqueeze(1), pos_weight=self.pos_weight,
                                                  reduction=self.reduction) / self.pos_weight

    def binary_seg_loss(self, inputs, targets):
        # BCE segmentation loss with weighting and ignore index
        # No relation whatever to matching

        # Process inputs
        # 对输入进行双线性插值，使其与目标的形状相匹配
        inputs = torch.nn.functional.interpolate(inputs, size=targets.shape[-2:], mode='bilinear', align_corners=True)
        inputs = inputs.squeeze(1)

        # Process targets
        # 创建一个有效地图（valid_map），其中所有不等于self.ignore_index的值都被视为有效。
        # 将无效索引（self.ignore_index）的目标值设置为0。
        # 将目标值转换为浮点型
        valid_map = (targets != self.ignore_index)
        targets[~valid_map] = 0
        targets = targets.float()

        # Negative weight as positive weight
        # 设置权重和计算二进制交叉熵损失
        loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight_seg,
                                                  reduction='none') / self.pos_weight_seg
        loss *= valid_map

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
