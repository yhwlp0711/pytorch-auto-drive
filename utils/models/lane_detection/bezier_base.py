import torch
import numpy as np

from .utils import lane_pruning
from ...curve_utils import BezierCurve
import torch.nn.functional


class BezierBaseNet(torch.nn.Module):
    def __init__(self, thresh=0.5, local_maximum_window_size=9):
        super().__init__()
        self.thresh = thresh
        self.local_maximum_window_size = local_maximum_window_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def bezier_to_coordinates(control_points, existence, resize_shape, dataset, bezier_curve, ppl=56, gap=10):
        # control_points: L x N x 2
        # control_points: 二维数组，形状为 L x N x 2，其中L是车道线的数量，N是每条车道线的控制点数量，2表示(x, y)坐标。
        # existence: 一个布尔数组，表示每条车道线是否存在。
        # resize_shape: 一个元组，表示调整后的图像形状，形如(H, W)。
        # dataset: 字符串，表示数据集类型，可以是'tusimple'、'culane'或'llamas'。
        # bezier_curve: 贝塞尔曲线对象，用于计算贝塞尔曲线上的点。
        # ppl: 整数，表示在'tusimple'数据集上采样的点数量。
        # gap: 整数，表示在'tusimple'数据集上采样点之间的垂直间隔
        H, W = resize_shape
        cps_of_lanes = []
        # 根据existence数组筛选出存在的车道线的控制点，并转换为列表格式
        for flag, cp in zip(existence, control_points):
            if flag:
                cps_of_lanes.append(cp.tolist())
        coordinates = []
        for cps_of_lane in cps_of_lanes:
            bezier_curve.assign_control_points(cps_of_lane)
            if dataset == 'tusimple':
                # Find x for TuSimple's fixed y eval positions (suboptimal)
                # 存在一组固定的y坐标，代表图像的垂直位置。对于这些固定的y坐标，
                # 代码需要确定对应的x坐标，即贝塞尔曲线在这些y坐标位置上的交点
                # 设置一个阈值，用于判断贝塞尔曲线上的点是否有效
                bezier_threshold = 5.0 / H
                # 生成固定y坐标的数组。其中ppl是采样点的数量，gap是y坐标之间的垂直间隔
                h_samples = np.array([1.0 - (ppl - i) * gap / H for i in range(ppl)], dtype=np.float32)
                # 使用quick_sample_point方法从贝塞尔曲线上快速采样点
                sampled_points = bezier_curve.quick_sample_point(image_size=None)
                temp = []
                # 对于每一个固定的y坐标，计算它与所有采样点y坐标的距离
                dis = np.abs(np.expand_dims(h_samples, -1) - sampled_points[:, 1])
                # 找到距离最小的采样点的索引
                idx = np.argmin(dis, axis=-1)
                # 根据阈值和边界条件，决定对应的x坐标。
                # 如果距离大于阈值或x坐标超出了边界（0到1之间），则将x坐标设置为-2。
                # 否则，将x坐标乘以图像宽度W，得到实际的x坐标
                for i in range(ppl):
                    h = H - (ppl - i) * gap
                    if dis[i][idx[i]] > bezier_threshold or sampled_points[idx[i]][0] > 1 or sampled_points[idx[i]][0] < 0:
                        temp.append([-2, h])
                    else:
                        temp.append([sampled_points[idx[i]][0] * W, h])
                coordinates.append(temp)
            elif dataset in ['culane', 'llamas']:
                temp = bezier_curve.quick_sample_point(image_size=None)
                temp[:, 0] = temp[:, 0] * W
                temp[:, 1] = temp[:, 1] * H
                coordinates.append(temp.tolist())
            else:
                raise ValueError

        return coordinates

    @torch.no_grad()
    # 模型推理：根据forward参数决定是否进行前向传播。如果forward为True，则调用self.forward(inputs)进行前向传播。
    # 计算存在概率：使用模型的输出outputs计算存在概率existence_conf，并根据阈值self.thresh确定每条车道线是否存在。
    # 局部最大值检测：如果self.local_maximum_window_size > 0，则进行局部最大值检测，确保每个局部最大值点都是车道线的开始或结束点。
    # 控制点处理：获取模型输出中的控制点control_points，并根据max_lane参数进行车道线数量的修剪。
    # 返回控制点：如果return_cps为True，则计算并返回调整后的控制点cps。
    # 贝塞尔曲线处理：根据dataset选择贝塞尔曲线的采样数量。然后，对每条车道线的控制点和存在标志调用bezier_to_coordinates方法，得到车道线的坐标。
    # 返回结果：返回车道线的坐标
    def inference(self, inputs, input_sizes, gap, ppl, dataset, max_lane=0, forward=True, return_cps=False, n=50):
        # 前向传播
        outputs = self.forward(inputs) if forward else inputs  # Support no forwarding inside this function
        # 计算车道线存在的概率
        existence_conf = outputs['logits'].sigmoid()
        # 确定是否存在 即得到的概率是否大于阈值
        existence = existence_conf > self.thresh

        # Test local maxima
        # 是否检测局部最大值
        if self.local_maximum_window_size > 0:
            # existence_conf.unsqueeze(1)：将existence_conf的形状从[B, N]变为[B, 1, N]。
            # kernel_size=self.local_maximum_window_size：设置池化窗口的大小。
            # stride=1：设置池化的步长。
            # padding=(self.local_maximum_window_size - 1) // 2：设置填充大小，确保池化后的输出与输入大小相同。
            # return_indices=True：返回池化操作后的最大值索引
            _, max_indices = torch.nn.functional.max_pool1d(existence_conf.unsqueeze(1),
                                                            kernel_size=self.local_maximum_window_size, stride=1,
                                                            padding=(self.local_maximum_window_size - 1) // 2,
                                                            return_indices=True)
            # 保存每个池化窗口内的最大值索引
            max_indices = max_indices.squeeze(1)  # B x Q
            # 生成一个索引数组，与max_indices具有相同的形状
            indices = torch.arange(0, existence_conf.shape[1],
                                   dtype=existence_conf.dtype,
                                   device=existence_conf.device).unsqueeze(0).expand_as(max_indices)
            # 将max_indices与indices进行比较，得到一个布尔数组local_maxima，表示哪些点是局部最大值
            local_maxima = max_indices == indices
            # 使用布尔数组local_maxima更新existence，将不是局部最大值的点的存在性置为0
            existence *= local_maxima

        # 控制点
        control_points = outputs['curves']
        # 如果max_lane不为0，代码会调用lane_pruning函数对车道线的存在性和存在概率进行修剪，
        # 以确保模型检测到的车道线数量不超过max_lane指定的最大值
        if max_lane != 0:  # Lane max number prior for testing
            existence, _ = lane_pruning(existence, existence_conf, max_lane=max_lane)

        # 如果return_cps为True，这段代码会计算并返回调整后的控制点cps。
        # 调整包括将控制点从相对于输入尺寸的比例转换为实际的坐标，并根据existence数组筛选出存在的控制点
        if return_cps:
            image_size = torch.tensor([input_sizes[1][1], input_sizes[1][0]],
                                      dtype=torch.float32, device=control_points.device)
            cps = control_points * image_size
            cps = [cps[i][existence[i]].cpu().numpy() for i in range(existence.shape[0])]

        # 进行了数据类型和设备的转换，以及获取输入图像的尺寸
        existence = existence.cpu().numpy()
        control_points = control_points.cpu().numpy()
        H, _ = input_sizes[1]
        # 阶数为3，num_sample_points 是要在曲线上采样的点的数量
        b = BezierCurve(order=3, num_sample_points=H if dataset == 'tusimple' else n)

        lane_coordinates = []
        for j in range(existence.shape[0]):
            lane_coordinates.append(self.bezier_to_coordinates(control_points=control_points[j], existence=existence[j],
                                                               resize_shape=input_sizes[1], dataset=dataset,
                                                               bezier_curve=b, gap=gap, ppl=ppl))
        if return_cps:
            return cps, lane_coordinates
        else:
            return lane_coordinates
