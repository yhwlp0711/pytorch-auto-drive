# Copied and modified from facebookresearch/detr and liuruijin17/LSTR
# Refactored and added comments
# Hungarian loss for LSTR
import torch
from torch import Tensor
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from ._utils import WeightedLoss
from ..models.lane_detection import cubic_curve_with_projection
from ..ddp_utils import is_dist_avail_and_initialized, get_world_size
from .builder import LOSSES


@torch.no_grad()
def lane_normalize_in_batch(keypoints):
    # Calculate normalization weights for lanes with different number of valid sample points,
    # so they can produce loss in a similar scale: rather weird but it is what LSTR did
    # https://github.com/liuruijin17/LSTR/blob/6044f7b2c5892dba7201c273ee632b4962350223/models/py_utils/matcher.py#L59
    # keypoints: [..., N, 2], ... means arbitrary number of leading dimensions
    # No gather/reduce is considered here as in the original implementation
    valid_points = keypoints[..., 0] > 0
    norm_weights = (valid_points.sum().float() / valid_points.sum(dim=-1).float()) ** 0.5
    norm_weights /= norm_weights.max()

    return norm_weights, valid_points  # [...], [..., N]


# TODO: Speed-up Hungarian on GPU with tensors
# Nothing will happen with DDP (for at last we use image-wise results)
class HungarianMatcher(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    # 该类计算目标和网络预测之间的赋值
    #
    # 由于效率原因，目标不包括 no_object。因此，一般情况下
    # 预测值多于目标值。在这种情况下，我们对最佳预测结果进行 1 对 1 匹配、
    # 而其他预测则不匹配（因此被视为非对象）。

    def __init__(self, upper_weight=2, lower_weight=2, curve_weight=5, label_weight=3):
        super().__init__()
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight
        self.curve_weight = curve_weight
        self.label_weight = label_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        # Compute the matrices for an entire batch (computation is all pairs, in a way includes the real loss function)
        # 算整个批次的矩阵（计算的是所有对，在某种程度上包括真实损失函数）

        # targets: each target: ['keypoints': L x N x 2, 'padding_mask': H x W, 'uppers': L, 'lowers': L, 'labels': L]
        # 'keypoints': 一个形状为 L x N x 2 的张量，表示每条车道的关键点坐标。L 是车道数量，N 是每条车道上的关键点数量，2 是坐标维度（例如，x 和 y）
        # 'padding_mask': 一个形状为 H x W 的张量，表示图像中的一个掩码，用于指示哪些区域是有效的（值为 1），哪些区域是无效的或填充的（值为 0）
        # 'uppers': 一个形状为 L 的张量，表示每条车道的上部坐标。
        # 'lowers': 一个形状为 L 的张量，表示每条车道的下部坐标。
        # 'labels': 一个形状为 L 的张量，表示每条车道的标签

        # B: bs; Q: max lanes per-pred, L: num lanes, N: num     keypoints per-lane, G: total num ground-truth-lanes
        # B: 批量大小 (batch size)。它表示每一批次有多少图像
        # Q: 每个预测的最大车道数。这表示模型预测中每个图像可能存在的最大车道数量
        # L: 实际的车道数量。这表示每个图像中真实的车道数量
        # N: 每条预测车道的关键点数量。这表示每条车道上的关键点数量
        # G: 总的真实车道数量。这表示在所有批次中，所有图像的车道总数

        # bs: 批量大小 (batch size)，即当前批次的图像数量。
        # num_queries: 每个图像的查询数，这通常是模型预测的最大车道数
        bs, num_queries = outputs["logits"].shape[:2]
        # 对"logits"进行softmax操作，以获取每个预测的概率。输出的形状为 BQ x 2，其中 B 是批量大小，Q 是每个预测的最大车道数，2 是两个类别的概率（例如，背景和车道）
        out_prob = outputs["logits"].softmax(dim=-1)  # BQ x 2
        # BQ * 三阶贝塞尔曲线的控制点
        out_lane = outputs['curves'].flatten(end_dim=-2)  # BQ x 8
        # 将目标中的所有上部坐标连接成一个张量。targets 是目标列表，每个目标都是一个字典，包含车道的上部坐标
        target_uppers = torch.cat([i['uppers'] for i in targets])
        target_lowers = torch.cat([i['lowers'] for i in targets])
        # 计算每个目标中的车道数量，并保存在sizes列表中
        sizes = [target['labels'].shape[0] for target in targets]
        # 计算所有目标中的总车道数量
        num_gt = sum(sizes)

        # 1. Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # Then 1 can be omitted due to it is only a constant.
        # For binary classification, it is just prob (understand this prob as objectiveness in OD)
        # 计算分类成本。与损失相反，我们不使用 NLL，而是将其近似为 1 - prob[目标类别]。
        # 对于二元分类，它只是 prob（将 prob 理解为 OD 中的客观性）。
        # out_prob[..., 1] 是预测的车道存在概率，我们取其负对数作为成本。
        # 这样，概率较高的车道（即存在的车道）将有较低的成本，概率较低的车道（即不存在的车道）将有较高的成本
        # 其中每一列是一个预测车道对所有真实车道的成本
        cost_label = -out_prob[..., 1].unsqueeze(-1).flatten(end_dim=-2).repeat(1, num_gt)  # BQ x G

        # 2. Compute the L1 cost between lowers and uppers
        # p=1 表示使用L1范数（也称为曼哈顿距离）来计算距离
        # 分别表示了预测车道的上界和下界与所有真实车道上界和下界之间的L1距离
        cost_upper = torch.cdist(out_lane[:, 0:1], target_uppers.unsqueeze(-1), p=1)  # BQ x G
        cost_lower = torch.cdist(out_lane[:, 1:2], target_lowers.unsqueeze(-1), p=1)  # BQ x G

        # 3. Compute the curve cost
        # 将所有真实车道的关键点（keypoints）按行连接在一起
        # G 是总的车道数量，N 是每条车道上的关键点数量，2 是每个关键点的坐标
        target_keypoints = torch.cat([i['keypoints'] for i in targets], dim=0)  # G x N x 2
        # 一个形状为 G 的张量，代表每条车道的归一化权重
        # 一个形状为 G x N 的布尔张量，代表哪些关键点是有效的
        norm_weights, valid_points = lane_normalize_in_batch(target_keypoints)  # G, G x N

        # Masked torch.cdist(p=1)
        expand_shape = [bs * num_queries, num_gt, target_keypoints.shape[-2]]  # BQ x G x N
        # coefficients: [k", f", m", n", b", b''']
        coefficients = out_lane[:, 2:].unsqueeze(1).expand(*expand_shape[:-1], -1)  # BQ x G x 6
        # 基于贝塞尔曲线系数和y坐标值计算x坐标值
        out_x = cubic_curve_with_projection(y=target_keypoints[:, :, 1].unsqueeze(0).expand(expand_shape),
                                            coefficients=coefficients)  # BQ x G x N
        # 每个预测车道与每个真实车道之间的成本
        cost_curve = ((out_x - target_keypoints[:, :, 0].unsqueeze(0).expand(expand_shape)).abs() *
                      valid_points.unsqueeze(0).expand(expand_shape)).sum(-1)  # BQ x G
        cost_curve *= norm_weights  # BQ x G

        # Final cost matrix
        # cost_label: 标签的损失成本，基于预测的概率输出。
        # cost_curve: 曲线的损失成本，基于预测的曲线与目标曲线之间的距离。
        # cost_lower: 下界的损失成本，基于预测的下界与目标下界之间的距离。
        # cost_upper: 上界的损失成本，基于预测的上界与目标上界之间的距离
        C = self.label_weight * cost_label + self.curve_weight * cost_curve + \
            self.lower_weight * cost_lower + self.upper_weight * cost_upper
        C = C.view(bs, num_queries, -1).cpu()

        # Hungarian (weighted) on each image
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # Return (pred_indices, target_indices) for each image
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# The Hungarian loss for LSTR
@LOSSES.register()
class HungarianLoss(WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, upper_weight=2, lower_weight=2, curve_weight=5, label_weight=3,
                 weight=None, size_average=None, reduce=None, reduction='mean'):
        # 接受多个权重参数：upper_weight, lower_weight, curve_weight, label_weight。
        # 调用父类 WeightedLoss 的初始化方法，并传递基本的损失函数参数。
        # 初始化匈牙利匹配器 HungarianMatcher，并传入权重参数
        super(HungarianLoss, self).__init__(weight, size_average, reduce, reduction)
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight
        self.curve_weight = curve_weight
        self.label_weight = label_weight
        self.matcher = HungarianMatcher(upper_weight, lower_weight, curve_weight, label_weight)

    @staticmethod
    def get_src_permutation_idx(indices):
        # Permute predictions following indices
        # 对下列指数进行置换预测
        # 2-dim indices: (dim0 indices, dim1 indices)
        # 接受一个索引列表 indices，这个列表包含两个维度的索引：dim0 indices 和 dim1 indices
        # batch_idx: 是一个一维张量，包含了每个预测的批次索引。
        # image_idx: 是一个一维张量，包含了每个预测在其批次内的图像索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        image_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, image_idx

    def forward(self, inputs: Tensor, targets: Tensor, net):
        # Support arbitrary auxiliary losses for transformer-based methods
        if 'padding_mask' in targets[0].keys():  # For multi-scale training support
            padding_masks = torch.stack([i['padding_mask'] for i in targets])
            outputs = net(inputs, padding_masks)
        else:
            outputs = net(inputs)
        loss, log_dict = self.calc_full_loss(outputs=outputs, targets=targets)
        if 'aux' in outputs:
            for i in range(len(outputs['aux'])):
                aux_loss, aux_log_dict = self.calc_full_loss(outputs=outputs['aux'][i], targets=targets)
                loss += aux_loss
                for k in list(log_dict):  # list(dict) is needed for Python3, since .keys() does not copy like Python2
                    log_dict[k + ' aux' + str(i)] = aux_log_dict[k]

        return loss, log_dict

    def calc_full_loss(self, outputs, targets):
        # Match
        indices = self.matcher(outputs=outputs, targets=targets)
        idx = self.get_src_permutation_idx(indices)

        # Targets (rearrange each lane in the whole batch)
        # B x N x ... -> BN x ...
        target_lowers = torch.cat([t['lowers'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_uppers = torch.cat([t['uppers'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.zeros(outputs['logits'].shape[:-1], dtype=torch.int64, device=outputs['logits'].device)
        target_labels[idx] = 1  # Any matched lane has the same label 1

        # Loss
        loss_label = self.classification_loss(inputs=outputs['logits'].permute(0, 2, 1), targets=target_labels)
        output_curves = outputs['curves'][idx]
        norm_weights, valid_points = lane_normalize_in_batch(target_keypoints)
        out_x = cubic_curve_with_projection(coefficients=output_curves[:, 2:],
                                            y=target_keypoints[:, :, 1].clone().detach())
        loss_curve = self.point_loss(inputs=out_x, targets=target_keypoints[:, :, 0],
                                     norm_weights=norm_weights, valid_points=valid_points)
        loss_upper = self.point_loss(inputs=output_curves[:, 0], targets=target_uppers)
        loss_lower = self.point_loss(inputs=output_curves[:, 1], targets=target_lowers)
        loss = self.label_weight * loss_label + self.curve_weight * loss_curve + \
            self.lower_weight * loss_lower + self.upper_weight * loss_upper

        return loss, {'training loss': loss, 'loss label': loss_label, 'loss curve': loss_curve,
                      'loss upper': loss_upper, 'loss lower': loss_lower}

    def point_loss(self, inputs: Tensor, targets: Tensor, norm_weights=None, valid_points=None) -> Tensor:
        # L1 loss on sample points, shouldn't it be direct regression?
        # Also, loss_lowers and loss_uppers in original LSTR code can be done with this same function
        # No need for permutation, assume target is matched to inputs
        # inputs/targets: L x N
        loss = F.l1_loss(inputs, targets, reduction='none')
        if norm_weights is not None:  # Weights for each lane
            loss *= norm_weights.unsqueeze(-1).expand_as(loss)
        if valid_points is not None:  # Valid points
            loss = loss[valid_points]
        if self.reduction == 'mean':
            normalizer = torch.as_tensor([targets.shape[0]], dtype=inputs.dtype, device=inputs.device)
            if is_dist_avail_and_initialized():  # Global normalizer should be same across devices
                torch.distributed.all_reduce(normalizer)
            normalizer = torch.clamp(normalizer / get_world_size(), min=1).item()
            loss = loss.sum() / normalizer  # Reduce only by number of curves (not number of points)
        elif self.reduction == 'sum':  # Usually not needed, but let's have it anyway
            loss = loss.sum()

        return loss

    def classification_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Typical classification loss (cross entropy)
        # No need for permutation, assume target is matched to inputs
        return F.cross_entropy(inputs, targets, reduction=self.reduction)
