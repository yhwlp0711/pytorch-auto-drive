import torch
from torch import Tensor
from typing import Optional
from torch.nn import _reduction as _Reduction


class _Loss(torch.nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class WeightedLoss(_Loss):
    # 实现加权损失函数，允许用户传入权重来调整损失函数中不同样本的重要性
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None,
                 reduction: str = 'mean') -> None:
        # weight: 可选的权重张量，用于加权损失计算。
        # size_average: 这个参数已经被弃用，原来用于指定损失是否应该在所有样本上进行平均。现在的做法是使用 reduction 参数来指定如何计算损失。
        # reduce: 这个参数已经被弃用，原来用于指定是否应该对损失进行缩减。现在的做法是使用 reduction 参数来指定如何计算损失。
        # reduction: 指定如何计算损失的方法，可以是 'none'、'mean' 或 'sum'
        super(WeightedLoss, self).__init__(size_average, reduce, reduction)
        # 如果 weight 参数不为空且不是 Tensor 类型，则将其转换为 Tensor 类型并移动到 GPU
        if weight is not None and not isinstance(weight, Tensor):
            weight = torch.tensor(weight).cuda()
            # 使用 self.register_buffer 方法将 weight 添加为模型的缓冲区。
            # 这样做的好处是 weight 参数会被自动传递到模型的所有设备（如 GPU）
        self.register_buffer('weight', weight)
