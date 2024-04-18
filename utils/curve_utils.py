import torch
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.special import comb as n_over_k


def upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    # https://github.com/pytorch/vision/pull/3383
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


class Polynomial(object):
    # Define Polynomials for curve fitting 
    def __init__(self, order):
        self.order = order

    def poly_fit(self, x_list, y_list, interpolate=False):
        self.coeff = np.polyfit(y_list, x_list, self.order)

    def compute_x_based_y(self, y, image_size):
        out = 0
        for i in range(self.order + 1):
            out += (y ** (self.order - i)) * self.coeff[i]
        if image_size is not None:
            out = out * image_size[-1]

        return out

    def print_coeff(self):
        print(self.coeff)

    def get_sample_point(self, y_list, image_size):
        coord_list = []
        for y in y_list:
            x = self.compute_x_based_y(y, None)
            coord_list.append([round(x, 3), y])
        coord_list = np.array(coord_list)
        if image_size is not None:
            coord_list[:, 0] = coord_list[:, 0] * image_size[-1]
            coord_list[:, -1] = coord_list[:, -1] * image_size[0]

        return coord_list


class BezierCurve(object):
    # Define Bezier curves for curve fitting
    def __init__(self, order, num_sample_points=50):
        # self.num_point：贝塞尔曲线的控制点数量，它等于 order + 1。
        # self.control_points：用于存储贝塞尔曲线的控制点，初始为空列表。
        # self.bezier_coeff：贝塞尔曲线的贝塞尔系数，通过 self.get_bezier_coefficient() 方法获取。
        # self.num_sample_points：贝塞尔曲线的采样点数量，默认为50。
        # self.c_matrix：伯恩斯坦矩阵（Bernstein Matrix），通过 self.get_bernstein_matrix() 方法获取
        self.num_point = order + 1
        self.control_points = []
        self.bezier_coeff = self.get_bezier_coefficient()
        self.num_sample_points = num_sample_points
        # 贝塞尔系数的矩阵形式
        self.c_matrix = self.get_bernstein_matrix()

    def get_bezier_coefficient(self):
        # 这是一个lambda函数，用于计算贝塞尔系数的每一项。它接受三个参数：n（曲线阶数），t（参数），k（索引）
        Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
        # 这是一个lambda函数，接受一个参数 ts，它是一个参数 t 的列表或数组。这个函数使用 Mtk 函数计算每个 t 对应的贝塞尔系数
        BezierCoeff = lambda ts: [[Mtk(self.num_point - 1, t, k) for k in range(self.num_point)] for t in ts]

        return BezierCoeff

    def interpolate_lane(self, x, y, n=50):
        # Spline interpolation of a lane. Used on the predictions
        # 对一个车道进行样条插值（Spline Interpolation）。它将输入的 x 和 y 坐标用样条曲线进行插值，从而得到更平滑的曲线
        assert len(x) == len(y)

        tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(x) - 1))

        u = np.linspace(0., 1., n)
        return np.array(splev(u, tck)).T

    def get_control_points(self, x, y, interpolate=False):
        # 是否使用插值方法
        if interpolate:
            # 使用 interpolate_lane 方法对 x 和 y 进行插值，得到平滑的车道坐标
            points = self.interpolate_lane(x, y)
            x = np.array([x for x, _ in points])
            y = np.array([y for _, y in points])

        # 将每两个连续的中间控制点作为一个控制点对添加到 self.control_points 列表中
        middle_points = self.get_middle_control_points(x, y)
        for idx in range(0, len(middle_points) - 1, 2):
            self.control_points.append([middle_points[idx], middle_points[idx + 1]])

    def get_bernstein_matrix(self):
        # 用于计算伯恩斯坦矩阵（Bernstein Matrix），它是贝塞尔曲线上一系列点的贝塞尔系数的矩阵形式
        tokens = np.linspace(0, 1, self.num_sample_points)
        c_matrix = self.bezier_coeff(tokens)
        return np.array(c_matrix)

    def save_control_points(self):
        return self.control_points

    def assign_control_points(self, control_points):
        self.control_points = control_points

    def quick_sample_point(self, image_size=None):
        # 快速采样贝塞尔曲线上的点。
        # 它使用之前计算的伯恩斯坦矩阵 self.c_matrix 和控制点矩阵来计算贝塞尔曲线上的采样点
        control_points_matrix = np.array(self.control_points)
        sample_points = self.c_matrix.dot(control_points_matrix)
        if image_size is not None:
            sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
            sample_points[:, -1] = sample_points[:, -1] * image_size[0]
        return sample_points

    def get_sample_point(self, n=50, image_size=None):
        '''
            :param n: the number of sampled points
            :return: a list of sampled points
        '''
        # 不同方法进行采样
        # 与 quick_sample_point 方法相比，get_sample_point 方法更为直接地使用了贝塞尔系数和控制点来计算采样点
        t = np.linspace(0, 1, n)
        coeff_matrix = np.array(self.bezier_coeff(t))
        control_points_matrix = np.array(self.control_points)
        sample_points = coeff_matrix.dot(control_points_matrix)
        if image_size is not None:
            sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
            sample_points[:, -1] = sample_points[:, -1] * image_size[0]

        return sample_points

    def get_middle_control_points(self, x, y):
        # 这个 get_middle_control_points 方法用于计算贝塞尔曲线中间的控制点。
        # 它使用了输入的 x 和 y 坐标来计算中间控制点，以便更好地逼近输入的坐标点
        dy = y[1:] - y[:-1]
        dx = x[1:] - x[:-1]
        dt = (dx ** 2 + dy ** 2) ** 0.5
        t = dt / dt.sum()
        t = np.hstack(([0], t))
        t = t.cumsum()
        data = np.column_stack((x, y))
        Pseudoinverse = np.linalg.pinv(self.bezier_coeff(t))  # (9,4) -> (4,9)
        control_points = Pseudoinverse.dot(data)  # (4,9)*(9,2) -> (4,2)
        medi_ctp = control_points[:, :].flatten().tolist()

        return medi_ctp


class BezierSampler(torch.nn.Module):
    # Fast Batch Bezier sampler
    # 用于批量采样贝塞尔曲线上点的PyTorch模块。它使用了贝塞尔系数和伯恩斯坦矩阵来快速计算贝塞尔曲线上的采样点
    def __init__(self, order, num_sample_points, proj_coefficient=0):
        super().__init__()
        # order：贝塞尔曲线的阶数。
        # num_sample_points：要采样的点的数量。
        # proj_coefficient：投影系数，用于修改采样点的分布。
        self.proj_coefficient = proj_coefficient
        self.num_control_points = order + 1
        self.num_sample_points = num_sample_points
        self.control_points = []
        # self.bezier_coeff = self.get_bezier_coefficient()
        self.bernstein_matrix = self.get_bernstein_matrix()

    def Mtk(self, n, t, k):
        return t ** k * (1 - t) ** (n - k) * n_over_k(n, k)

    def bezier_coeff(self, ts):
        return  [[self.Mtk(self.num_control_points - 1, t, k) for k in range(self.num_control_points)] for t
                                  in ts]

    # def get_bezier_coefficient(self):
    #     Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
    #     BezierCoeff = lambda ts: [[Mtk(self.num_control_points - 1, t, k) for k in range(self.num_control_points)] for t in ts]
    #     return BezierCoeff

    def get_bernstein_matrix(self):
        t = torch.linspace(0, 1, self.num_sample_points)
        if self.proj_coefficient != 0:
            # tokens = tokens + (1 - tokens) * tokens ** self.proj_coefficient
            t[t > 0.5] = t[t > 0.5] + (1 - t[t > 0.5]) * t[t > 0.5] ** self.proj_coefficient
            t[t < 0.5] = 1 - (1 - t[t < 0.5] + t[t < 0.5] * (1 - t[t < 0.5]) ** self.proj_coefficient)
        c_matrix = torch.tensor(self.bezier_coeff(t))
        return c_matrix

    def get_sample_points(self, control_points_matrix):
        if control_points_matrix.numel() == 0:
            return control_points_matrix  # Looks better than a torch.Tensor
        if self.bernstein_matrix.device != control_points_matrix.device:
            self.bernstein_matrix = self.bernstein_matrix.to(control_points_matrix.device)

        return upcast(self.bernstein_matrix).matmul(upcast(control_points_matrix))


@torch.no_grad()
def get_valid_points(points):
    # ... x 2
    # 首先检查 points 是否为空
    if points.numel() == 0:
        return torch.tensor([1], dtype=torch.bool, device=points.device)
    # 对于非空的点集，使用逐元素的逻辑运算符 * 来检查每个点的坐标是否都在 (0, 1) 范围内
    return (points[..., 0] > 0) * (points[..., 0] < 1) * (points[..., 1] > 0) * (points[..., 1] < 1)


@torch.no_grad()
def cubic_bezier_curve_segment(control_points, sample_points):
    # Cut a batch of cubic bezier curves to its in-image segments (assume at least 2 valid sample points per curve).
    # Based on De Casteljau's algorithm, formula for cubic bezier curve is derived by:
    # https://stackoverflow.com/a/11704152/15449902
    # 批次为 B
    # control_points: B x 4 x 2
    # 样本点
    # sample_points: B x N x 2
    # 返回一个形状为 B x 4 x 2 的张量，代表切割后的贝塞尔曲线的控制点
    if control_points.numel() == 0 or sample_points.numel() == 0:
        return control_points
    B, N = sample_points.shape[:-1]
    # 获取有效的样本点的布尔掩码
    valid_points = get_valid_points(sample_points)  # B x N, bool
    # 生成参数 t，表示每个样本点在曲线上的位置
    t = torch.linspace(0.0, 1.0, steps=N, dtype=sample_points.dtype, device=sample_points.device)

    # First & Last valid index (B)
    # Get unique values for deterministic behaviour on cuda:
    # https://pytorch.org/docs/1.6.0/generated/torch.max.html?highlight=max#torch.max
    # 计算每条曲线的起始和结束参数 t0 和 t1，用于切割曲线
    t0 = t[(valid_points + torch.arange(N, device=valid_points.device).flip([0]) * valid_points).max(dim=-1).indices]
    t1 = t[(valid_points + torch.arange(N, device=valid_points.device) * valid_points).max(dim=-1).indices]

    # Generate transform matrix (old control points -> new control points = linear transform)
    # 将原始的控制点映射到新的控制点。这个变换是线性的，即可以用一个矩阵乘法来表示
    u0 = 1 - t0  # B
    u1 = 1 - t1  # B
    transform_matrix_c = [torch.stack([u0 ** (3 - i) * u1 ** i for i in range(4)], dim=-1),
                          torch.stack([3 * t0 * u0 ** 2,
                                       2 * t0 * u0 * u1 + u0 ** 2 * t1,
                                       t0 * u1 ** 2 + 2 * u0 * u1 * t1,
                                       3 * t1 * u1 ** 2], dim=-1),
                          torch.stack([3 * t0 ** 2 * u0,
                                       t0 ** 2 * u1 + 2 * t0 * t1 * u0,
                                       2 * t0 * t1 * u1 + t1 ** 2 * u0,
                                       3 * t1 ** 2 * u1], dim=-1),
                          torch.stack([t0 ** (3 - i) * t1 ** i for i in range(4)], dim=-1)]
    transform_matrix = torch.stack(transform_matrix_c, dim=-2).transpose(-2, -1)  # B x 4 x 4, f**k this!
    transform_matrix = transform_matrix.unsqueeze(1).expand(B, 2, 4, 4)

    # Matrix multiplication
    # 矩阵乘法
    res = transform_matrix.matmul(control_points.permute(0, 2, 1).unsqueeze(-1))  # B x 2 x 4 x 1

    return res.squeeze(-1).permute(0, 2, 1)
