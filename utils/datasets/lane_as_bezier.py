import os
import torch
import torchvision
import json
import numpy as np
from PIL import Image

from .builder import DATASETS
from ..curve_utils import BezierSampler, get_valid_points


class _BezierLaneDataset(torchvision.datasets.VisionDataset):
    # BezierLaneNet dataset, includes binary seg labels
    keypoint_color = [0, 0, 0]

    def __init__(self, root, image_set='train', transforms=None, transform=None, target_transform=None,
                 order=3, num_sample_points=100, aux_segmentation=False):
        super().__init__(root, transforms, transform, target_transform)
        # root: 数据集的根目录。
        # image_set: 指定数据集的子集，如 'train', 'val', 'test'。
        # transforms, transform, target_transform: 数据集的图像和目标转换。
        # order: 贝塞尔曲线的阶数。
        # num_sample_points: 贝塞尔曲线上的采样点数量。
        # aux_segmentation: 是否包含辅助的分割标注。
        self.aux_segmentation = aux_segmentation
        self.bezier_sampler = BezierSampler(order=order, num_sample_points=num_sample_points)
        if image_set == 'valfast':
            raise NotImplementedError('valfast Not supported yet!')
        elif image_set == 'test' or image_set == 'val':  # Different format (without lane existence annotations)
            self.test = 2
        elif image_set == 'val_train':
            self.test = 3
        else:
            self.test = 0

        self.init_dataset(root)

        if image_set != 'valfast':
            self.bezier_labels = os.path.join(self.bezier_labels_dir, image_set + '_' + str(order) + '.json')
        elif image_set == 'valfast':
            raise ValueError

        self.image_set = image_set
        self.splits_dir = os.path.join(root, 'lists')
        self._init_all()

    def init_dataset(self, root):
        raise NotImplementedError

    def __getitem__(self, index):
        # 根据给定的索引获取图像和目标
        # Return x (input image) & y (mask image, i.e. pixel-wise supervision) & lane existence (a list),
        # if not just testing,
        # else just return input image.
        img = Image.open(self.images[index]).convert('RGB')
        if self.test >= 2:
            target = self.masks[index]
        else:
            if self.aux_segmentation:
                target = {'keypoints': self.beziers[index],
                          'segmentation_mask': Image.open(self.masks[index])}
            else:
                target = {'keypoints': self.beziers[index]}

        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.test == 0:
            target = self._post_process(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def loader_bezier(self):
        # 加载贝塞尔标注的控制点
        results = []
        with open(self.bezier_labels, 'r') as f:
            results += [json.loads(x.strip()) for x in f.readlines()]
        beziers = []
        for lanes in results:
            temp_lane = []
            for lane in lanes['bezier_control_points']:
                temp_cps = []
                for i in range(0, len(lane), 2):
                    temp_cps.append([lane[i], lane[i + 1]])
                temp_lane.append(temp_cps)
            beziers.append(np.array(temp_lane, dtype=np.float32))
        return beziers

    def _init_all(self):
        # Got the lists from 4 datasets to be in the same format
        # 根据 image_set 和 test 属性初始化图像和目标的路径。
        # 加载图像和标注的列表文件
        data_list = 'train.txt' if self.image_set == 'val_train' else self.image_set + '.txt'
        split_f = os.path.join(self.splits_dir, data_list)
        with open(split_f, "r") as f:
            contents = [x.strip() for x in f.readlines()]
        if self.test == 2:  # Test
            self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x + self.output_suffix) for x in contents]
        elif self.test == 3:  # Test
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x[:x.find(' ')] + self.output_suffix) for x in contents]
        elif self.test == 1:  # Val
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
        else:  # Train
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            if self.aux_segmentation:
                self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
            self.beziers = self.loader_bezier()

    def _post_process(self, target, ignore_seg_index=255):
        # Get sample points and delete invalid lines (< 2 points)
        # 后处理方法，对目标进行一些额外的处理。
        # 对于车道的关键点，获取样本点并删除无效的车道。
        # 对于分割掩码，将其映射为二进制（0、1、255）
        if target['keypoints'].numel() != 0:  # No-lane cases can be handled in loss computation
            sample_points = self.bezier_sampler.get_sample_points(target['keypoints'])
            valid_lanes = get_valid_points(sample_points).sum(dim=-1) >= 2
            target['keypoints'] = target['keypoints'][valid_lanes]
            target['sample_points'] = sample_points[valid_lanes]
        else:
            target['sample_points'] = torch.tensor([], dtype=target['keypoints'].dtype)

        if 'segmentation_mask' in target.keys():  # Map to binary (0 1 255)
            positive_mask = (target['segmentation_mask'] > 0) * (target['segmentation_mask'] != ignore_seg_index)
            target['segmentation_mask'][positive_mask] = 1

        return target


# TuSimple
@DATASETS.register()
class TuSimpleAsBezier(_BezierLaneDataset):
    colors = [
        [0, 0, 0],  # background
        [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 255, 255],
        [0, 0, 0]  # ignore
    ]

    def init_dataset(self, root):
        # 设置图像的目录路径。这里，root是数据集的根目录，而图像存放在clips子目录下。
        # 设置Bezier标签的目录路径。Bezier标签通常包含贝塞尔曲线的控制点信息。
        # 设置分割掩码的目录路径。这些掩码通常用于标记图像中的特定区域或物体。
        # 设置输出文件名的前缀。在这里，输出文件的前缀与图像文件夹的名称相同。
        # 设置输出文件的后缀。这里，输出文件的后缀是.jpg，意味着输出的图像文件将以JPEG格式保存。
        # 设置图像文件的后缀。与输出文件的后缀相同，表示TuSimple数据集中的图像都是以JPEG格式存储的。
        self.image_dir = os.path.join(root, 'clips')
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.mask_dir = os.path.join(root, 'segGT6')
        self.output_prefix = 'clips'
        self.output_suffix = '.jpg'
        self.image_suffix = '.jpg'


# CULane
@DATASETS.register()
class CULaneAsBezier(_BezierLaneDataset):
    colors = [
        [0, 0, 0],  # background
        [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0],
        [0, 0, 0]  # ignore
    ]

    def init_dataset(self, root):
        self.image_dir = root
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.mask_dir = os.path.join(root, 'laneseg_label_w16')
        self.output_prefix = './output'
        self.output_suffix = '.lines.txt'
        self.image_suffix = '.jpg'
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)


# LLAMAS
@DATASETS.register()
class LLAMAS_AsBezier(_BezierLaneDataset):
    colors = [
        [0, 0, 0],  # background
        [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0],
        [0, 0, 0]  # ignore
    ]

    def init_dataset(self, root):
        self.image_dir = os.path.join(root, 'color_images')
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.mask_dir = os.path.join(root, 'laneseg_labels')
        self.output_prefix = './output'
        self.output_suffix = '.lines.txt'
        self.image_suffix = '.png'
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)


# Curvelanes
@DATASETS.register()
class Curvelanes_AsBezier(CULaneAsBezier):
    # TODO: Match formats
    colors = []

    def _init_all(self):
        # Got the lists from 4 datasets to be in the same format
        data_list = 'train.txt' if self.image_set == 'val_train' else self.image_set + '.txt'
        split_f = os.path.join(self.splits_dir, data_list)
        with open(split_f, "r") as f:
            contents = [x.strip() for x in f.readlines()]
        if self.test == 2:  # Test
            self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x + self.output_suffix) for x in contents]
        elif self.test == 3:  # Test
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x[:x.find(' ')] + self.output_suffix) for x in contents]
        elif self.test == 1:  # Val
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
        else:  # Train
            self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
            if self.aux_segmentation:
                self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
            self.beziers = self.loader_bezier()
