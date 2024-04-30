import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from mmpretrain import get_model
from .MyEfficientNet.cbam_model import cbam_EfficientNet

if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
else:
    from utils.torch_amp_dummy import autocast

from .bezier_base import BezierBaseNet
from ..builder import MODELS

# @MODELS.register()
# class SimpleBezierLaneNet(BezierBaseNet):
#     def __init__(self, backbone_cfg, num_outputs, thresh=0.5, local_maximum_window_size=9):
#         super(SimpleBezierLaneNet, self).__init__(thresh, local_maximum_window_size)
#
#         self.backbone = MODELS.from_dict(backbone_cfg)
#         self.model = EfficientNet.from_pretrained(self.backbone, num_classes=num_outputs)



@MODELS.register()
class BezierLaneNet(BezierBaseNet):
    # Curve regression network, similar design as simple object detection (e.g. FCOS)
    def __init__(self,
                 backbone_cfg,
                 reducer_cfg,
                 dilated_blocks_cfg,
                 feature_fusion_cfg,
                 head_cfg,
                 aux_seg_head_cfg,
                 image_height=360,
                 num_regression_parameters=8,
                 thresh=0.5,
                 local_maximum_window_size=9):
        super(BezierLaneNet, self).__init__(thresh, local_maximum_window_size)
        global_stride = 16
        # branch_channels = 256
        branch_channels = 112

        # ResNet
        # self.backbone = MODELS.from_dict(backbone_cfg)

        # "efficientnet-b0"
        # self.model = EfficientNet.from_pretrained("efficientnet-b0")
        # self.backbone = nn.Sequential(
        #     self.model._conv_stem,
        #     self.model._bn0,
        #     *list(self.model._blocks.children())[:-5]
        # )

        # CBMA_eff
        self.model = cbam_EfficientNet.from_pretrained('efficientnet-b0')
        # self.backbone = self.model.extract_features
        self.backbone = nn.Sequential(
            # self.model._conv_stem,
            # self.model._bn0,
            *list(self.model._blocks.children())[:-5]
        )


        # efficientnetv2-b0
        # self.model = get_model('efficientnetv2-b0_3rdparty_in1k', pretrained=True)
        # self.model = list(self.model.children())[1]
        # child = list(list(self.model.children())[0])
        # self.backbone = nn.Sequential(
        #     # model._conv_stem,
        #     # model._bn0,
        #     # *list(model.children())[:-5]
        #     *child[:-2]
        # )

        # self.backbone._fc = nn.Linear(self.backbone._fc.in_features, 45)
        self.reducer = MODELS.from_dict(reducer_cfg)
        self.dilated_blocks = MODELS.from_dict(dilated_blocks_cfg)
        self.simple_flip_2d = MODELS.from_dict(feature_fusion_cfg)  # Name kept for legacy weights
        # self.aggregator = nn.AvgPool2d(kernel_size=((image_height - 1) // global_stride + 1, 1), stride=1, padding=0)
        self.aggregator = nn.AvgPool2d(kernel_size=((image_height - 1) // global_stride, 1), stride=1, padding=0)
        self.regression_head = MODELS.from_dict(head_cfg)  # Name kept for legacy weights
        self.proj_classification = nn.Conv1d(branch_channels, 1, kernel_size=1, bias=True, padding=0)
        self.proj_regression = nn.Conv1d(branch_channels, num_regression_parameters,
                                         kernel_size=1, bias=True, padding=0)
        self.segmentation_head = MODELS.from_dict(aux_seg_head_cfg)

    def forward(self, x):
        # Return shape: B x Q, B x Q x N x 2
        # x: B * channels * H * W
        # print(self.backbone)
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))

        x = self.model.ca1(x) * x
        x = self.model.sa1(x) * x
        x = self.backbone(x)
        # x: B * 256 * 23 * 40
        if isinstance(x, dict):
            x = x['out']
        # # Segmentation task
        # if self.segmentation_head is not None:
        #     segmentations = self.segmentation_head(x)
        #     # segmentations: B * 1 * 23 * 40
        # else:
        #     segmentations = None

        if self.dilated_blocks is not None:
            x = self.dilated_blocks(x)
            # x: B * 256 * 23 * 40

        with autocast(False):  # TODO: Support fp16 like mmcv
            x = self.simple_flip_2d(x.float())
            # x: B * 256 * 23 * 40
        x = self.aggregator(x)[:, :, 0, :]
        # x: B * 256 * 40

        x = self.regression_head(x)
        # x: B * 256 * 40
        logits = self.proj_classification(x).squeeze(1)
        # logits: B * 40
        # 40 为预测总车道数
        curves = self.proj_regression(x)
        # curves: B * 8 * 40

        return {'logits': logits,
                'curves': curves.permute(0, 2, 1).reshape(curves.shape[0], -1, curves.shape[-2] // 2, 2).contiguous()}
                # 'segmentations': segmentations}

    def eval(self, profiling=False):
        super().eval()
        if profiling:
            self.segmentation_head = None
