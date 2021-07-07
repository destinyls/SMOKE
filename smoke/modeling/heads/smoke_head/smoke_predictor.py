import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import RoIAlign, RoIPool

from smoke.utils.registry import Registry
from smoke.modeling import registry
from smoke.layers.utils import sigmoid_hm, nms_hm, select_topk, select_point_of_interest
from smoke.modeling.make_layers import group_norm
from smoke.modeling.make_layers import _fill_fc_weights

from .loss import make_smoke_loss_evaluator

_HEAD_NORM_SPECS = Registry({
    "BN": nn.BatchNorm2d,
    "GN": group_norm,
})


def get_channel_spec(reg_channels, name):
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels)

    return slice(s, e, 1)


@registry.SMOKE_PREDICTOR.register("SMOKEPredictor")
class SMOKEPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SMOKEPredictor, self).__init__()

        classes = len(cfg.DATASETS.DETECT_CLASSES)
        regression = cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
        regression_channels = cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL
        head_conv = cfg.MODEL.SMOKE_HEAD.NUM_CHANNEL
        norm_func = _HEAD_NORM_SPECS[cfg.MODEL.SMOKE_HEAD.USE_NORMALIZATION]

        self.roi_align = RoIAlign(output_size=[1,1], spatial_scale=1, sampling_ratio=2)

        assert sum(regression_channels) == regression, \
            "the sum of {} must be equal to regression channel of {}".format(
                cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL, cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
            )

        self.max_objs = cfg.DATASETS.MAX_OBJECTS
        self.max_detection = cfg.TEST.DETECTIONS_PER_IMG
        self.loss_evaluator = make_smoke_loss_evaluator(cfg)

        self.dim_channel = get_channel_spec(regression_channels, name="dim")
        self.ori_channel = get_channel_spec(regression_channels, name="ori")

        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            norm_func(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv,
                      classes,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )
        # todo: what is datafill here
        self.class_head[-1].bias.data.fill_(-2.19)

        self.regression_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            nn.BatchNorm2d(head_conv),

            nn.ReLU(inplace=True),
        )
        _fill_fc_weights(self.regression_head)

        self.reg_3dbox = nn.Conv2d(640, regression, kernel_size=1, padding=1 // 2, bias=True)
        _fill_fc_weights(self.reg_3dbox)

    def forward(self, features, targets=None):
        up_level16, up_level8, up_level4 = features[0], features[1], features[2]

        head_class = self.class_head(up_level4)
        head_regression = self.regression_head(up_level4)

        head_class = sigmoid_hm(head_class)
        batch = head_class.shape[0]
        if self.training:
            targets_heatmap, targets_regression, targets_variables = self.loss_evaluator.prepare_targets(targets)
            proj_points = targets_variables["proj_points"]
        if not self.training:
            head_class_nms = nms_hm(head_class)
            scores, _, clses, ys, xs = select_topk(
                head_class_nms,
                K=self.max_detection,
            )
            proj_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1).unsqueeze(0)

        proj_points = proj_points.float()
        proj_points_8 = proj_points / 2
        proj_points_16 = proj_points / 4
        
        batch_id = torch.arange(batch, dtype=torch.float, device=proj_points.device).unsqueeze(1)
        roi_id = batch_id.repeat(1, proj_points.shape[1]).view(-1, 1)

        # RoIs
        proj_rois = torch.cat([proj_points - 2, proj_points + 2], dim=-1)
        proj_rois = proj_rois.view(-1, proj_rois.shape[-1])
        proj_rois = torch.cat([roi_id, proj_rois], dim=-1)
        
        proj_rois_8 = torch.cat([proj_points_8 - 1, proj_points_8 + 1], dim=-1)
        proj_rois_8 = proj_rois_8.view(-1, proj_rois_8.shape[-1])
        proj_rois_8 = torch.cat([roi_id, proj_rois_8], dim=-1)

        proj_rois_16 = torch.cat([proj_points_16 - 0.5, proj_points_16 + 0.5], dim=-1)
        proj_rois_16 = proj_rois_16.view(-1, proj_rois_16.shape[-1])
        proj_rois_16 = torch.cat([roi_id, proj_rois_16], dim=-1)
        
        regression_pois = self.roi_align(head_regression, proj_rois)
        up_level8_pois = self.roi_align(up_level8, proj_rois_8)
        up_level16_pois = self.roi_align(up_level16, proj_rois_16)

        regression_pois = regression_pois.view(batch, -1, 256 )
        up_level8_pois = up_level8_pois.view(batch, -1, 128)
        up_level16_pois = up_level16_pois.view(batch, -1, 256)

        regression_pois = torch.cat([regression_pois, up_level8_pois, up_level16_pois], dim=-1)
        regression_pois = regression_pois.permute(0, 2, 1).contiguous().unsqueeze(-1)

        # [N, 8, K, 1]
        head_regression = self.reg_3dbox(regression_pois)

        # (N, C, H, W)
        offset_dims = head_regression[:, self.dim_channel, ...].clone()
        head_regression[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5
        vector_ori = head_regression[:, self.ori_channel, ...].clone()
        head_regression[:, self.ori_channel, ...] = F.normalize(vector_ori)

        # [N, 8, K]
        head_regression = head_regression.squeeze(-1)
        if self.training:
            return [head_class, head_regression, targets_heatmap, \
                targets_regression, targets_variables]
        if not self.training:
            return [head_regression, scores, clses, ys, xs]

def make_smoke_predictor(cfg, in_channels):
    func = registry.SMOKE_PREDICTOR[
        cfg.MODEL.SMOKE_HEAD.PREDICTOR
    ]
    return func(cfg, in_channels)
