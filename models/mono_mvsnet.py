import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


class monoMVSNet(nn.Module):
    def __init__(self, base_channels=8):
        super(monoMVSNet, self).__init__()
        self.feature = FeatureNet(base_channels=base_channels)
        self.scale_prediction = ScalePrediction()
        self.depth_range_sample = depth_range_sample
        self.depth_prediction = DepthPrediction()
