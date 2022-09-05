import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


class monoMSNet(nn.Module):
    def __init__(self):
        super(monoMSNet, self).__init__()
        self.feature = FeatureNet()
