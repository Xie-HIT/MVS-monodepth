# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
import PIL.Image as pil

import torch
import torch.utils.data as data
from torchvision import transforms

from ..kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

# image size
img_width = 640
img_height = 480
image_size = [img_width, img_height]

# TUM RGB-D Intrinsic
fx = 525.0  # focal length x
fy = 525.0  # focal length y
cx = 320  # optical center x
cy = 240  # optical center y

class TUM_xyz1(MonoDataset):
    def __init__(self, *args, **kwargs):
        # Img_ext: jpeg
        super(TUM_xyz1, self).__init__(*args, **kwargs)

        self.K = np.array([[fx, 0, cx, 0],
                           [0, fy, cy, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.K[0, :] /= img_width
        self.K[1, :] /= img_height

        self.full_res_shape = (img_width, img_height)

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        image_path = os.path.join(self.data_path, folder + self.img_ext)
        return image_path
