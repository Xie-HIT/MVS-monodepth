#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.utils import *


class Replica_TANDEM(Dataset):
    def __init__(self, split, option):
        super(Replica_TANDEM, self).__init__()
        self.option = option
        self.root_path = self.option.training['Replica']['path']
        self.width = self.option.model['width']
        self.height = self.option.model['height']
        assert self.width % 32 == 0 and self.height % 32 == 0
        self.min_depth = self.option.model['min_depth']
        self.max_depth = self.option.model['max_depth']
        self.pyramid_scale = self.option.model['pyramid_scale']
        self.samples = []

        if split == 'train':
            self.training = True
            self.evaluation = False
        elif split == 'val':
            self.training = False
            self.evaluation = True
        else:
            raise NotImplementedError('\'split\' can only choose from \'{}\' or \'{}\''.format('train', 'val'))

        split_path = os.path.join(self.root_path, split + '.txt')
        with open(split_path, 'r') as f:
            line = f.readline()
            scenes = line.rstrip().split(' ')
        for scene in scenes:
            scene_path = os.path.join(self.root_path, scene)

            # read intrinsic
            scene_intrinsic_path = os.path.join(scene_path, 'camera.txt')
            with open(scene_intrinsic_path, 'r') as f:
                line = f.readline()
                line = line.rstrip().split(' ')
                fx, fy, cx, cy, _ = [float(x) for x in line[1:]]
                orin_intrinsic = np.zeros((3, 3), dtype=np.float32)
                orin_intrinsic[(0, 0, 1, 1, 2), (0, 2, 1, 2, 2)] = (fx, cx, fy, cy, 1)

                line = f.readline()
                line = line.rstrip().split(" ")
                orin_width, orin_height = int(line[0]), int(line[1])

            # make sample
            scene_pose_path = os.path.join(scene_path, 'poses_gt.txt')
            with open(scene_pose_path, 'r') as f:
                lines = f.readlines()

                frames_idx = []
                for line in lines:
                    line = line.rstrip().split(' ')
                    frame_idx = int(line[0])
                    frames_idx.append(frame_idx)

            if self.training is True:
                num_views = self.option.training['Replica']['view_selection']['num_views']
                dilation = self.option.training['Replica']['view_selection']['data_dilation']
                stride = self.option.training['Replica']['view_selection']['data_stride']
            elif self.evaluation is True:
                num_views = self.option.training['Replica']['view_selection']['num_views']
                dilation = self.option.training['Replica']['view_selection']['data_dilation']
                stride = self.option.training['Replica']['view_selection']['data_stride']
            else:
                raise NotImplementedError()

            sample_size = num_views + (num_views - 1) * dilation
            num_sample = (len(frames_idx) - sample_size) // stride + 1
            for i in range(num_sample):
                views_idx = []
                for j in range(num_views):
                    views_idx.append(i + j * (dilation + 1))
                sample = {
                    "scene": scene,
                    "width": orin_width,
                    "height": orin_height,
                    "scene_intrinsic": orin_intrinsic,
                    "views_idx": views_idx  # (ref, src, src)
                }
                self.samples.append(sample)
            self.num_views = num_views

        self.size = len(self.samples)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        scene = sample['scene']
        scene_path = os.path.join(self.root_path, scene)

        # read poses
        scene_pose_path = os.path.join(scene_path, 'poses_gt.txt')
        with open(scene_pose_path, 'r') as f:
            lines = f.readlines()

            poses = []
            for view_idx in sample['views_idx']:
                line = lines[view_idx].rstrip().split(' ')
                assert view_idx == int(line[0])
                pose = np.reshape(np.array([float(line[i]) for i in range(1, 17)], dtype=np.float32), (4, 4))
                pose = torch.from_numpy(pose)
                poses.append(pose)

        # compute intrinsic
        scale_x = float(self.width) / float(sample['width'])
        scale_y = float(self.height) / float(sample['height'])

        intrinsic = sample['scene_intrinsic'].copy()
        intrinsic[0] = intrinsic[0] * scale_x
        intrinsic[1] = intrinsic[1] * scale_y

        intrinsics = [torch.from_numpy(intrinsic) for i in range(self.num_views)]

        # read images
        images = []
        for view_idx in sample['views_idx']:
            scene_image_path = os.path.join(scene_path, 'images', f'{view_idx:06d}.jpg')
            image = read_image(scene_image_path, size=(self.height, self.width)).squeeze(0)  # (3, H, W)
            images.append(image)
        images = torch.stack(images, dim=0)

        # read depth map of reference frame
        ref_idx = sample['views_idx'][0]
        scene_depth_path = os.path.join(scene_path, 'depths', f'{ref_idx:06d}.png')
        scene_depth_scale_path = os.path.join(scene_path, 'depths', 'scale.txt')
        with open(scene_depth_scale_path, 'r') as f:
            line = f.readline()
            line = line.rstrip().split(' ')
            depth_scale = float(line[0])
        depth = read_depth(scene_depth_path, depth_scale, size=(self.height, self.width)).squeeze(0)  # (H, W)

        # multi-stage depth
        depth_stage1 = self.resize(depth, size=(int(self.height / self.pyramid_scale ** 2),
                                                int(self.width / self.pyramid_scale ** 2)))
        depth_stage2 = self.resize(depth, size=(int(self.height / self.pyramid_scale),
                                                int(self.width / self.pyramid_scale)))
        depth_stage3 = depth

        # make depth mask based on [min_depth, max_depth]
        depth_stage1, depth_mask_stage1 = self.mask_depth(depth_stage1,
                                                          min_depth=self.min_depth, max_depth=self.max_depth)
        depth_stage2, depth_mask_stage2 = self.mask_depth(depth_stage2,
                                                          min_depth=self.min_depth, max_depth=self.max_depth)
        depth_stage3, depth_mask_stage3 = self.mask_depth(depth_stage3,
                                                          min_depth=self.min_depth, max_depth=self.max_depth)

        # re-define sample
        del sample
        sample = {
            "images": images,                 # (V, C, H, W)
            "width": self.width,              # 640
            "height": self.height,            # 480
            "intrinsics": intrinsics,         # (V) (3, 3)
            "cam_to_world": poses,            # (V) (4, 4)
            "depth": {
                "stage1": depth_stage1,       # (H, W)
                "stage2": depth_stage2,       # (H, W)
                "stage3": depth_stage3        # (H, W)
            },
            "depth_mask": {
                "stage1": depth_mask_stage1,  # (H, W)
                "stage2": depth_mask_stage2,  # (H, W)
                "stage3": depth_mask_stage3   # (H, W)
            }
        }

        return sample

    @staticmethod
    def resize(image: torch.Tensor, size: tuple):
        resized_image = F.interpolate(image.unsqueeze(0).unsqueeze(0), size=size, mode='bilinear', align_corners=False)
        return resized_image.squeeze()

    @staticmethod
    def mask_depth(depth: torch.Tensor, min_depth: float, max_depth: float):
        depth_mask = torch.logical_and(depth >= min_depth, depth <= max_depth)
        depth[torch.logical_not(depth_mask)] = 0

        return depth, depth_mask


if __name__ == "__main__":
    # read config file
    opt = Option()
    opt.read('../config/default.yaml')

    dataset = Replica_TANDEM(split='train', option=opt)
    test_sample = dataset[10598]
    print('Have {} samples'.format(len(dataset)))
