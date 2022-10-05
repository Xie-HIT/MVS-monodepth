#!/usr/bin/python
# -*- coding: UTF-8 -*-

import PIL.Image as pil
import matplotlib.pyplot as plt
from torchvision import transforms
import yaml
import torch
import numpy as np
import random


def read_image(image_path, size=None, crop=False):
    """

    :param image_path: path to test image
    :param size: resize to which size (H, W)
    :param crop: whether crop the center of the image
    :return: resized image
    """
    input_color = pil.open(image_path).convert('RGB')
    original_width, original_height = input_color.size
    if size is None:
        height, width = original_height, original_width
    else:
        height, width = size[0], size[1]

    left = (original_width - width) / 2
    top = (original_height - height) / 2
    right = (original_width + width) / 2
    bottom = (original_height + height) / 2

    if crop is True:
        input_color = input_color.crop((left, top, right, bottom))
    else:
        input_color = input_color.resize((width, height), pil.LANCZOS)
    input_color = transforms.ToTensor()(input_color).unsqueeze(0)

    return input_color


def show_image(input):
    if len(input.shape) is 4:
        image = input[0]
    else:
        image = input
    if image.max() > 1 or image.min() < 0:
        image = (image - image.min()) / (image.max() - image.min())
    img = transforms.ToPILImage()(image)
    img.show()


def read_depth(depth_path, scale, size=None):
    depth = pil.open(depth_path)
    original_width, original_height = depth.size
    if size is None:
        height, width = original_height, original_width
    else:
        height, width = size[0], size[1]
    depth = depth.resize((width, height), pil.LANCZOS)
    depth = transforms.ToTensor()(depth)

    # depth = (depth - depth.min()) / (depth.max() - depth.min())
    # show_image(depth)

    return depth * scale


def abs_error(pred, gt, mask=None):
    """
    https://github.com/kwea123/CasMVSNet_pl
    """
    if mask is None:
        return (pred - gt).abs()
    pred, gt = pred[mask], gt[mask]
    return (pred - gt).abs()


def acc_threshold(pred, gt, mask, threshold):
    """
    https://github.com/kwea123/CasMVSNet_pl
    computes the percentage of pixels whose error is less than @threshold
    """
    errors = abs_error(pred, gt, mask)
    acc_mask = errors < threshold
    return acc_mask.float()


class Option:
    def __init__(self):
        self.model = {}
        self.loss = {}
        self.training = {}
        self.evaluation = {}

        self.num_dataset = {
            "Training": 0,
            "Evaluation": 0,
        }

    def read(self, path):
        with open(path, 'r') as f:
            configs = yaml.safe_load_all(f.read())

            for config in configs:
                for k, v in config.items():
                    if k == 'Model':
                        self.model = v
                    elif k == 'Loss':
                        self.loss = v
                    elif k == 'Training':
                        self.training = v
                        self.num_dataset['Training'] = len(v)
                    elif k == 'Evaluation':
                        self.evaluation = v
                        self.num_dataset['Evaluation'] = len(v)
                    else:
                        raise NotImplementedError('Invalid config: {}'.format(k))
