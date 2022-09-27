import sys
import os

import PIL.ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module import *
from thirdparty import *
from thirdparty.networks import DepthUncertaintyDecoder
from thirdparty.monodepth2 import networks as legacy
from thirdparty.monodepth2.layers import disp_to_depth
from utils.utils import *


class monoMVSNet(nn.Module):
    def __init__(self, opt):
        super(monoMVSNet, self).__init__()
        self.feature = FeatureNet(base_channels=opt.model['FeatureNet_base_channels'])
        self.scale_prediction = ScalePrediction()
        self.depth_range_sample = depth_range_sample
        self.depth_prediction = DepthPrediction()

        # scale for image pyramid: only implemented for 3 stages
        self.scale = {
            "stage1": opt.model['pyramid_scale'] * opt.model['pyramid_scale'],
            "stage2": opt.model['pyramid_scale'],
            "stage3": 1.0
        }
        self.num_stage = len(self.scale)
        assert self.num_stage == 3

        # scale hypothesis
        self.min_depth = opt.model['min_depth']
        self.max_depth = opt.model['max_depth']
        step = opt.model['scale_step']
        self.scale_hypo = torch.arange(start=self.min_depth, end=self.max_depth + step, step=step)

        # depth hypothesis
        self.num_depth = opt.model['num_depth']
        self.depth_interval = opt.model['depth_interval']
        self.depth_interal_ratio = opt.model['depth_interal_ratio']

        # load monodepth2
        encoder = legacy.ResnetEncoder(num_layers=18, pretrained=False)
        encoder_path = opt.model['monodepth2_encoder_path']
        encoder_dict = torch.load(encoder_path)
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        encoder.eval()
        self.monodepth2_encoder = encoder

        decoder = DepthUncertaintyDecoder(encoder.num_ch_enc, num_output_channels=1, uncert=True, dropout=False)
        decoder_path = opt.model['monodepth2_decoder_path']
        decoder_dict = torch.load(decoder_path)
        decoder.load_state_dict(decoder_dict)
        decoder.eval()
        self.monodepth2_decoder = decoder

        # TODO: Can we change size without re-training monodepth2 ?
        #  ->   NO! You should keep same with training size while evaluation, and interpolate to original size later
        self.monodepth2_height = encoder_dict['height']  # 192
        self.monodepth2_width = encoder_dict['width']  # 640
        self.height = opt.model['height']  # 512
        self.width = opt.model['width']  # 640

    def forward(self, images, intrinsics, cam_to_world):
        """

        :param images: (B, N, C, H, W)
        :param intrinsics: (V) (B, 3, 3)
        :param cam_to_world: (V) (B, 4, 4)
        :return: scale: (B)  depth_stage1: (B, H, W)  depth_stage2: (B, H, W)  depth_stage3: (B, H, W)
        """
        # run monodepth2
        with torch.no_grad():
            ref_image = images[:, 0, :]
            ref_image = F.interpolate(ref_image, size=(self.monodepth2_height, self.monodepth2_width), mode='bilinear')
            monodepth2_output = self.monodepth2_decoder(self.monodepth2_encoder(ref_image))
            pred_disp, pred_depth = disp_to_depth(monodepth2_output[("disp", 0)], self.min_depth, self.max_depth)
            pred_uncert = torch.exp(monodepth2_output[("uncert", 0)])
            pred_uncert = (pred_uncert - torch.min(pred_uncert)) / (torch.max(pred_uncert) - torch.min(pred_uncert))
            pred_disp = F.interpolate(pred_disp, size=(self.height, self.width), mode='bilinear')
            pred_depth = F.interpolate(pred_depth, size=(self.height, self.width), mode='bilinear')
            pred_uncert = F.interpolate(pred_uncert, size=(self.height, self.width), mode='bilinear')

            # show_image(pred_disp)
            # show_image(pred_uncert)
            # plt.imsave(os.path.join('../test', 'test_disp.png'), pred_disp[0, 0].numpy(), cmap='magma')
            # plt.imsave(os.path.join('../test', 'test_uncert.png'), pred_uncert[0, 0].numpy(), cmap='hot')

        # feature extraction: FPN
        features = []
        for i in range(images.shape[1]):
            image = images[:, i]
            feature = self.feature(image)
            features.append(feature)

        # coarse to fine
        output = {}
        depth = None
        for stage_idx in range(self.num_stage):
            stage = "stage{}".format(stage_idx + 1)

            # feature for current stage: (V) (B, C, H, W)
            features_stage = [feature[stage] for feature in features]
            height_stage = features_stage[stage_idx].shape[2]
            width_stage = features_stage[stage_idx].shape[3]

            # intrinsic for current stage: (V) (B, 3, 3)
            num_views = len(intrinsics)
            intrinsics_stage = []
            stage_scale = self.scale[stage]
            for view_idx in range(num_views):
                intrinsic_stage = intrinsics[view_idx].clone()  # (B, 3, 3)
                intrinsic_stage[:, :2, :2] = intrinsic_stage[:, :2, :2] / stage_scale
                intrinsic_stage[:, :2, 2:3] = intrinsic_stage[:, :2, 2:3] / stage_scale
                intrinsics_stage.append(intrinsic_stage)

            # scale prediction and depth hypothesis
            if stage == 'stage1':
                pred_depth_stage = F.interpolate(pred_depth, size=(height_stage, width_stage), mode='bilinear').squeeze(1)
                pred_depth_stage = pred_depth_stage / pred_depth_stage.median()
                pred_uncert_stage = F.interpolate(pred_uncert, size=(height_stage, width_stage), mode='bilinear',
                                                  align_corners=True).squeeze(1)
                scale = self.scale_prediction(features_stage, intrinsics_stage, cam_to_world, self.scale_hypo,
                                              pred_depth_stage, pred_uncert_stage)  # (B)
                output["scale"] = scale
                cur_depth = scale * pred_depth_stage
            else:
                cur_depth = depth.detach()
                cur_depth = F.interpolate(cur_depth.unsqueeze(1), size=(height_stage, width_stage), mode='bilinear').squeeze(1)
                pred_uncert_stage = F.interpolate(pred_uncert, size=(height_stage, width_stage), mode='bilinear',
                                                  align_corners=True).squeeze(1)

            depth_interval = (2 - pred_uncert_stage) / 2 * self.depth_interval * self.depth_interal_ratio[stage_idx]
            depth_hypo = self.depth_range_sample(cur_depth, self.num_depth[stage_idx],
                                                 depth_interval, self.min_depth, self.max_depth)

            # depth prediction: (B, H, W)
            depth = self.depth_prediction(features_stage, intrinsics_stage, cam_to_world, depth_hypo, pred_uncert_stage)
            output["depth_stage{}".format(stage_idx + 1)] = depth

        return output


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    import PIL.Image as pil
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # read config file
    opt = Option()
    opt.read('../config/default.yaml')

    network = monoMVSNet(opt)

    B = 1
    N = 3
    image_path = '../test/test_image3.jpg'
    test_images = []
    test_intrinsics = []
    test_cam_to_world = []
    for i in range(N):
        test_image = read_image(image_path, size=(network.height, network.width))
        # show_image(test_image)
        test_images.append(test_image)
        test_intrinsics.append(torch.eye(3).unsqueeze(0).repeat(B, 1, 1))
        test_cam_to_world.append(torch.eye(4).unsqueeze(0).repeat(B, 1, 1))
    test_images = torch.stack(test_images, dim=1)
    C, H, W = test_images.shape[2:]

    test_output = network(test_images, test_intrinsics, test_cam_to_world)

    print("========== shape ==========")
    print("scale: {}".format(test_output['scale'].shape))
    print("depth_stage1: {}".format(test_output['depth_stage1'].shape))
    print("depth_stage2: {}".format(test_output['depth_stage2'].shape))
    print("depth_stage3: {}".format(test_output['depth_stage3'].shape))

    print("========== value ==========")
    print("scale: {}".format(test_output['scale'].detach()))
    test_disp = 10 / test_output['depth_stage3'].detach()
    show_image(test_disp)
    # plt.imsave(os.path.join('../test', 'test_disp.png'), test_disp[0].numpy(), cmap='magma')
