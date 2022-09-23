import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module import *
from mono_uncertainty import *
from mono_uncertainty.networks import DepthUncertaintyDecoder
from mono_uncertainty.monodepth2 import networks as legacy
from mono_uncertainty.monodepth2.layers import disp_to_depth


class monoMVSNet(nn.Module):
    def __init__(self, base_channels=8):
        super(monoMVSNet, self).__init__()
        self.feature = FeatureNet(base_channels=base_channels)
        self.scale_prediction = ScalePrediction()
        self.depth_range_sample = depth_range_sample
        self.depth_prediction = DepthPrediction()

        # scale for image pyramid: only implemented for 3 stages
        self.scale = {
            "stage1": 4.0,
            "stage2": 2.0,
            "stage3": 1.0
        }
        self.num_stage = len(self.scale)
        assert self.num_stage == 3

        # median scale hypothesis
        self.min_depth = 0.1
        self.max_depth = 100.0
        step = 0.1
        self.scale_hypo = torch.arange(start=self.min_depth, end=self.max_depth + step, step=step)

        # load monodepth2
        encoder = legacy.ResnetEncoder(num_layers=18, pretrained=False)
        encoder_path = os.path.join("../mono_uncertainty/models/MS/Monodepth2-Self/models/weights_19/encoder.pth")
        encoder_dict = torch.load(encoder_path)
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        encoder.eval()
        self.monodepth2_encoder = encoder

        decoder = DepthUncertaintyDecoder(encoder.num_ch_enc, num_output_channels=1, uncert=True, dropout=False)
        decoder_path = os.path.join("../mono_uncertainty/models/MS/Monodepth2-Self/models/weights_19/depth.pth")
        decoder_dict = torch.load(decoder_path)
        decoder.load_state_dict(decoder_dict)
        decoder.eval()
        self.monodepth2_decoder = decoder

        # TODO: Can we have different size without re-training monodepth2 ?
        self.height = encoder_dict['height']
        self.width = encoder_dict['width']

    def forward(self, images, intrinsics, cam_to_world):
        """

        :param images: (B, N, C, H, W)
        :param intrinsics: (V) (B, 3, 3)
        :param cam_to_world: (V) (B, 4, 4)
        :return:
        """
        # run monodepth2
        with torch.no_grad():
            ref_image = images[:, 0, :]
            monodepth2_output = self.monodepth2_decoder(self.monodepth2_encoder(ref_image))
            pred_disp, pred_depth = disp_to_depth(monodepth2_output[("disp", 0)], self.min_depth, self.max_depth)
            pred_depth = pred_depth / pred_depth.median()
            pred_uncert = torch.exp(monodepth2_output[("uncert", 0)])
            pred_uncert = (pred_uncert - torch.min(pred_uncert)) / (torch.max(pred_uncert) - torch.min(pred_uncert))

            # show_image(pred_disp)
            # show_image(pred_uncert)
            # plt.imsave(os.path.join('../test', 'disp.png'), pred_disp[0, 0].numpy(), cmap='magma')
            # plt.imsave(os.path.join('../test', 'uncert.png'), pred_uncert[0, 0].numpy(), cmap='hot')

        # feature extraction: FPN
        features = []
        for i in range(images.shape[1]):
            image = images[:, i]
            feature = self.feature(image)
            features.append(feature)

        # coarse to fine
        for stage_idx in range(self.num_stage):
            stage = 'stage{}'.format(stage_idx + 1)

            # feature for current stage: (V) (B, C, H, W)
            features_stage = [feature[stage] for feature in features]
            height_stage = features_stage[0].shape[2]
            width_stage = features_stage[0].shape[3]

            # intrinsic for current stage: (V) (B, 3, 3)
            num_views = len(intrinsics)
            intrinsics_stage = []
            stage_scale = self.scale[stage]
            for view_idx in range(num_views):
                intrinsic_stage = intrinsics[view_idx].clone()  # (B, 3, 3)
                intrinsic_stage[:, :2, :2] = intrinsic_stage[:, :2, :2] / stage_scale
                intrinsic_stage[:, :2, 2:3] = intrinsic_stage[:, :2, 2:3] / stage_scale
                intrinsics_stage.append(intrinsic_stage)

            # scale prediction
            if stage == 'stage1':
                pred_depth_stage = F.interpolate(pred_depth, size=(height_stage, width_stage), mode='bilinear')
                pred_uncert_stage = F.interpolate(pred_uncert, size=(height_stage, width_stage), mode='bilinear')
                scale = self.scale_prediction(features_stage, intrinsics_stage, cam_to_world, self.scale_hypo,
                                              pred_depth_stage.squeeze(1), pred_uncert_stage.squeeze(1))

            # adaptive depth hypothesis

            # depth prediction


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    import PIL.Image as pil
    import matplotlib.pyplot as plt
    from torchvision import transforms

    def read_image(image_path, size=None):
        """

        :param image_path: path to test image
        :param size: resize to which size
        :return: resized image
        """
        input_color = pil.open(image_path).convert('RGB')
        original_height, original_width = input_color.size
        if size is None:
            height, width = original_height, original_width
        else:
            height, width = size[0], size[1]
        input_color = input_color.resize((width, height), pil.LANCZOS)
        input_color = transforms.ToTensor()(input_color).unsqueeze(0)

        return input_color

    def show_image(input):
        if len(input.shape) is 4:
            image = input[0]
        else:
            image = input
        img = transforms.ToPILImage()(image)
        img.show()

    network = monoMVSNet(base_channels=8)

    B = 1
    N = 3
    image_path = '../test/test_image.jpg'
    test_images = []
    test_intrinsics = []
    test_cam_to_world = []
    for i in range(N):
        test_image = read_image(image_path, (network.height, network.width))
        # show_image(test_image)
        test_images.append(test_image)
        test_intrinsics.append(torch.randn(B, 3, 3))
        test_cam_to_world.append(torch.randn(B, 4, 4))
    test_images = torch.stack(test_images, dim=1)
    C, H, W = test_images.shape[2:]

    network(test_images, test_intrinsics, test_cam_to_world)
