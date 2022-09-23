import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_range_sample(cur_depth, num_depth, depth_interval):
    """

    :param cur_depth: (B, H, W)
    :param num_depth: number of depth hypothesis
    :param depth_interval: (B, H, W)
    :return: depth_hypo: (B, Ndepth, H, W)
    """
    device = cur_depth.device
    dtype = cur_depth.dtype

    cur_depth_min = cur_depth - num_depth / 2 * depth_interval  # (B, H, W)
    cur_depth_max = cur_depth + num_depth / 2 * depth_interval  # (B, H, W)
    new_interval = (cur_depth_max - cur_depth_min) / (num_depth - 1)  # (B, H, W)

    depth_hypo = cur_depth_min.unsqueeze(1) + \
                 torch.arange(0, num_depth, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1) * \
                 new_interval.unsqueeze(1)  # (B, Ndepth, H, W)

    return depth_hypo


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FeatureNet(nn.Module):
    def __init__(self, base_channels):
        super(FeatureNet, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            ConvBnRelu(in_channels=3, out_channels=self.base_channels, kernel_size=3, stride=1,
                       padding=1, bias=False),
            ConvBnRelu(in_channels=self.base_channels, out_channels=self.base_channels, kernel_size=3, stride=1,
                       padding=1, bias=False)
        )

        self.conv1 = nn.Sequential(
            ConvBnRelu(in_channels=self.base_channels, out_channels=self.base_channels * 2, kernel_size=5, stride=2,
                       padding=2, bias=False),
            ConvBnRelu(in_channels=self.base_channels * 2, out_channels=self.base_channels * 2, kernel_size=3, stride=1,
                       padding=1, bias=False),
            ConvBnRelu(in_channels=self.base_channels * 2, out_channels=self.base_channels * 2, kernel_size=3, stride=1,
                       padding=1, bias=False),
        )

        self.conv2 = nn.Sequential(
            ConvBnRelu(in_channels=self.base_channels * 2, out_channels=self.base_channels * 4, kernel_size=5, stride=2,
                       padding=2, bias=False),
            ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels * 4, kernel_size=3, stride=1,
                       padding=1, bias=False),
            ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels * 4, kernel_size=3, stride=1,
                       padding=1, bias=False),
        )

        self.inner1 = ConvBnRelu(in_channels=self.base_channels, out_channels=self.base_channels * 4, kernel_size=1,
                                 stride=1, padding=0, bias=True)

        self.inner2 = ConvBnRelu(in_channels=self.base_channels * 2, out_channels=self.base_channels * 4, kernel_size=1,
                                 stride=1, padding=0, bias=True)

        self.out1 = ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.out2 = ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels * 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.out3 = ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels * 4, kernel_size=1,
                               stride=1, padding=0, bias=False)

        self.out_channels = [self.base_channels * 4, self.base_channels * 2, self.base_channels]

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)

        intra_feat3 = x3
        intra_feat2 = F.interpolate(intra_feat3, scale_factor=2, mode='nearest') + self.inner2(x2)
        intra_feat1 = F.interpolate(intra_feat2, scale_factor=2, mode='nearest') + self.inner1(x1)

        outputs = dict()
        outputs['stage1'] = self.out3(intra_feat3)  # (B, 4*C, H/4, W/4)
        outputs['stage2'] = self.out2(intra_feat2)  # (B, 2*C, H/2, W/2)
        outputs['stage3'] = self.out1(intra_feat1)  # (B, C, H, W)

        outputs['stage1'] = F.normalize(outputs['stage1'], p=2, dim=1)  # (B, 4*C, H/4, W/4)
        outputs['stage2'] = F.normalize(outputs['stage2'], p=2, dim=1)  # (B, 2*C, H/4, W/4)
        outputs['stage3'] = F.normalize(outputs['stage3'], p=2, dim=1)  # (B, C, H/4, W/4)

        return outputs


class ScalePrediction(nn.Module):
    def __init__(self):
        super(ScalePrediction, self).__init__()

        def homo_warping(src_feature, scale_hypo, depth_init,
                         src_intrinsics, src_cam_to_world,
                         ref_intrinsics, ref_cam_to_world):
            """

            :param src_feature: (B, C, H, W)
            :param scale_hypo: (B, Nscale)
            :param depth_init: (B, 1, H, W)
            :param src_intrinsics: (V-1) (B, 3, 3)
            :param src_cam_to_world: (V) (B, 4, 4)
            :param ref_intrinsics: (B, 3, 3)
            :param ref_cam_to_world: (B, 4, 4)
            :return: warped_volume: (B, C, Nscale, H, W)
            """
            device = src_feature.device
            batch, channels, height, width = list(src_feature.shape)
            num_scale = scale_hypo.shape[1]

            with torch.no_grad():
                src_world_to_cam = torch.inverse(src_cam_to_world)  # (B, 4, 4)
                ref_world_to_cam = torch.inverse(ref_cam_to_world)  # (B, 4, 4)

                src_intrinsics_ = torch.clone(src_world_to_cam)  # (B, 4, 4)
                src_intrinsics_[:, :3, :3] = src_intrinsics
                ref_intrinsics_ = torch.clone(ref_world_to_cam)  # (B, 4, 4)
                ref_intrinsics_[:, :3, :3] = ref_intrinsics

                src_proj = torch.matmul(src_intrinsics_, src_world_to_cam)
                ref_proj = torch.matmul(ref_intrinsics_, ref_world_to_cam)

                proj = torch.matmul(src_proj, torch.inverse(ref_proj))  # ref_to_src: (B, 4, 4)
                rot = proj[:, :3, :3]
                trans = proj[:, :3, 3:4]

                scale_hypo = scale_hypo.reshape(batch, 1, num_scale, 1, 1)  # (B, 1, Nscale, 1, 1)
                depth = depth_init.unsqueeze(2).repeat(1, 1, num_scale, 1, 1) * scale_hypo  # (B, 1, Nscale, H, W)

                y, x = torch.meshgrid(torch.arange(0, height, dtype=torch.float32, device=device),
                                      torch.arange(0, width, dtype=torch.float32, device=device)
                                      )
                y, x = y.contiguous(), x.contiguous()
                y, x = y.view(height * width), x.view(height * width)
                xyz = torch.stack((x, y, torch.ones_like(x)))  # (3, H * W)
                xyz = xyz.unsqueeze(0).repeat(batch, 1, 1)  # (B, 3, H * W)
                rot_xyz = torch.matmul(rot, xyz)  # (B, 3, H * W)
                rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_scale, 1) * \
                                depth.view(batch, 1, num_scale, -1)  # (B, 3, Nscale, H * W)
                proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # (B, 3, Nscale, H * W)
                proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # pixel plane: (B, 2, Nscale, H * W)

                proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
                proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
                proj_xy_normalized = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # (B, Nscale, H * W, 2)
                grid = proj_xy_normalized

            warped_volume = F.grid_sample(src_feature, grid.view(batch, num_scale * height, width, 2),
                                          mode='bilinear', padding_mode='zeros')  # (B, C, Nscale * H, W)
            warped_volume = warped_volume.view(batch, channels, num_scale, height, width)  # (B, C, Nscale, H, W)

            return warped_volume

        def compute_cost(ref_volume, warped_volume, uncertainty, type):
            """

            :param ref_volume: (B, C, Nscale, H, W)
            :param warped_volume: (B, C, Nscale, H, W)
            :param uncertainty: (B, H, W)
            :param type: choose from 'L2' or 'correlation'
            :return: cost_volume_pairwise: (B, 1, Nscale, H, W)
            """
            if type is 'L2':
                cost_volume_pairwise = torch.norm(ref_volume.sub_(warped_volume), p=2, dim=1, keepdim=True)  # L2 norm
            elif type is 'correlation':
                cost_volume_pairwise = torch.sum(ref_volume * warped_volume, dim=1, keepdim=True)  # (B, Nscale, H, W)
            else:
                raise NotImplementedError()

            if uncertainty is not None:
                num_scale = ref_volume.shape[2]
                uncertainty = F.sigmoid(uncertainty.unsqueeze(1).unsqueeze(2).repeat(1, 1, num_scale, 1, 1))
                cost_volume_pairwise = cost_volume_pairwise * uncertainty  # (B, 1, Nscale, H, W)

            return cost_volume_pairwise

        def build_cost_volume(ref_volume, warped_volumes, uncertainty):
            """

            :param ref_volume: (B, C, Nscale, H, W)
            :param warped_volumes: (V-1) (B, C, Nscale, H, W)
            :param uncertainty: (B, H, W)
            :return: cost_volume: (B, 1, Nscale, 1, 1)
            """
            num_src_views = len(warped_volumes)
            cost_volume = 0.0

            for i in range(num_src_views):
                cost_volume_pairwise = compute_cost(ref_volume, warped_volumes[i],
                                                    uncertainty, 'correlation')  # (B, 1, Nscale, H, W)
                cost_volume = cost_volume + cost_volume_pairwise

            cost_volume.div_(num_src_views)  # (B, 1, Nscale, H, W)
            cost_volume = torch.mean(cost_volume, dim=(3, 4), keepdim=True)  # (B, 1, Nscale, 1, 1)

            return cost_volume

        def compute_expectation(prob_volume, scale_hypo):
            """

            :param prob_volume: (B, Nscale)
            :param scale_hypo: (B, Nscale)
            :return: scale: (B)
            """
            scale = torch.sum(prob_volume * scale_hypo, dim=1)

            return scale

        self.homo_warping = homo_warping
        self.build_cost_volume = build_cost_volume
        self.compute_expectation = compute_expectation

    def forward(self, features, intrinsics, cam_to_world, scale_hypo, depth_init, uncertainty):
        """
                    
        :param features: (V) (B, C, H, W)
        :param intrinsics: (V) (B, 3, 3)
        :param cam_to_world: (V) (B, 4, 4)
        :param scale_hypo: (Nscale), between [min_depth, max_depth], step 0.1
        :param depth_init: (B, H, W), median depth 1
        :param uncertainty: (B, H, W), between [0, 1]
        :return: scale: (B)
        """
        num_views = len(features)
        num_scale = len(scale_hypo)
        batch = depth_init.shape[0]
        scale_hypo = scale_hypo.unsqueeze(0).repeat(batch, 1)

        ref_feature, src_feature_tuple = features[0], features[1:]
        ref_intrinsics, src_intrinsics_tuple = intrinsics[0], intrinsics[1:]
        ref_cam_to_world, src_cam_to_world_tuple = cam_to_world[0], cam_to_world[1:]

        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_scale, 1, 1)  # (B, C, Nscale, H, W)
        warped_volumes = []

        for src_feature, src_intrinsics, src_cam_to_world in zip(
                src_feature_tuple, src_intrinsics_tuple, src_cam_to_world_tuple):
            warped_volume = self.homo_warping(
                src_feature, scale_hypo, depth_init.unsqueeze(1),
                src_intrinsics, src_cam_to_world,
                ref_intrinsics, ref_cam_to_world
            )  # (B, C, Nscale, H, W)
            warped_volumes.append(warped_volume)

        cost_volume = self.build_cost_volume(ref_volume, warped_volumes, uncertainty)\
            .squeeze(1).squeeze(2).squeeze(2)  # (B, Nscale)
        prob_volume = F.softmax(cost_volume, dim=1)
        scale = self.compute_expectation(prob_volume, scale_hypo)  # (B)

        return scale


class DepthPrediction(nn.Module):
    def __init__(self):
        super(DepthPrediction, self).__init__()

        def homo_warping(src_feature, depth_hypo,
                         src_intrinsics, src_cam_to_world,
                         ref_intrinsics, ref_cam_to_world):
            """

            :param src_feature: (B, C, H, W)
            :param depth_hypo: (B, Ndepth, H, W)
            :param src_intrinsics: (V-1) (B, 3, 3)
            :param src_cam_to_world: (V) (B, 4, 4)
            :param ref_intrinsics: (B, 3, 3)
            :param ref_cam_to_world: (B, 4, 4)
            :return: warped_volume: (B, C, Ndepth, H, W)
            """
            device = src_feature.device
            batch, channels, height, width = list(src_feature.shape)
            num_depth = depth_hypo.shape[1]

            with torch.no_grad():
                src_world_to_cam = torch.inverse(src_cam_to_world)  # (B, 4, 4)
                ref_world_to_cam = torch.inverse(ref_cam_to_world)  # (B, 4, 4)

                src_intrinsics_ = torch.clone(src_world_to_cam)  # (B, 4, 4)
                src_intrinsics_[:, :3, :3] = src_intrinsics
                ref_intrinsics_ = torch.clone(ref_world_to_cam)  # (B, 4, 4)
                ref_intrinsics_[:, :3, :3] = ref_intrinsics

                src_proj = torch.matmul(src_intrinsics_, src_world_to_cam)
                ref_proj = torch.matmul(ref_intrinsics_, ref_world_to_cam)

                proj = torch.matmul(src_proj, torch.inverse(ref_proj))  # ref_to_src: (B, 4, 4)
                rot = proj[:, :3, :3]
                trans = proj[:, :3, 3:4]

                depth = depth_hypo.unsqueeze(1)  # (B, 1, Ndepth, H, W)

                y, x = torch.meshgrid(torch.arange(0, height, dtype=torch.float32, device=device),
                                      torch.arange(0, width, dtype=torch.float32, device=device)
                                      )
                y, x = y.contiguous(), x.contiguous()
                y, x = y.view(height * width), x.view(height * width)
                xyz = torch.stack((x, y, torch.ones_like(x)))  # (3, H * W)
                xyz = xyz.unsqueeze(0).repeat(batch, 1, 1)  # (B, 3, H * W)
                rot_xyz = torch.matmul(rot, xyz)  # (B, 3, H * W)
                rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * \
                                depth.view(batch, 1, num_depth, -1)  # (B, 3, Ndepth, H * W)
                proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # (B, 3, Ndepth, H * W)
                proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # pixel plane: (B, 2, Ndepth, H * W)

                proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
                proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
                proj_xy_normalized = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # (B, Ndepth, H * W, 2)
                grid = proj_xy_normalized

            warped_volume = F.grid_sample(src_feature, grid.view(batch, num_depth * height, width, 2),
                                          mode='bilinear', padding_mode='zeros')  # (B, C, Ndepth * H, W)
            warped_volume = warped_volume.view(batch, channels, num_depth, height, width)  # (B, C, Ndepth, H, W)

            return warped_volume

        def compute_cost(ref_volume, warped_volume, uncertainty, type):
            """

            :param ref_volume: (B, C, Ndepth, H, W)
            :param warped_volume: (B, C, Ndepth, H, W)
            :param uncertainty: (B, H, W)
            :param type: choose from 'L2' or 'correlation'
            :return: cost_volume_pairwise: (B, 1, Ndepth, H, W)
            """
            if type is 'L2':
                cost_volume_pairwise = torch.norm(ref_volume.sub_(warped_volume), p=2, dim=1, keepdim=True)  # L2 norm
            elif type is 'correlation':
                cost_volume_pairwise = torch.sum(ref_volume * warped_volume, dim=1, keepdim=True)  # (B, 1, Ndepth, H, W)
            else:
                raise NotImplementedError()

            if uncertainty is not None:
                num_depth = ref_volume.shape[2]
                uncertainty = F.sigmoid(uncertainty.unsqueeze(1).unsqueeze(2).repeat(1, 1, num_depth, 1, 1))
                cost_volume_pairwise = cost_volume_pairwise * uncertainty  # (B, 1, Ndepth, H, W)

            return cost_volume_pairwise

        def build_cost_volume(ref_volume, warped_volumes, uncertainty):
            """

            :param ref_volume: (B, C, Ndepth, H, W)
            :param warped_volumes: (V-1) (B, C, Ndepth, H, W)
            :param uncertainty: (B, H, W)
            :return: cost_volume: (B, 1, Ndepth, H, W)
            """
            num_src_views = len(warped_volumes)
            cost_volume_pairwises = []

            for i in range(num_src_views):
                cost_volume_pairwise = compute_cost(ref_volume, warped_volumes[i],
                                                    uncertainty, 'correlation')  # (B, 1, Ndepth, H, W)
                cost_volume_pairwises.append(cost_volume_pairwise)

            cost_volume, _ = torch.min(torch.stack(cost_volume_pairwises, dim=0), dim=0)  # (B, 1, Ndepth, H, W)

            return cost_volume

        def cost_aggregation(cost_volume, ref_feature, depth_hypo):
            """

            :param cost_volume: (B, Ndepth, H, W)
            :param ref_feature: (B, C, H, W)
            :param depth_hypo: (B, Ndepth, H, W)
            :return: cost_volume_aggregated: (B, Ndepth, H, W)
            """
            batch, num_depth, height, width = list(cost_volume.shape)
            channels = ref_feature.shape[1]

            cost_volume_neighborhood = F.unfold(input=cost_volume, kernel_size=5, stride=1,
                                                padding=2)  # (B, Ndepth * kernel_size * kernel_size, H * W)
            cost_volume_neighborhood = cost_volume_neighborhood.reshape(batch, num_depth, -1, height, width)  # (B, Ndepth, N, H, W)
            num_neighbors = cost_volume_neighborhood.shape[2]

            with torch.no_grad():
                depth_hypo_neighborhood = F.unfold(input=depth_hypo, kernel_size=5, stride=1,
                                                   padding=2)  # (B, Ndepth * kernel_size * kernel_size, H * W)
                depth_hypo_neighborhood = depth_hypo_neighborhood.reshape(batch, num_depth, num_neighbors, height, width)
                weight_depth = depth_hypo_neighborhood - depth_hypo_neighborhood[:, :, num_neighbors // 2, :, :].unsqueeze(2)
                weight_depth = F.softmax(-torch.abs(weight_depth), dim=2)  # (B, Ndepth, N, H, W)

                ref_feature_neighborhood = F.unfold(input=ref_feature, kernel_size=5, stride=1,
                                                    padding=2)  # (B, C * kernel_size * kernel_size, H * W)
                ref_feature_neighborhood = ref_feature_neighborhood.reshape(batch, channels, num_neighbors, height, width)
                # weight_feature = ref_feature_neighborhood - ref_feature_neighborhood[:, :, num_neighbors // 2, :, :].unsqueeze(2)
                # weight_feature = weight_feature.norm(p=2, dim=1, keepdim=True).repeat(1, num_depth, 1, 1, 1)  # (B, Ndepth, N, H, W)
                ref_feature_center = ref_feature_neighborhood[:, :, num_neighbors // 2, :, :].unsqueeze(2).repeat(1, 1, num_neighbors, 1, 1)
                weight_feature = torch.sum(ref_feature_center * ref_feature_neighborhood, dim=1, keepdim=True)
                weight_feature = F.softmax(weight_feature.repeat(1, num_depth, 1, 1, 1), dim=2)  # (B, Ndepth, N, H, W)

                weight = F.softmax(weight_depth * weight_feature, dim=2)  # (B, Ndepth, N, H, W)

            cost_volume_aggregated = torch.sum(cost_volume_neighborhood * weight, dim=2)  # (B, Ndepth, H, W)

            return cost_volume_aggregated

        def compute_expectation(prob_volume, depth_hypo):
            """

            :param prob_volume: (B, Ndepth, H, W)
            :param depth_hypo: (B, Ndepth, H, W)
            :return: depth: (B, H, W)
            """
            depth = torch.sum(prob_volume * depth_hypo, dim=1)

            return depth

        self.homo_warping = homo_warping
        self.build_cost_volume = build_cost_volume
        self.cost_aggregation = cost_aggregation
        self.compute_expectation = compute_expectation

    def forward(self, features, intrinsics, cam_to_world, depth_hypo, uncertainty):
        """

        :param features: (V) (B, C, H, W)
        :param intrinsics: (V) (B, 3, 3)
        :param cam_to_world: (V) (B, 4, 4)
        :param depth_hypo: (B, Ndepth, H, W)
        :param uncertainty: (B, H, W)
        :return: depth: (B, H, W)
        """
        num_views = len(features)
        num_depth = depth_hypo.size(1)

        ref_feature, src_feature_tuple = features[0], features[1:]
        ref_intrinsics, src_intrinsics_tuple = intrinsics[0], intrinsics[1:]
        ref_cam_to_world, src_cam_to_world_tuple = cam_to_world[0], cam_to_world[1:]

        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)  # (B, C, Ndepth, H, W)
        warped_volumes = []

        for src_feature, src_intrinsics, src_cam_to_world in zip(
                src_feature_tuple, src_intrinsics_tuple, src_cam_to_world_tuple):
            warped_volume = self.homo_warping(
                src_feature, depth_hypo,
                src_intrinsics, src_cam_to_world,
                ref_intrinsics, ref_cam_to_world
            )  # (B, C, Ndepth, H, W)
            warped_volumes.append(warped_volume)

        cost_volume = self.build_cost_volume(ref_volume, warped_volumes, uncertainty).squeeze()  # (B, Ndepth, H, W)

        cost_volume_aggregated = self.cost_aggregation(cost_volume, ref_feature, depth_hypo)  # (B, Ndepth, H, W)

        prob_volume = F.softmax(cost_volume_aggregated, dim=1)  # (B, Ndepth, H, W)
        depth = self.compute_expectation(prob_volume, depth_hypo)  # (B, H, W)

        return depth


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    B = 2
    N = 3
    C = 3
    H = 100
    W = 200
    Nscale = 10
    Ndepth = 10
    test_image = torch.randn(B, C, H, W)
    test_features = []
    test_intrinsics = []
    test_cam_to_world = []
    for i in range(N):
        test_features.append(torch.randn(B, C, H, W))
        test_intrinsics.append(torch.randn(B, 3, 3))
        test_cam_to_world.append(torch.randn(B, 4, 4))
    test_scale_hypo = torch.arange(Nscale, dtype=torch.float32)
    test_depth_init = torch.randn(B, H, W)
    test_depth_hypo = torch.randn(B, Ndepth, H, W)
    test_uncertainty = torch.randn(B, H, W)

    feature_extractor = FeatureNet(base_channels=8)
    feature = feature_extractor(test_image)
    print('feature stage1: {}, stage2: {}, stage3: {}'.format(feature["stage1"].shape, feature["stage2"].shape, feature["stage3"].shape))

    scale_predictor = ScalePrediction()
    scale = scale_predictor(test_features, test_intrinsics, test_cam_to_world, test_scale_hypo, test_depth_init, test_uncertainty)
    print('scale shape: {}'.format(scale.shape))

    depth_predictor = DepthPrediction()
    depth = depth_predictor(test_features, test_intrinsics, test_cam_to_world, test_depth_hypo, test_uncertainty)
    print('depth shape: {}'.format(depth.shape))
