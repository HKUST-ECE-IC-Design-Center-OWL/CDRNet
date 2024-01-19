"""All semseg heads now are pointwise convolution single-layer network"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import MinkowskiEngine as ME


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


BatchNorm = nn.BatchNorm2d


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model_path = '/home/zhongad/3D_workspace/cdrnet_related_works/BPNet/initmodel/resnet18-5c106cde.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model


class SemSegResUNetDecoder2D(nn.Module):
    def __init__(self, classes=40, image_size=[640, 480]):
        super(SemSegResUNetDecoder2D, self).__init__()
        # resnet18 = torchvision.models.resnet18(pretrained=True, progress=True)  # cannot load the official constructor, since it is only raw resnet, but here we need to modify it into resnet-unet
        # resnet weights are not used, just use to init the structure here
        resnet = resnet18(pretrained=False, deep_base=False)
        # self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3
        block = BasicBlock
        layers = [2, 2, 2, 2]
        self.up3 = nn.Sequential(nn.Conv2d(80, 40, kernel_size=3, stride=1, padding=1), BatchNorm(40), nn.ReLU())
        # resnet.inplanes = 120
        resnet.inplanes = 80  # input channel of the BasicBlock instance
        self.delayer3 = resnet._make_layer(block, 40, layers[-2])
        self.up2 = nn.Sequential(nn.Conv2d(40, 20, kernel_size=3, stride=1, padding=1), BatchNorm(20), nn.ReLU())
        # resnet.inplanes = 100
        resnet.inplanes = 44
        self.delayer2 = resnet._make_layer(block, 24, layers[-3])
        self.cls = nn.Sequential(
            nn.Conv2d(24, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, classes, kernel_size=1)
        )
        self.h, self.w = image_size[1], image_size[0]

    def forward(self, backbone_feats, nyfeats):  # not yet handled feats
        [feat_quat, feat_eighth, feat_sixteenth] = backbone_feats  # channels: [24, 40, 80]
        p3 = self.up3(F.interpolate(feat_sixteenth, feat_eighth.shape[-2:], mode='bilinear', align_corners=True))
        p3 = torch.cat([p3, feat_eighth], dim=1)
        p3 = self.delayer3(p3)
        p2 = self.up2(F.interpolate(p3, feat_quat.shape[-2:], mode='bilinear', align_corners=True))
        p2 = torch.cat([p2, feat_quat], dim=1)
        p2 = self.delayer2(p2)
        x = self.cls(p2)
        x = F.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=True)

        return x, [p2, p3, feat_sixteenth]  # semseg trained 2d feat_eight, feat_quat


class SemSegHead3D(nn.Module):
    """ Predicts voxel semantic segmentation label. Just like Neucon TSDF MLP predictor."""

    def __init__(self, cfg, in_channels, multiscale=True):
        super(SemSegHead3D, self).__init__()
        self.cfg = cfg
        self.multi_scale = multiscale

        out_channels = cfg.CDR.SEMSEG_CLASS_3D
        if self.multi_scale:
            decoders = [nn.Linear(c, out_channels, 1) for c in in_channels]
        else:
            decoders = [nn.Linear(in_channels[-1], out_channels, 1)]

        self.decoders = nn.ModuleList(decoders)

    def forward(self, input_feat, stage_idx):
        """
        @param stage_idx:
        @param input_feat: extracted 3d features
        @return: pred: semseg output
        """

        # make x into 5d array
        x = input_feat

        # compute semantic voxel labels output
        pred = self.decoders[stage_idx](x)

        return pred


class SparseSemSegHead3D(nn.Module):
    """Used for 3D voxels, since it is flattened so Linear suffices."""

    def __init__(self, input, output):
        super(SparseSemSegHead3D, self).__init__()
        self.cls = ME.MinkowskiLinear(input, output, bias=True)

    def forward(self, x):
        return self.cls(x).F


class SingleSemSegHead3D(nn.Module):

    def __init__(self, input, output):
        super(SingleSemSegHead3D, self).__init__()
        self.cls = nn.Linear(input, output, bias=True)

    def forward(self, x):
        return self.cls(x)


class Semseg2DHead(nn.Module):
    """ 2D image semantic segmentation, from Atlas."""

    def __init__(self, cfg, stride):
        super(Semseg2DHead).__init__()

        self.loss_weight = cfg.HEADS2D.SEMSEG.LOSS_WEIGHT
        channels_in = cfg.BACKBONE3D.CHANNELS[0]
        self.stride = stride
        self.decoder = nn.Conv2d(channels_in,
                                 cfg.HEADS2D.SEMSEG.NUM_CLASSES,
                                 1, bias=False)

    def forward(self, x, targets=None):
        output = {}
        losses = {}

        output['semseg'] = F.interpolate(self.decoder(x),
                                         scale_factor=self.stride)

        if targets is not None and 'semseg' in targets:
            losses['semseg'] = F.cross_entropy(
                output['semseg'], targets['semseg'], ignore_index=-1
            ) * self.loss_weight

        return output, losses


class Semseg3DHead(nn.Module):
    """ From Atlas, using nn.Conv3d because it is dense there."""

    def __init__(self, cfg):
        super(Semseg3DHead, self).__init__()
        scales = len(cfg.SEMSEGDHEADS.CHANNELS)
        final_size = int(cfg.VOXEL_SIZE * 100)
        num_class = cfg.NUM_SEMSEG_CLASS

        self.voxel_size = [final_size * 2 ** i for i in range(scales)][::-1]
        self.decoders = nn.ModuleList(
            [nn.Conv3d(c, num_class, 1, bias=False) for c in cfg.SEMSEGHEADS.CHANNELS[::-1]][::-1])

    def forward(self, xs):
        for voxel_size, decoder, x in zip(self.voxel_size, self.decoders, xs):
            key = 'vol%02d_semseg' % voxel_size
            output[key] = decoder(x)
        return output


class SemSegHead2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 scale_factor=4  # since using feat_quarter as the input of the decoder
                 ):
        super(SemSegHead2D, self).__init__()
        self.scale_factor = scale_factor
        # same as nn.Linear, just permute the input then can take Linear and faster
        self.decoder = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.decoder(x)
        output = F.interpolate(x, scale_factor=self.scale_factor)
        return output


if __name__ == '__main__':
    from backbone import MnasMulti

    x = torch.rand(1, 3, 480, 640)
    alpha = float(1)
    model = MnasMulti(alpha)
    semseg_unet = SemSegResUNetDecoder2D(40)

    fpn_feats, not_yet_feats = model(x)
    x, semseg_refined_2dfeat = semseg_unet(fpn_feats, not_yet_feats)
    print('test')
