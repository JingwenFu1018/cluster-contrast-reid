from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer
from copy import deepcopy
from .modules import AdaptiveRotatedConv2d, RountingFunction, AdaptiveRotatedScaledConv2d, AdaptiveRotatedScaledConv2d_FJW
import os.path as osp


__all__ = ['ARCResNet', 'arcresnet18', 'arcresnet34', 'arcresnet50_FJW', 'arcresnet101',
           'arcresnet152']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 n_stage=None,
                 n_block=None,
                 replace=None,
                 kernel_number=1,):
        super(Bottleneck, self).__init__()

        self.n_stage = n_stage
        self.n_block = n_block
        self.replace = replace
        self.kernel_number = kernel_number

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        if str(self.n_block) not in self.replace:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        else:
            self.conv2 = AdaptiveRotatedScaledConv2d_FJW(
                in_channels=width,
                out_channels=width,
                kernel_size=3, 
                stride=stride,
                padding=dilation,
                groups=groups,
                dilation=dilation,
                rounting_func=RountingFunction(
                    in_channels=planes,
                    kernel_number=self.kernel_number,
                ),
                kernel_number=self.kernel_number,
            )

        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if str(self.n_block) not in self.replace:
            out = self.conv2(out)
        else:
            out, alphas, angles, scales = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if str(self.n_block) not in self.replace:
            return out
        else:
            return out, alphas, angles, scales


class FJWNet222(nn.ModuleList):
    def __init__(self, list_of_layers):
        super(FJWNet222, self).__init__(list_of_layers)
        
    def forward(self, x, alphas=None, angles=None, scales=None):
        if alphas is None and angles is None and scales is None:  # Stage does not replace Conv with ARConv
            for bottleneck_block in self:
                x = bottleneck_block(x)
            return x
        else:
            assert isinstance(alphas, list)
            assert isinstance(angles, list)
            assert isinstance(scales, list)

            for bottleneck_block in self:
                x, this_alphas, this_angles, this_scales = bottleneck_block(x)  # this_alphas.shape = this_angles.shape = this_scales = [bs, num_kernels]
                alphas.append(this_alphas)
                angles.append(this_angles)
                scales.append(this_scales)

            return x, alphas, angles, scales


class ARCResNetRaw(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,
                 replace=None,
                 kernel_number=None):
        super(ARCResNetRaw, self).__init__()
        
        self.replace = replace
        self.kernel_number = kernel_number

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       n_stage=0,
                                       replace=self.replace[0], 
                                       kernel_number=self.kernel_number)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       n_stage=1,
                                       replace=self.replace[1], 
                                       kernel_number=self.kernel_number)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       n_stage=2,
                                       replace=self.replace[2], 
                                       kernel_number=self.kernel_number)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       n_stage=3,
                                       replace=self.replace[3], 
                                       kernel_number=self.kernel_number)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, n_stage=None, replace=None, kernel_number=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            n_stage=n_stage,
                            n_block=0,
                            replace=replace,
                            kernel_number=kernel_number,)) 
        self.inplanes = planes * block.expansion
        for idx_block in range(1, blocks): #这里的blocks是指一个stage里面有多少个block [3,4,6,3] 
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                n_stage=n_stage,
                                n_block=idx_block,
                                replace=replace,
                                kernel_number=kernel_number,))  

        return FJWNet222(layers)

    # def _forward_impl(self, x):
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     # x = self.fc(x)

    #     return x

    # def forward(self, x):
    #     return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    base_dir = kwargs.pop('base_dir')
    model = ARCResNetRaw(block, layers, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # state_dict = torch.load(base_dir + '/checkpoint-best.pth') 
        # model.load_state_dict(state_dict, strict=False)
        
        # def rename_state_dict_keys(state_dict):
        #     new_state_dict = {}
        #     for key, value in state_dict.items():
        #         if key.startswith('conv1'):
        #             new_key = key.replace('conv1', 'base.b1')
        #         elif key.startswith('bn1'):
        #             new_key = key.replace('bn1', 'base.b2')
        #         elif key.startswith('relu'):
        #             new_key = key.replace('relu', 'base.b3')
        #         elif key.startswith('maxpool'):
        #             new_key = key.replace('maxpool', 'base.b4')
        #         elif key.startswith('layer1'):
        #             new_key = key.replace('layer1', 'base.b5')
        #         elif key.startswith('layer2'):
        #             new_key = key.replace('layer2', 'base.b6')
        #         elif key.startswith('layer3'):
        #             new_key = key.replace('layer3', 'base.b7')
        #         elif key.startswith('layer4'):
        #             new_key = key.replace('layer4', 'base.b8')
        #         else:
        #             new_key = key
        #         new_state_dict[new_key] = value
        #     return new_state_dict


        checkpoint = torch.load(base_dir + '/checkpoint-best.pth')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']  
        else:
            state_dict = checkpoint
        # # 重命名 state_dict 中的键
        # state_dict = rename_state_dict_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)

        # print("state_dict.keys():", state_dict.keys())
        # missing_unexpected = model.load_state_dict(state_dict, strict=False)
        # print("Missing keys in model:", missing_unexpected.missing_keys)
        # print("Unexpected keys in model:", missing_unexpected.unexpected_keys)


    return model


def arc_resnet50_raw(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)





class FJWNet222(nn.ModuleList):
    def __init__(self, list_of_layers):
        super(FJWNet222, self).__init__(list_of_layers)
        
    def forward(self, x, alphas=None, angles=None, scales=None):
        if alphas is None and angles is None and scales is None:  # Stage does not replace Conv with ARConv
            for bottleneck_block in self:
                x = bottleneck_block(x)
            return x
        else:
            assert isinstance(alphas, list)
            assert isinstance(angles, list)
            assert isinstance(scales, list)

            for bottleneck_block in self:
                x, this_alphas, this_angles, this_scales = bottleneck_block(x)  # this_alphas.shape = this_angles.shape = this_scales = [bs, num_kernels]
                alphas.append(this_alphas)
                angles.append(this_angles)
                scales.append(this_scales)

            return x, alphas, angles, scales

class FJWNet(nn.ModuleList):
    def __init__(self, list_of_blocks):
        super(FJWNet, self).__init__(list_of_blocks)
        # self.b1 = b1
        # self.b2 = b2
        # self.b3 = b3
        # self.b4 = b4
        # self.b5 = b5
        # self.b6 = b6
        # self.b7 = b7
        # self.b8 = b8

    def forward(self, x):
        for idx, this_block in enumerate(self):
            if idx in [0, 1, 2, 3, 4]:
                x = this_block(x)
            elif idx == 5:
                x, alphas, angles, scales = this_block(x, alphas=[], angles=[], scales=[])
            elif idx in [6, 7]:
                x, alphas, angles, scales = this_block(x, alphas, angles, scales)
        return x, alphas, angles, scales
        # x = self.b1(x)
        # x = self.b2(x)
        # x = self.b3(x)
        # x = self.b4(x)

        # x = self.b5(x)  # becasue xFFF
        # x, alphas, angles, scales = self.b6(x, alphas=[], angles=[], scales=[])
        # x, alphas, angles, scales = self.b7(x, alphas, angles, scales)
        # x, alphas, angles, scales = self.b8(x, alphas, angles, scales)
        # return x, alphas, angles, scales

        
class ARCResNet(nn.Module):
    __factory = {
        # 18: torchvision.models.resnet18,
        # 34: torchvision.models.resnet34,
        # 50: torchvision.models.resnet50,
        # 101: torchvision.models.resnet101,
        # 152: torchvision.models.resnet152,
        50: arc_resnet50_raw,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='avg',
                 replace=None, kernel_number=None, base_dir=None):
        super(ARCResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ARCResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ARCResNet.__factory[depth](pretrained=pretrained, replace=replace, kernel_number=kernel_number, base_dir=base_dir)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        # self.base = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.base = FJWNet(
            [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
        )

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        bs = x.size(0)
        # x = self.base(x)
        x, alphas, angles, scales = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x, alphas, angles, scales

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            return bn_x, alphas, angles, scales

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x, alphas, angles, scales

        return prob, alphas, angles, scales

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


# def arcresnet18(**kwargs):
#     return ARCResNet(18, **kwargs)


# def arcresnet34(**kwargs):
#     return ARCResNet(34, **kwargs)


def arcresnet50_FJW(**kwargs):
    return ARCResNet(50, **kwargs)


# def arcresnet101(**kwargs):
#     return ARCResNet(101, **kwargs)


# def arcresnet152(**kwargs):
#     return ARCResNet(152, **kwargs)
