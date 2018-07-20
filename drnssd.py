import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os

BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    # def __init__(self, block, layers, num_classes=1000,
    #              channels=(16, 32, 64, 128, 256, 512, 512, 512),
    #              out_map=False, out_middle=False, pool_size=28, arch='D'):
    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512, 512),
                 out_middle=False, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        # self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                                   padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                          bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4], stride=2,
                                       dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], stride=2, dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(
                    channels[6], layers[6], stride=2, dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(
                    channels[7], layers[7], stride=2, dilation=1)
            # used for batch32 / 4g don't have batchnorm
            self.layer9 = None if layers[8] == 0 else \
                self._make_conv_layers(
                    channels[8], layers[8], stride=2, padding=0, dilation=1, batchnorm=False)

            # # used for batch64/4g
            # self.layer9 = None if layers[8] == 0 else \
            #     self._make_conv_layers(
            #         channels[8], layers[8], stride=2, padding=0, dilation=1)

        # if num_classes > 0:
        #     self.avgpool = nn.AvgPool2d(pool_size)
        #     self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
        #                         stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, padding=None, stride=1, dilation=1, batchnorm=True):
        modules = []
        for i in range(convs):
            if batchnorm:
                modules.extend([
                    nn.Conv2d(self.inplanes, channels, kernel_size=3,
                              stride=stride if i == 0 else 1,
                              padding=dilation if padding is None else padding, bias=False, dilation=dilation),
                    BatchNorm(channels),
                    nn.ReLU(inplace=True)])
            else:
                modules.extend([
                    nn.Conv2d(self.inplanes, channels, kernel_size=3,
                              stride=stride if i == 0 else 1,
                              padding=dilation if padding is None else padding, bias=False, dilation=dilation),
                    nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.layer9 is not None:
            x = self.layer9(x)
            y.append(x)

        # if self.out_map:
        #     x = self.fc(x)
        # else:
        #     x = self.avgpool(x)
        #     x = self.fc(x)
        #     x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


def drn_d_23(channels, out_middle=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1, 1],
                channels=channels, out_middle=out_middle, arch='D', **kwargs)
    return model


def drn_d_24(channels, out_middle=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2],
                channels=channels, out_middle=out_middle, arch='D', **kwargs)
    return model


def drn_d_39(channels, out_middle=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1, 1],
                channels=channels, out_middle=out_middle, arch='D', **kwargs)
    return model


class DRN_SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    # def __init__(self, phase, size, base, extras, head, num_classes, scale=1.0):
    def __init__(self, phase, size, base, head, cfg, num_classes, l2norm=True, scale=1.0):
        super(DRN_SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # self.cfg = (coco, voc)[num_classes == 21]
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg) if phase is not 'config' else None
        self.priors = Variable(self.priorbox.forward(),
                               volatile=True) if phase is not 'config' else None
        self.l2norm = l2norm
        self.size = size

        # SSD network
        self.drn = base
        if self.l2norm:
            # Layer learns to scale the l2 normalized features from conv4_3
            self.L2Norm = L2Norm(128, 20)
        # self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # # apply vgg up to conv4_3 relu
        # for k in range(23):
        #     x = self.drn[k](x)

        # s = self.L2Norm(x)
        # sources.append(s)

        # # apply vgg up to fc7
        # for k in range(23, len(self.vgg)):
        #     x = self.vgg[k](x)
        # sources.append(x)

        x, y = self.drn(x)

        if self.l2norm:
            s = self.L2Norm(y[3])
            sources.append(s)
            sources.extend(y[4:])
        else:
            sources.extend(y[3:])

        # # apply extra layers and cache source layer outputs
        # for k, v in enumerate(self.extras):
        #     x = F.relu(v(x), inplace=True)
        #     if k % 2 == 1:
        #         sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def multibox(cfg, predict_source, channels, num_classes):
    loc_layers = []
    conf_layers = []
    drn_source = predict_source
    for k, v in enumerate(drn_source):
        loc_layers += [nn.Conv2d(channels[v - 1],
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(channels[v - 1],
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


# drn_channels = [16, 32, 64, 128, 256, 512, 512, 512, 512]

# the index of the predict layer,such as:layer4,layer5,……,layer9
predict_source = {
    'drn_d_23': [4, 5, 6, 7, 8, 9],
    'drn_d_39': [4, 5, 6, 7, 8, 9],
}
drn_channels = {
    'drn_d_23': [16, 32, 64, 128, 256, 512, 512, 512, 512],
    'drn_d_39': [16, 32, 64, 128, 256, 512, 512, 512, 512],
}

mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '321': [4, 6, 6, 6, 4, 4],
    '512': [],
}

build_function = {
    'drn_d_23': drn_d_23,
    'drn_d_39': drn_d_39,
}


def build_drnssd(phase, arch, size=300, num_classes=21, cfg=None, l2norm=True, scale=1.0):
    if phase != "test" and phase != "train" and phase != "config":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    head_ = multibox(mbox[str(size)], predict_source[str(
        arch)], drn_channels[str(arch)], num_classes)
    base_ = build_function[str(arch)](drn_channels[str(arch)], out_middle=True)
    drnssd_cfg = None if phase == 'config' else cfg
    return DRN_SSD(phase, size, base_, head_, drnssd_cfg, num_classes, l2norm=l2norm, scale=scale)
