from data import *
from ssd import build_ssd
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse

from drnssd import build_drnssd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


backbone_choice = ['vgg16', 'drn_d_23']

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--backbone', default='vgg16', type=str, choices=backbone_choice,
                    help='The backbone models')
parser.add_argument('--image_size', default=300, type=int,
                    help='Image size')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def forward_hook(self, input, output):
    print('{} forward\t input: {}\t output: {}\t'.format(
        self.__class__.__name__, input[0].size(), output.data.size()))


def get_featuremap_size():
    input_size = (1, 3, args.image_size, args.image_size)
    num_map = {'half': 0.5, 'quarter': 0.25}
    cfg = coco if args.dataset == 'COCO' else voc
    # cfg['min_dim'] = input_size[-1]

    if args.backbone.startswith('vgg'):
        try:
            scale = num_map[args.backbone.split('_')[1]]
        except:
            scale = 1.0
        ssd_net = build_ssd('config', input_size[-1], 21, scale)
        net = ssd_net
        for layer in net.loc.children():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(forward_hook)

    elif args.backbone.startswith('drn'):
        drnssd_net = build_drnssd('config', args.backbone, input_size[-1], 21)
        net = drnssd_net
        for layer in net.loc.children():
            layer.register_forward_hook(forward_hook)

    if args.cuda:
        net = net.cuda()

    net.train()
    input = torch.randn(input_size)

    if args.cuda:
        input = Variable(input.cuda())
    else:
        input = Variable(input)
    out = net(input)
    # print(out)


if __name__ == '__main__':
    get_featuremap_size()
