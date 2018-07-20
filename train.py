from collections import OrderedDict
import shutil

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

from drnssd import build_drnssd
from layers import *
import math

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--backbone', default='vgg16', type=str,
                    help='The backbone models')
parser.add_argument('--image_size', default=300, type=int,
                    help='The size of the input image')
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

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

featuremap_list = []


def forward_hook(self, input, output):
    print('{} forward\t input: {}\t output: {}\t'.format(
        self.__class__.__name__, input[0].size(), output.data.size()))
    featuremap_list.append(input[0].size(-1))


def get_locfeaturemap_size(net, input_size=(300, 300)):
    hook_list = []
    size = (1, 3, input_size[0], input_size[1])
    input = torch.randn(size)
    if args.cuda:
        net = net.cuda()
        input = Variable(input.cuda())
    else:
        input = Variable(input)

    for layer in net.loc.children():
        hook = layer.register_forward_hook(forward_hook)
        hook_list.append(hook)

    print('The feature map size of the predict layers')
    output = net(input)
    # print(featuremap_list)

    for hook in hook_list:
        hook.remove()


def build_model(phase, cfg, input_size=(300, 300)):
    cfg['min_dim'] = input_size[-1]
    num_map = {'half': 0.5, 'quarter': 0.25}
    print('build {}_ssd'.format(args.backbone))
    if args.backbone.startswith('vgg'):
        try:
            scale = num_map[args.backbone.split('_')[1]]
        except:
            scale = 1.0
        net = build_ssd(phase, cfg['min_dim'], cfg['num_classes'], cfg,
                        scale=scale)
    elif args.backbone.startswith('drn'):
        net = build_drnssd(
            phase, args.backbone, cfg['min_dim'], cfg['num_classes'], cfg)

    if phase == 'config':
        get_locfeaturemap_size(net, input_size)
        cfg['feature_maps'] = featuremap_list
        # cfg['steps'] = [math.ceil(cfg['min_dim'] / v)
        #                 for v in cfg['feature_maps']]
        # cfg['steps'] = [cfg['min_dim'] / v for v in cfg['feature_maps']]
        return cfg
    else:
        return net


def train():
    # Copy the python script to job_dir.
    py_file = os.path.abspath(__file__)
    shutil.copy(py_file, args.save_folder)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    input_size = (args.image_size, args.image_size)

    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        cfg['min_dim'] = input_size[-1]
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        cfg['min_dim'] = input_size[-1]
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom()

    # cfg = build_model('config', cfg, input_size)
    print('The config item is: ')
    print(cfg)
    print('')
    ssd_net = build_model('train', cfg, input_size)
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = False

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        pretrained_weights = torch.load(
            args.save_folder.split('/')[0] + '/' + args.basenet)
        print('Loading base network...')
        if args.backbone.startswith('vgg'):
            ssd_net.vgg.load_state_dict(pretrained_weights)

            # random initialize the weights
            # ssd_net.vgg.apply(weights_init)

            # # load the personal vgg16 pretrained model
            # model_dict = ssd_net.vgg.state_dict()
            # pretrained_dict = OrderedDict()
            # # for k, v in vgg_weights['state_dict'].items():
            # for k, v in vgg_weights.items():
            #     if k.startswith('features'):
            #         name = k[16:]
            #         pretrained_dict[name] = v

            # pretrained_dict = {k: v for k,
            #                    v in pretrained_dict.items() if k in model_dict}

            # model_dict.update(pretrained_dict)
            # ssd_net.vgg.load_state_dict(model_dict)

            # # load the ssd.pytoch's author's vgg16 model except the last conv6 and conv7
            # # which initialize throug the xavier method
            # model_dict = ssd_net.vgg.state_dict()
            # pretrained_dict = OrderedDict()
            # for k, v in vgg_weights.items():
            #     if int(k.split('.')[0]) <= 28:
            #         pretrained_dict[k] = v
            # model_dict.update(pretrained_dict)
            # ssd_net.vgg.load_state_dict(model_dict)
        elif args.backbone.startswith('drn'):
            model_dict = ssd_net.drn.state_dict()
            pretrained_weights = {k: v for k,
                                  v in pretrained_weights.items() if k in model_dict}
            model_dict.update(pretrained_weights)
            ssd_net.drn.load_state_dict(model_dict)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        if args.backbone.startswith('vgg'):
            ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Epoch_size: {}batch/epoch'.format(epoch_size))
    epoch_number = cfg['max_iter'] * args.batch_size // len(dataset)
    print('Epoch: {}'.format(epoch_number))
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    end = time.time()
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        # images, targets = next(batch_iterator)

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        data_time.update(time.time() - end)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % 10 == 0:
            # print('timer: %.4f sec.' % (t1 - t0))
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                  'iter ' + repr(iteration) + '/(%d) || Loss: %.4f ||' %
                  (cfg['max_iter'], loss.data[0]),
                  'timer: %.4f sec\t' % (t1 - t0),
                  'batch: %.3f(%.3f)' % (batch_time.val, batch_time.avg),
                  'data: %.3f(%.3f)' % (data_time.val, data_time.avg)
                  )

        if iteration % 100 == 0 and args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(
            ), '{}/ssd300_{}_{}.pth'.format(args.save_folder, args.dataset, iteration))

    torch.save(ssd_net.state_dict(),
               args.save_folder + '/' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu().numpy(),
        Y=torch.zeros((1, 3)).cpu().numpy(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu().numpy() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]
                       ).unsqueeze(0).cpu().numpy() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu().numpy(),
            Y=torch.Tensor([loc, conf, loc + conf]
                           ).unsqueeze(0).cpu().numpy() / epoch_size,
            win=window1,
            update=True
        )


if __name__ == '__main__':
    train()
