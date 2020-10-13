import os
import sys
sys.path.insert(0, './')
import argparse
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime

from util.attack import parse_attacker
from util.models import MNIST_LeNet, CIFAR10_ResNet, Curve_MNIST_LeNet, Curve_CIFAR10_ResNet
from util.curves import curve_scan
from util.dataset import mnist, cifar10
from util.optim_parser import parse_optim
from util.device_parser import config_visible_gpu
from util.param_parser import DictParser, ListParser, IntListParser, FloatListParser, BooleanParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10"')
    parser.add_argument('--batch_size', type = int, default = 100,
        help = 'The batch size, default = 100')

    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The model type, default = "lenet", supported = ["lenet", "resnet"]')
    parser.add_argument('--width', type = int, default = 8,
        help = 'The width of MNIST_LeNet, default = 8')
    parser.add_argument('--bias', action = BooleanParser, default = True,
        help = 'Whether or not use bias term, default = True')
    parser.add_argument('--fix_points', action = IntListParser, default = None,
        help = 'The fix points flag in the curve, 0 means unfixed and non-zero means fixed.')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded as the fix point, default = None')

    parser.add_argument('--curve_type', type = str, default = 'bezier',
        help = 'The type of the curve, default = "bezier".')

    parser.add_argument('--out_folder', type = str, default = None,
        help = 'The output folder')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model')

    parser.add_argument('--step_size', type = float, default = 0.02,
        help = 'The size of the step, default = 0.02')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None.')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default = None')

    args = parser.parse_args()

    # Configure GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Dataset and model
    fix_points = [pt != 0 for pt in args.fix_points]
    criterion = nn.CrossEntropyLoss()
    if args.dataset.lower() in ['mnist',]:
        train_loader, test_loader, classes = mnist(batch_size = args.batch_size)
        if args.model_type.lower() in ['lenet',]:
            model = Curve_MNIST_LeNet(fix_points, width = args.width, bias = args.bias)
        else:
            raise ValueError('Unrecognized model type: %s' % args.model_type)
    elif args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader, classes = cifar10(batch_size = args.batch_size)
        if args.model_type.lower() in ['resnet',]:
            model = Curve_CIFAR10_ResNet(fix_points, width = args.width)
        else:
            raise ValueError('Unrecognized model type: %s' % args.model_type)
    else:
        raise ValueError('Invalid dataset: %s' % args.dataset)

    model = model.cuda() if use_gpu else model
    criterion = criterion.cuda() if use_gpu else criterion

    ckpt2load = torch.load(args.model2load)
    model.load_state_dict(ckpt2load)

    # Parse IO
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Parse the attacker and t_list
    attacker = None if args.attack == None else parse_attacker(**args.attack)
    t_list = np.arange(0, 1 + args.step_size, args.step_size)

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'train_loss': {}, 'test_loss': {}, 'train_acc': {}, 'test_acc':{},
        'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    curve_scan(model = model, curve_type = args.curve_type, t_list = t_list, train_loader = train_loader, test_loader = test_loader, attacker = attacker,
        out_folder = args.out_folder, model_name = args.model_name, device = device, criterion = criterion, tosave = tosave)

