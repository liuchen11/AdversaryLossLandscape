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
from util.dataset import mnist, cifar10
from util.train import curve_train
from util.seq_parser import continuous_seq
from util.optim_parser import parse_optim
from util.device_parser import config_visible_gpu
from util.param_parser import DictParser, ListParser, IntListParser, FloatListParser, BooleanParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10"')
    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'The batch size, default = 128')
    parser.add_argument('--epoch_num', type = int, default = 20,
        help = 'The number of epochs, default = 200')

    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The model type, default = "lenet", supported = ["lenet", "resnet"]')
    parser.add_argument('--width', type = int, default = 8,
        help = 'The width of MNIST_LeNet, default = 8')
    parser.add_argument('--bias', action = BooleanParser, default = True,
        help = 'Whether or not use bias term, default = True')
    parser.add_argument('--fix_points', action = IntListParser, default = None,
        help = 'The fix points flag in the curve, 0 means unfixed and non-zero means fixed.')
    parser.add_argument('--model2load', action = ListParser, default = None,
        help = 'The models to be loaded as the fix point, default = None')

    parser.add_argument('--curve_type', type = str, default = 'bezier',
        help = 'The type of the curve, default = "bezier".')

    parser.add_argument('--out_folder', type = str, default = None,
        help = 'The output folder')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model')
    parser.add_argument('--optim', action = DictParser, default = {'name': 'sgd', 'lr': 1e-1, 'momentum': 0.9, 'weight_decay': 1e-6},
        help = 'The optimizer, default is name=sgd,lr=1e-1,momentum=0.9,weight_decay=1e-6')
    parser.add_argument('--lr_schedule', action = DictParser, default = None,
        help = 'The learning rate schedule, default is None')

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
        train_batches = 60000 // args.batch_size
        if args.model_type.lower() in ['lenet',]:
            model = Curve_MNIST_LeNet(fix_points, width = args.width, bias = args.bias)
            base_model_func_raw = lambda : MNIST_LeNet(width = args.width, bias = args.bias)
        else:
            raise ValueError('Unrecognized model type: %s' % args.model_type)
    elif args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader, classes = cifar10(batch_size = args.batch_size)
        train_batches = 50000 // args.batch_size
        if args.model_type.lower() in ['resnet',]:
            model = Curve_CIFAR10_ResNet(fix_points, width = args.width)
            base_model_func_raw = lambda : CIFAR10_ResNet(width = args.width)
        else:
            raise ValueError('Unrecognized model type: %s' % args.model_type)
    else:
        raise ValueError('Invalid dataset: %s' % args.dataset)

    if use_gpu == True:
        model = model.cuda()
        criterion = criterion.cuda()
        base_model_func = lambda: base_model_func_raw().cuda()
    else:
        base_model_func = base_model_func_raw
    for idx, fix_model in enumerate(args.model2load):
        if fix_model == '' or fix_points[idx] == False:
            continue
        base_model = base_model_func()
        ckpt2load = torch.load(fix_model)
        base_model.load_state_dict(ckpt2load)
        model.load_points(base_model, idx)
    model.init()

    # Parse IO
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Parse optimizer and attacker
    optimizer = parse_optim(policy = args.optim, params = model.parameters())
    lr_func = None if args.lr_schedule == None else continuous_seq(**args.lr_schedule)
    attacker = None if args.attack == None else parse_attacker(**args.attack)

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'train_loss': {}, 'train_acc': {}, 'test_loss': {}, 'test_acc': {},
        'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    for param in list(sorted(tosave['setup_config'].keys())):
        print('%s\t=>%s' % (param, tosave['setup_config'][param]))

    curve_train(model = model, curve_type = args.curve_type, train_loader = train_loader, test_loader = test_loader, train_batches = train_batches,
        attacker = attacker, epoch_num = args.epoch_num, optimizer = optimizer, lr_func = lr_func, out_folder = args.out_folder,
        model_name = args.model_name, device = device, criterion = criterion, tosave = tosave)

