import os
import sys
sys.path.insert(0, './')
import argparse
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime

from util.attack import parse_attacker
from util.models import MNIST_LeNet, CIFAR10_LeNet, CIFAR10_VGG, CIFAR10_ResNet
from util.train import attack_list
from util.dataset import mnist, cifar10
from util.device_parser import config_visible_gpu
from util.param_parser import DictParser, ListParser, FloatListParser, BooleanParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10"')
    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'The batch size, default = 128')

    parser.add_argument('--model_type', type = str, default = 'lenet',
        help = 'The model type, default = "lenet", supported = ["lenet", "vgg", "resnet"]')
    parser.add_argument('--width', type = int, default = 8,
        help = 'The width of the model, default = 8')
    parser.add_argument('--bias', action = BooleanParser, default = True,
        help = 'Whether or not use bias term, default = True')
    parser.add_argument('--model2load', action = ListParser, default = None,
        help = 'The models to be loaded, default = None')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file, default = None')
    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default = None')

    args = parser.parse_args()

    # Configure GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Parse IO
    if args.out_file != None:
        out_dir = os.path.dirname(args.out_file)
        if out_dir != '' and os.path.exists(out_dir) == False:
           os.makedirs(out_dir)

    # Parse model and dataset
    if args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader, classes = cifar10(batch_size = args.batch_size)
        if args.model_type.lower() in ['lenet',]:
            make_model = lambda: CIFAR10_LeNet(width = args.width, bias = args.bias)
        elif args.model_type.lower() in ['vgg',]:
            make_model = lambda: CIFAR10_VGG(width = args.width, bias = args.bias)
        elif args.model_type.lower() in ['resnet',]:
            make_model = lambda: CIFAR10_ResNet(width = args.width)
            if args.bias == True:
                print('WARNING: ResNet18 does not have bias term in its layers.')
        else:
            raise ValueError('Invalid model_type: %s' % args.model_type)
    elif args.dataset.lower() in ['mnist',]:
        train_loader, test_loader, classes = mnist(batch_size = args.batch_size)
        if args.model_type.lower() in ['lenet',]:
            make_model = lambda: MNIST_LeNet(width = args.width, bias = args.bias)
        else:
            raise ValueError('Invalid model_type: %s' % args.model_type)
    else:
        raise ValueError('Invalid dataset: %s' % args.dataset)
    model_list = []
    for file2load in args.model2load:
        model = make_model()
        model = model.cuda() if use_gpu else model
        ckpt2load = torch.load(file2load)
        model.load_state_dict(ckpt2load)
        model_list.append(model)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda() if use_gpu else criterion

    # Parse the optimizer
    attacker = None if args.attack == None else parse_attacker(**args.attack)
    optimizer = None

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'train_acc': None, 'train_loss': None, 'test_acc': None, 'test_loss': None,
        'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    for param in list(sorted(tosave['setup_config'].keys())):
        print('%s\t=>%s' % (param, tosave['setup_config'][param]))

    attack_list(model_list = model_list, loader = test_loader, attacker = attacker, optimizer = None, out_file = args.out_file,
        device = device, criterion = criterion, tosave = tosave)
