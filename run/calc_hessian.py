import os
import sys
sys.path.insert(0, './')
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime

from util.attack import parse_attacker
from util.models import MNIST_LeNet, CIFAR10_LeNet, CIFAR10_VGG, CIFAR10_ResNet
from util.dataset import mnist, cifar10
from util.device_parser import config_visible_gpu
from util.hessian import calc_hessian_eigen_full_dataset
from util.param_parser import DictParser, IntListParser, FloatListParser, BooleanParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'mnist',
        help = 'Which dataset to use, default = "mnist".')
    parser.add_argument('--subset', type = str, default = 'train',
        help = 'Use training or test set, default = "train".')
    parser.add_argument('--batch_size', type = int, default = 500,
        help = 'The batch size, default = 500.')

    parser.add_argument('--model_type', type = str, default = 'lenet',
        help = 'The model type, default = "lenet", supported = ["lenet", "vgg", "resnet"].')
    parser.add_argument('--width', type = int, default = 8,
        help = 'The width of MNIST_LeNet, default = 8.')
    parser.add_argument('--bias', action = BooleanParser, default = True,
        help = 'Whether or not use bias term, default = True.')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded, default = None.')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file.')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None.')

    parser.add_argument('--topk', type = int, default = 1,
        help = 'The number of hessian vectors & values to calculate, default = 1.')
    parser.add_argument('--max_iter', type = int, default = 50,
        help = 'The number of maximum iterations in power iterations, default = 50.')
    parser.add_argument('--tol', type = float, default = 1e-3,
        help = 'The precise tolerence, default = 1e-3.')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default = None.')

    args = parser.parse_args()

    # Configure GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Dataset
    if args.dataset.lower() in ['mnist',]:
        train_loader, test_loader, classes = mnist(batch_size = args.batch_size)
    elif args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader, classes = cifar10(batch_size = args.batch_size)
    else:
        raise ValueError('Unrecognized dataset: %s' % args.dataset)
    loader = {'train': train_loader, 'test': test_loader}[args.subset.lower()]

    # Parse IO
    if args.out_file != None:
        dir_name = os.path.dirname(args.out_file)
        if dir_name != '' and not os.path.exists(dir_name):
            os.makedirs(dir_name)
    else:
        print('WARNING: The results will NOT be saved!')

    # Parse model
    if args.dataset.lower() in ['mnist',]:
        assert args.model_type.lower() in ['lenet',], 'For MNIST, only LeNet is supported'
        model = MNIST_LeNet(width = args.width, bias = args.bias)
    elif args.dataset.lower() in ['cifar10',]:
        if args.model_type.lower() in ['lenet',]:
            model = CIFAR10_LeNet(width = args.width, bias = args.bias)
        elif args.model_type.lower() in ['vgg',]:
            model = CIFAR10_VGG(width = args.width, bias = args.bias)
        elif args.model_type.lower() in ['resnet',]:
            model = CIFAR10_ResNet(width = args.width)
            if args.bias == True:
                print('WARNING: ResNet18 does not have bias term in its layers.')
        else:
            raise ValueError('Invalid model_type: %s' % args.model_type)
    else:
        raise ValueError('Unrecognized dataset: %s' % args.dataset)
    model = model.cuda() if use_gpu else model
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda() if use_gpu else criterion
    assert os.path.exists(args.model2load), 'File %s does not exist!' % args.model2load
    ckpt2load = torch.load(args.model2load)
    model.load_state_dict(ckpt2load)

    # Parse the optimizer
    attacker = None if args.attack == None else parse_attacker(**args.attack)

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'eigenvalue_list': None, 'eigenvec_list': None,
        'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    eigenvalue_list, eigenvec_list = calc_hessian_eigen_full_dataset(model, loader, criterion, tosave, args.out_file, use_gpu, attacker, args.topk, args.max_iter, args.tol)

