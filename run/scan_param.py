import os
import sys
sys.path.insert(0, './')
import json
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime

from util.io import eigenvec2ckpt
from util.attack import parse_attacker
from util.models import MNIST_LeNet, CIFAR10_LeNet, CIFAR10_VGG, CIFAR10_ResNet
from util.dataset import mnist, cifar10
from util.seq_parser import discrete_seq
from util.device_parser import config_visible_gpu
from util.param_scanner import generate_vec, param_scan
from util.param_parser import DictParser, IntListParser, FloatListParser, BooleanParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset to use, default = "cifar10"')
    parser.add_argument('--batch_size', type = int, default = 100,
        help = 'The batch size, default = 100')
    parser.add_argument('--subset', type = str, default = 'train',
        help = 'Which subset is used, default = "train"')

    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The type of the model, default = "resnet"')
    parser.add_argument('--width', type = int, default = 16,
        help = 'The width of MNIST_LeNet, default = 16')
    parser.add_argument('--bias', type = BooleanParser, default = True,
        help = 'Whether or not to use bias term, default = True')

    parser.add_argument('--out_folder', type = str, default = None,
        help = 'The output folder')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model')

    parser.add_argument('--attack', action = DictParser,
        default = {'step_size': 2, 'threshold': 8, 'iter_num': 10, 'order': -1},
        help = 'Play adversarial attack or not, default = step_size=2,threshold=8,iter_num=10,order=-1')
    parser.add_argument('--adv_budget_list', action = DictParser, default = None,
        help = 'The list of adversarial budget used, default = None.')

    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded, default = None')

    parser.add_argument('--vec1_scan', action = DictParser,
        default = {'min': -1., 'max': 1. , 'step': 0.05, 'adv_calc_freq': 1},
        help = 'The configuration of vec1, default = min=-1.,max=1.,step=0.1,adv_calc_freq=1')
    parser.add_argument('--vec2_scan', action = DictParser, default = None,
        help = 'The configuration of vec2, default = None')
    parser.add_argument('--load_vec1', type = str, default = None,
        help = 'The file to load vec1, default = None')
    parser.add_argument('--load_vec2', type = str, default = None,
        help = 'The file to load vec2, default = None')
    parser.add_argument('--vec_sample_mode', type = str, default = 'normalized',
        help = 'The way to generate random directions, default = "normalized"')
    parser.add_argument('--vec_scale', type = float, default = 1.,
        help = 'The scale of direction vector, default = 1.')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default = None')

    args = parser.parse_args()

    # Configure GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Parse IO
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Parse model and dataset
    if args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader, classes = cifar10(batch_size = args.batch_size)
        loader = {'train': train_loader, 'test': test_loader}[args.subset]
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
    elif args.dataset.lower() in ['mnist',]:
        train_loader, test_loader, classes = mnist(batch_size = args.batch_size)
        loader = {'train': train_loader, 'test': test_loader}[args.subset]
        if args.model_type.lower() in ['lenet',]:
            model = MNIST_LeNet(width = args.width, bias = args.bias)
        else:
            raise ValueError('Invalid model_type: %s' % args.model_type)
    else:
        raise ValueError('Invalid dataset: %s' % args.dataset)
    model = model.cuda() if use_gpu else model
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda() if use_gpu else criterion
    assert os.path.exists(args.model2load), 'File %s does not exist!' % args.model2load
    ckpt2load = torch.load(args.model2load)
    model.load_state_dict(ckpt2load)

    # Parse vectors
    if args.load_vec1 is None:
        vec1 = generate_vec(model = model, mode = args.vec_sample_mode, scale = args.vec_scale)
        print('vec1 is generated!')
    else:
        if args.load_vec1.endswith('ckpt'):
            vec1 = torch.load(args.load_vec1)
        elif args.load_vec1.endswith('pkl'):
            vec1 = eigenvec2ckpt(model = model, eigen_info_file = args.load_vec1, index = 0, use_gpu = use_gpu)
        else:
            raise ValueError('Unrecognized format: %s' % args.load_vec1)
        print('vec1 is loaded!')

    if args.load_vec2 is None and args.vec2_scan != None:
        vec2 = generate_vec(model = model, mode = args.vec_sample_mode, scale = args.vec_scale)
        print('vec2 is generated!')
    elif args.vec2_scan != None:
        if args.load_vec2.endswith('ckpt'):
            vec2 = torch.load(args.load_vec2)
        elif args.load_vec2.endswith('pkl'):
            vec2 = eigenvec2ckpt(model = model, eigen_info_file = args.load_vec2, index = 1, use_gpu = use_gpu)
        else:
            raise ValueError('Unrecognized format: %s' % args.load_vec2)
        print('vec2 is loaded!')
    else:
        vec2 = None

    print('1D scanning' if vec2 is None else '2D scanning')

    # Parse the attacker
    attacker = parse_attacker(**args.attack)
    adv_budget_list = [attacker.threshold,] if args.adv_budget_list is None else discrete_seq(**args.adv_budget_list)

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'results': {adv_budget: {} for adv_budget in adv_budget_list},
        'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    tosave = param_scan(model = model, device = device, attacker = attacker, loader = loader, adv_budget_list = adv_budget_list,
        vec1 = vec1, vec2 = vec2, vec1_scan = args.vec1_scan, vec2_scan = args.vec2_scan, tosave = tosave)

    vec1_file = args.load_vec1 if args.load_vec1 != None else os.path.join(args.out_folder, '%s_vec1.ckpt' % args.model_name)
    vec2_file = None if args.vec2_scan == None else os.path.join(args.out_folder, '%s_vec2.ckpt' % args.model_name) if args.load_vec2 is None else args.load_vec2
    tosave['vec1_file'] = vec1_file
    tosave['vec2_file'] = vec2_file

    if args.load_vec1 is None:
        torch.save(vec1, vec1_file)
    if args.load_vec2 is None and args.vec2_scan != None:
        torch.save(vec2, vec2_file)

    json.dump(tosave, open(os.path.join(args.out_folder, '%s_results.json' % args.model_name), 'w'))


