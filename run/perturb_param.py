import os
import sys
sys.path.insert(0, './')
import json
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from datetime import datetime

from util.io import eigenvec2ckpt
from util.models import MNIST_LeNet, CIFAR10_ResNet
from util.dataset import mnist, cifar10
from util.device_parser import config_visible_gpu
from util.param_parser import DictParser, ListParser, IntListParser, FloatListParser, BooleanParser

from analysis.param_space_scan import generate_vec, param_scan

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10"')
    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The model type, default = "lenet", supported = ["lenet", "resnet"]')
    parser.add_argument('--width', type = int, default = 8,
        help = 'The width of MNIST_LeNet, default = 8')
    parser.add_argument('--bias', action = BooleanParser, default = True,
        help = 'Whether or not use bias term, default = True')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The models to be loaded as the fix point, default = None')

    parser.add_argument('--out_folder', type = str, default = None,
        help = 'The output folder')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model')

    parser.add_argument('--vec2load', type = str, default = None,
        help = 'The vector perturbation to be loaded')
    parser.add_argument('--eigenvec_idx', type = int, default = 0,
        help = 'The index of eigenvector used, default = 0')
    parser.add_argument('--vec_scale', type = float, default = 1,
        help = 'The scale of direction vector, default = 1')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default = None')

    args = parser.parse_args()

    # Configure GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Parse IO
    if args.out_folder != None and os.path.exists(args.out_folder) == False:
        os.makedirs(args.out_folder)

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

    # Parse vectors
    if args.vec2load is None:
        vec = OrderedDict({name: torch.randn_like(param) for name, param in model.named_parameters()})
        summation = sum([torch.sum(vec[key] * vec[key]) for key in vec]) ** 0.5
        vec = OrderedDict({key: vec[key] / (summation + 1e-6) for key in vec})
    else:
        if args.vec2load.endswith('ckpt'):
            vec = torch.load(args.vec2load)
        elif args.vec2load.endswith('pkl'):
            vec = eigenvec2ckpt(model = model, eigen_info_file = args.vec2load, index = args.eigenvec_idx, use_gpu = use_gpu)
        else:
            raise ValueError('Unrecognized format: %s' % args.vec2load)

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs,
        'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    for param in list(sorted(tosave['setup_config'].keys())):
        print('%s\t=>%s' % (param, tosave['setup_config'][param]))

    ori_dict = OrderedDict({name: param for name, param in model.named_parameters()})
    vec_dict = vec
    new_dict = OrderedDict()

    assert len(ori_dict.keys()) == len(vec_dict.keys()), 'The length of both dictionaries must be the same.'
    tosave_dict = model.state_dict()

    for name in vec_dict:
        tosave_dict[name] = ori_dict[name] + vec_dict[name] * args.vec_scale

    json.dump(tosave, open(os.path.join(args.out_folder, '%s.json' % args.model_name), 'w'))
    torch.save(tosave_dict, os.path.join(args.out_folder, '%s.ckpt' % args.model_name))

