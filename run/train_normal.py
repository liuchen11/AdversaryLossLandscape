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
from util.train import vanilla_train
from util.dataset import mnist, cifar10
from util.seq_parser import continuous_seq
from util.optim_parser import parse_optim
from util.device_parser import config_visible_gpu
from util.param_parser import DictParser, IntListParser, FloatListParser, BooleanParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10"')
    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'The batch size, default = 128')
    parser.add_argument('--epoch_num', type = int, default = 200,
        help = 'The number of epochs, default = 200')
    parser.add_argument('--epoch_ckpts', action = IntListParser, default = [],
        help = 'The checkpoint epoch, default = []')

    parser.add_argument('--model_type', type = str, default = 'lenet',
        help = 'The model type, default = "lenet", supported = ["lenet", "vgg", "resnet"]')
    parser.add_argument('--width', type = int, default = 8,
        help = 'The width of the model, default = 8')
    parser.add_argument('--bias', action = BooleanParser, default = True,
        help = 'Whether or not use bias term, default = True')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded, default = None')

    parser.add_argument('--out_folder', type = str, default = None,
        help = 'The output folder')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model')
    parser.add_argument('--optim', action = DictParser, default = {'name': 'sgd', 'lr': 1e-1, 'momentum': 0.9, 'weight_decay': 1e-4},
        help = 'The optimizer, default is name=sgd,lr=1e-1,momentum=0.9,weight_decay=1e-4')
    parser.add_argument('--lr_schedule', action = DictParser, default = {'name': 'constant', 'start_v': 1e-1},
        help = 'The learning rate schedule, default is name=constant,start_v=1e-1')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None.')
    parser.add_argument('--attack_threshold_schedule', action = DictParser, default = None,
        help = 'The adversarial budget schedule, default = None')

    parser.add_argument('--schedule_update_mode', type = str, default = 'epoch',
        help = 'the schdule unit of the scheduler, default = "epoch"')

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
        train_batches = (50000 - 1) // args.batch_size + 1
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
        train_batches = (60000 - 1) // args.batch_size + 1
        if args.model_type.lower() in ['lenet',]:
            model = MNIST_LeNet(width = args.width, bias = args.bias)
        else:
            raise ValueError('Invalid model_type: %s' % args.model_type)
    else:
        raise ValueError('Invalid dataset: %s' % args.dataset)
    model = model.cuda() if use_gpu else model
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda() if use_gpu else criterion
    if args.model2load is not None:
        assert os.path.exists(args.model2load), 'File %s does not exist!' % args.model2load
        ckpt2load = torch.load(args.model2load)
        model.load_state_dict(ckpt2load)

    # Parse the optimizer
    optimizer = parse_optim(policy = args.optim, params = model.parameters())
    lr_func = continuous_seq(**args.lr_schedule) if args.lr_schedule != None else None
    eps_func = continuous_seq(**args.attack_threshold_schedule) if args.attack_threshold_schedule != None else None

    # Parse the optimizer
    attacker = None if args.attack == None else parse_attacker(**args.attack)

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'train_loss': {}, 'train_acc': {}, 'test_loss': {}, 'test_acc': {},
        'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    for param in list(sorted(tosave['setup_config'].keys())):
        print('%s\t=>%s' % (param, tosave['setup_config'][param]))

    vanilla_train(model = model, train_loader = train_loader, test_loader = test_loader, attacker = attacker,
        epoch_num = args.epoch_num, epoch_ckpts = args.epoch_ckpts, train_batches = train_batches, optimizer = optimizer,
        lr_func = lr_func, eps_func = eps_func, schedule_update_mode = args.schedule_update_mode, out_folder = args.out_folder,
        model_name = args.model_name, device = device, criterion = criterion, tosave = tosave, mask = None)



