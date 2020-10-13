import os
import sys
sys.path.insert(0, './')
import pickle
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from datetime import datetime

from util.io import eigenvec2ckpt
from util.attack import parse_attacker
from util.evaluation import *
from util.models import MNIST_LeNet, CIFAR10_ResNet
from util.dataset import mnist, cifar10
from util.device_parser import config_visible_gpu
from util.param_parser import DictParser, ListParser, IntListParser, FloatListParser, BooleanParser

from analysis.param_space_scan import generate_vec, param_scan

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
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The models to be loaded as the fix point, default = None')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None.')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default = None')

    args = parser.parse_args()

    # Configure GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Parse IO
    out_folder = os.path.dirname(args.out_file)
    if out_folder != '' and not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Parse model
    if args.dataset.lower() in ['mnist',]:
        train_loader, test_loader, classes = mnist(batch_size = args.batch_size, shuffle = False, data_augmentation = False)
        assert args.model_type.lower() in ['lenet',], 'For MNIST, only LeNet is supported'
        model = MNIST_LeNet(width = args.width, bias = args.bias)
    elif args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader, classes = cifar10(batch_size = args.batch_size, shuffle = False, data_augmentation = False)
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

    # Parse the attacker
    attacker = None if args.attack == None else parse_attacker(**args.attack)

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'train_adv': None, 'test_adv': None,
        'log': {'cmd': 'python' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    # Generate adversary
    model.eval()
    train_adv = []
    test_adv = []

    acc_calculator = AverageCalculator()

    print('Scan the training set')
    acc_calculator.reset()
    for idx, (data_batch, label_batch) in enumerate(train_loader, 0):

        sys.stdout.write('Instance %d\r' % idx)

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        if attacker != None:
            optimizer = torch.optim.SGD(model.parameters(), lr = 1.)
            data_batch, label_batch = attacker.attack(model, optimizer, data_batch, label_batch, criterion)

        logits = model(data_batch)
        acc = accuracy(logits.data, label_batch)
        acc_calculator.update(acc.item(), data_batch.size(0))

        data_batch = data_batch.reshape(data_batch.size(0), -1)
        data_batch = data_batch.data.cpu().numpy()
        train_adv.append(data_batch)
    print('Train Accuracy: %.2f%%' % (acc_calculator.average * 100.))

    print('Scan the test set')
    acc_calculator.reset()
    for idx, (data_batch, label_batch) in enumerate(test_loader, 0):

        sys.stdout.write('Instance %d\r' % idx)

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        if attacker != None:
            optimizer = torch.optim.SGD(model.parameters(), lr = 1.)
            data_batch, label_batch = attacker.attack(model, optimizer, data_batch, label_batch, criterion)

        logits = model(data_batch)
        acc = accuracy(logits.data, label_batch)
        acc_calculator.update(acc.item(), data_batch.size(0))

        data_batch = data_batch.reshape(data_batch.size(0), -1)
        data_batch = data_batch.data.cpu().numpy()
        test_adv.append(data_batch)
    print('Test Accuracy: %.2f%%' % (acc_calculator.average * 100.))

    tosave['train_adv'] = np.concatenate(train_adv, axis = 0)
    tosave['test_adv'] = np.concatenate(test_adv, axis = 0)

    pickle.dump(tosave, open(args.out_file, 'wb'))
