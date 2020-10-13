import os
import sys
sys.path.insert(0, './')
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime

from util.dataset import mnist, cifar10
from util.param_parser import DictParser, ListParser, IntListParser, FloatListParser, BooleanParser

def compare_adversary(train_ori_data, test_ori_data, adv1_info, adv2_info):

    train_adv1_data = adv1_info['train_adv']
    train_adv2_data = adv2_info['train_adv']
    test_adv1_data = adv1_info['test_adv']
    test_adv2_data = adv2_info['test_adv']

    train_delta1_data = train_adv1_data - train_ori_data
    train_delta2_data = train_adv2_data - train_ori_data
    test_delta1_data = test_adv1_data - test_ori_data
    test_delta2_data = test_adv2_data - test_ori_data

    train_delta1_sign_data = np.sign(train_delta1_data)
    train_delta2_sign_data = np.sign(train_delta2_data)
    test_delta1_sign_data = np.sign(test_delta1_data)
    test_delta2_sign_data = np.sign(test_delta2_data)

    # File1
    train_delta1_norm = np.linalg.norm(train_delta1_data, axis = 1)
    test_delta1_norm = np.linalg.norm(test_delta1_data, axis = 1)
    train_delta1_sign_norm = np.linalg.norm(np.sign(train_delta1_data), axis = 1)
    test_delta1_sign_norm = np.linalg.norm(np.sign(test_delta1_data), axis = 1)

    # File2
    train_delta2_norm = np.linalg.norm(train_delta2_data, axis = 1)
    test_delta2_norm = np.linalg.norm(test_delta2_data, axis = 1)
    train_delta2_sign_norm = np.linalg.norm(np.sign(train_delta2_data), axis = 1)
    test_delta2_sign_norm = np.linalg.norm(np.sign(test_delta2_data), axis = 1)

    # Compare
    train_cosine = np.sum(train_delta1_data * train_delta2_data, axis = 1) / train_delta1_norm / train_delta2_norm
    test_cosine = np.sum(test_delta1_data * test_delta2_data, axis = 1) / test_delta1_norm / test_delta2_norm
    train_sign_cosine = np.sum(train_delta1_sign_data * train_delta2_sign_data, axis = 1) / train_delta1_sign_norm / train_delta2_sign_norm
    test_sign_cosine = np.sum(test_delta1_sign_data * test_delta2_sign_data, axis = 1) / test_delta1_sign_norm / test_delta2_sign_norm

    print('The cosine similarity in the training set: %.4f' % (train_cosine.mean()))
    print('The cosine similarity in the test set: %.4f' % (test_cosine.mean()))
    print('The signed cosine similarity in the training set: %.4f' % (train_sign_cosine.mean()))
    print('The signed cosine similarity in the test set: %.4f' % (test_sign_cosine.mean()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 100,
        help = 'The batch size, default = 100')
    parser.add_argument('--dataset', type = str, default = 'mnist',
        help = 'The dataset we use')

    parser.add_argument('--folder', type = str, default = None,
        help = 'The folder to be scanned')

    args = parser.parse_args()

    if args.dataset.lower() in ['mnist',]:
        train_loader, test_loader, classes = mnist(batch_size = args.batch_size, shuffle = False, data_augmentation = False)
    elif args.dataset.lower() in ['cifar10',]:
        train_loader, test_loader, classes = cifar10(batch_size = args.batch_size, shuffle = False, data_augmentation = False)
    else:
        raise ValueError('Unrecognized dataset: %s' % args.dataset)

    train_ori_data = []
    test_ori_data = []

    for idx, (data_batch, label_batch) in enumerate(train_loader, 0):
        data_batch = data_batch.reshape(data_batch.size(0), -1)
        train_ori_data.append(data_batch.data.cpu().numpy())
    for idx, (data_batch, label_batch) in enumerate(test_loader, 0):
        data_batch = data_batch.reshape(data_batch.size(0), -1)
        test_ori_data.append(data_batch.data.cpu().numpy())

    train_ori_data = np.concatenate(train_ori_data, axis = 0)
    test_ori_data = np.concatenate(test_ori_data, axis = 0)

    adv_info_list = []
    adv_f_list = []
    for f in os.listdir(args.folder):
        if os.path.isfile(args.folder + os.sep + f) and f.endswith('pkl'):
            adv_info = pickle.load(open(args.folder + os.sep + f, 'rb'))
            adv_info_list.append(adv_info)
            adv_f_list.append(args.folder + os.sep + f)

    adv_info_list_len = len(adv_info_list)
    for idx1 in range(adv_info_list_len):
        for idx2 in range(idx1 + 1, adv_info_list_len):
            print('File 1 = %s' % adv_f_list[idx1])
            print('File 2 = %s' % adv_f_list[idx2])

            compare_adversary(train_ori_data, test_ori_data, adv1_info = adv_info_list[idx1], adv2_info = adv_info_list[idx2])


