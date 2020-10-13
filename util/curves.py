import os
import sys
import math
import json
from scipy.special import comb

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluation import *
from .utility import update_bn_curve

def poly_chain(t, pt_num):

    weight_list = [0,] * pt_num
    seg_index = int(t * (pt_num - 1)) % (pt_num - 1)

    weight_p2 = t * (pt_num - 1) - seg_index
    weight_p1 = 1. - weight_p2

    weight_list[seg_index] = weight_p1
    weight_list[seg_index] = weight_p2

    return weight_list

def bezier_curve(t, pt_num):

    return [(1. - t) ** (pt_num - 1 - idx) * t ** idx * comb(pt_num - 1, idx) for idx in range(pt_num)]

def curve_scan(model, curve_type, t_list, train_loader, test_loader, attacker,
        out_folder, model_name, device, criterion, tosave):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    for t_idx, t in enumerate(t_list):

        if curve_type.lower() in ['poly_chain',]:
            coeffs = poly_chain(t, pt_num = model.num_bends)
        elif curve_type.lower() in ['bezier_curve', 'bezier']:
            coeffs = bezier_curve(t, pt_num = model.num_bends)
        else:
            raise ValueError('Unrecognized curve type: %s' % curve_type)

        model.train()
        update_bn_curve(model, coeffs, train_loader, attacker, criterion, use_gpu)
        model.eval()

        acc_calculator = AverageCalculator()
        loss_calculator = AverageCalculator()

        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):

            sys.stdout.write('Instance Idx: %d\r' % idx)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            if attacker != None:
                optimizer = torch.optim.SGD(model.parameters(), lr = 1.)
                data_batch, label_batch = attacker.attack_curve(model, optimizer, data_batch, label_batch, criterion, coeffs)

            logits = model(data_batch, coeffs)
            loss = criterion(logits, label_batch)
            acc = accuracy(logits.data, label_batch)

            loss_calculator.update(loss.item(), data_batch.size(0))
            acc_calculator.update(acc.item(), data_batch.size(0))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Training Set: t = %.2f, loss = %.4f, acc = %.2f%%' % (t, loss_this_epoch, acc_this_epoch * 100.))
        tosave['train_loss'][t] = loss_this_epoch
        tosave['train_acc'][t] = acc_this_epoch

        acc_calculator.reset()
        loss_calculator.reset()

        for idx, (data_batch, label_batch) in enumerate(test_loader, 0):

            sys.stdout.write('Instance Idx: %d\r' % idx)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            if attacker != None:
                optimizer = torch.optim.SGD(model.parameters(), lr = 1.)
                data_batch, label_batch = attacker.attack_curve(model, optimizer, data_batch, label_batch, criterion, coeffs)

            logits = model(data_batch, coeffs)
            loss = criterion(logits, label_batch)
            acc = accuracy(logits.data, label_batch)

            loss_calculator.update(loss.item(), data_batch.size(0))
            acc_calculator.update(acc.item(), data_batch.size(0))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Test Set: t = %.2f, loss = %.4f, acc = %.2f%%' % (t, loss_this_epoch, acc_this_epoch * 100.))
        tosave['test_loss'][t] = loss_this_epoch
        tosave['test_acc'][t] = acc_this_epoch

        json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))

