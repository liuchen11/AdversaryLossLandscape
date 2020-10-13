import os
import sys
import copy
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

from .evaluation import *

def generate_vec(model, mode = 'normalized', scale = 1., **kwargs):

    vec = OrderedDict()
    for name, param in model.named_parameters():

        if mode.lower() in ['random',]:
            vec[name] = torch.randn(param.shape, device = param.device) * scale
        elif mode.lower() in ['normalized',]:
            vec_init = torch.randn(param.shape, device = param.device).view(param.size(0), -1)
            filter_norms = param.view(param.size(0), -1).norm(p = 2, dim = 1, keepdim = True)
            vec_init = vec_init / vec_init.norm(p = 2, dim = 1, keepdim = True) * filter_norms
            vec[name] = vec_init.view(param.shape) * scale
        else:
            raise ValueError('Unrecognized mode: %s' % mode)

    return vec

def move_param(model, vec1, vec2, x1, x2):

    model_copy = copy.deepcopy(model)

    for (name, param), (name_copy, param_copy) in zip(model.named_parameters(), model_copy.named_parameters()):

        assert name == name_copy
        delta = vec1[name] * x1 if vec2 == None else vec1[name] * x1 + vec2[name] * x2
        param_copy.data = param.data + delta

    return model_copy

def param_scan_1d(model, device, attacker, loader, vec, min_pt, max_pt, step_pt, adv_calc_freq = 1):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(device) if use_gpu else criterion

    x_list = np.arange(min_pt, max_pt + 1e-8, step_pt)
    x_len = len(x_list)
    x_adv_update = np.ceil(x_len / adv_calc_freq).__int__()

    acc_value_list = []
    loss_value_list = []

    for x_adv_idx in range(x_adv_update):

        print('scanning: (%d) in the grid of (%d)' % (x_adv_idx, x_adv_update))

        # Prepare the list of model as well as loss & accuracy calculators
        base_x_idx = x_adv_idx * adv_calc_freq
        model_num_this_group = min(adv_calc_freq, len(x_list) - base_x_idx)

        model_list = [move_param(model, vec, None, x_list[base_x_idx + x_idx], None) for x_idx in range(model_num_this_group)]
        acc_calc_list = [AverageCalculator() for _ in range(model_num_this_group)]
        loss_calc_list = [AverageCalculator() for _ in range(model_num_this_group)]

        for idx, (data_batch, label_batch) in enumerate(loader, 0):

            sys.stdout.write('Instance %d\r' % idx)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            # Generate adversarial examples based on first model in each group
            optim = torch.optim.SGD(model_list[0].parameters(), lr = 1.)
            data_batch, label_batch = attacker.attack(model_list[0], optim, data_batch, label_batch, criterion)

            # Loss and Accuracy are calculated for each model
            logits_list = [model(data_batch) for model in model_list]
            loss_list = [criterion(logits, label_batch) for logits in logits_list]
            acc_list = [accuracy(logits.data, label_batch) for logits in logits_list]

            for _idx in range(model_num_this_group):
                loss_calc_list[_idx].update(loss_list[_idx].item(), data_batch.size(0))
                acc_calc_list[_idx].update(acc_list[_idx].item(), data_batch.size(0))

        acc_value_this_group = [calc.average for calc in acc_calc_list]
        loss_value_this_group = [calc.average for calc in loss_calc_list]

        acc_value_list = acc_value_list + acc_value_this_group
        loss_value_list = loss_value_list + loss_value_this_group

    return acc_value_list, loss_value_list

def param_scan_2d(model, device, attacker, loader, vec1, min_pt1, max_pt1, step_pt1, vec2, min_pt2, max_pt2, step_pt2, adv_calc_freq = 1):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(device) if use_gpu else criterion

    x1_list = np.arange(min_pt1, max_pt1 + 1e-8, step_pt1)
    x2_list = np.arange(min_pt2, max_pt2 + 1e-8, step_pt2)
    x1_len = len(x1_list)
    x2_len = len(x2_list)
    x1_adv_update = np.ceil(x1_len / adv_calc_freq).__int__()
    x2_adv_update = np.ceil(x2_len / adv_calc_freq).__int__()

    acc_value_list = []
    loss_value_list = []

    for x1_adv_idx in range(x1_adv_update):

        base_x1_idx = x1_adv_idx * adv_calc_freq
        x1_num_this_group = min(adv_calc_freq, len(x1_list) - base_x1_idx)

        acc_value_list = acc_value_list + [[] for _ in range(x1_num_this_group)]
        loss_value_list = loss_value_list + [[] for _ in range(x1_num_this_group)]

        for x2_adv_idx in range(x2_adv_update):

            print('scanning: (%d, %d) in the grid of (%d, %d)' % (x1_adv_idx, x2_adv_idx, x1_adv_update, x2_adv_update))

            base_x2_idx = x2_adv_idx * adv_calc_freq
            x2_num_this_group = min(adv_calc_freq, len(x2_list) - base_x2_idx)

            # Prepare the list of model as well as loss & accuracy calculators
            model_list = [[move_param(model, vec1, vec2, x1_list[base_x1_idx + x1_idx], x2_list[base_x2_idx + x2_idx]) \
                for x2_idx in range(x2_num_this_group)] for x1_idx in range(x1_num_this_group)]
            acc_calc_list = [[AverageCalculator() for _2 in range(x2_num_this_group)] for _1 in range(x1_num_this_group)]
            loss_calc_list = [[AverageCalculator() for _2 in range(x2_num_this_group)] for _1 in range(x1_num_this_group)]

            for idx, (data_batch, label_batch) in enumerate(loader, 0):

                sys.stdout.write('Instance %d\r' % idx)

                data_batch = data_batch.cuda(device) if use_gpu else data_batch
                label_batch = label_batch.cuda(device) if use_gpu else label_batch

                # Generate adversarial examples based on the first model in each group
                optim = torch.optim.SGD(model_list[0][0].parameters(), lr = 1.)
                data_batch, label_batch = attacker.attack(model_list[0][0], optim, data_batch, label_batch, criterion)

                logits_list = [[model_list[_1][_2](data_batch) for _2 in range(x2_num_this_group)] for _1 in range(x1_num_this_group)]
                loss_list = [[criterion(logits_list[_1][_2], label_batch) for _2 in range(x2_num_this_group)] for _1 in range(x1_num_this_group)]
                acc_list = [[accuracy(logits_list[_1][_2].data, label_batch) for _2 in range(x2_num_this_group)] for _1 in range(x1_num_this_group)]

                for _1 in range(x1_num_this_group):
                    for _2 in range(x2_num_this_group):
                        loss_calc_list[_1][_2].update(loss_list[_1][_2].item(), data_batch.size(0))
                        acc_calc_list[_1][_2].update(acc_list[_1][_2].item(), data_batch.size(0))

            acc_value_this_group = [[acc_calc_list[_1][_2].average for _2 in range(x2_num_this_group)] for _1 in range(x1_num_this_group)]
            loss_value_this_group = [[loss_calc_list[_1][_2].average for _2 in range(x2_num_this_group)] for _1 in range(x1_num_this_group)]

            for _1 in range(x1_num_this_group):
                acc_value_list[base_x1_idx + _1] = acc_value_list[base_x1_idx + _1] + acc_value_this_group[_1]
                loss_value_list[base_x1_idx + _1] = loss_value_list[base_x1_idx + _1] + loss_value_this_group[_1]

    return acc_value_list, loss_value_list

def param_scan(model, device, attacker, loader, adv_budget_list, vec1, vec2, vec1_scan, vec2_scan, tosave):

    for adv_budget in adv_budget_list:

        # Update attacker
        attacker.adjust_threshold(adv_budget)

        print('Test under adversarial budget %.3f' % adv_budget)
        print('theshold = %.3f, step_size = %.3f' % (attacker.threshold, attacker.step_size))

        if vec2 == None:        # 1d scan
            min_pt, max_pt, step_pt, adv_calc_freq = float(vec1_scan['min']), float(vec1_scan['max']), float(vec1_scan['step']), int(vec1_scan['adv_calc_freq'])
            acc_value_list, loss_value_list = param_scan_1d(model = model, device = device, attacker = attacker, loader = loader,
                vec = vec1, min_pt = min_pt, max_pt = max_pt, step_pt = step_pt, adv_calc_freq = adv_calc_freq)
            tosave['results'][adv_budget]['acc_value_list'] = acc_value_list
            tosave['results'][adv_budget]['loss_value_list'] = loss_value_list
        else:                   # 2d scan
            min_pt1, max_pt1, step_pt1, adv_calc_freq1 = float(vec1_scan['min']), float(vec1_scan['max']), float(vec1_scan['step']), int(vec1_scan['adv_calc_freq'])
            min_pt2, max_pt2, step_pt2, adv_calc_freq2 = float(vec2_scan['min']), float(vec2_scan['max']), float(vec2_scan['step']), int(vec2_scan['adv_calc_freq'])
            assert adv_calc_freq1 == adv_calc_freq2, 'adv_calc_freq should match in both dimensions, but they are %d and %d respectively' % (adv_calc_freq1, adv_calc_freq2)
            adv_calc_freq = adv_calc_freq1
            acc_value_list, loss_value_list = param_scan_2d(model = model, device = device, attacker = attacker, loader = loader, vec1 = vec1, min_pt1 = min_pt1, max_pt1 = max_pt1,
                step_pt1 = step_pt1, vec2 = vec2, min_pt2 = min_pt2, max_pt2 = max_pt2, step_pt2 = step_pt2, adv_calc_freq = adv_calc_freq)
            tosave['results'][adv_budget]['acc_value_list'] = acc_value_list
            tosave['results'][adv_budget]['loss_value_list'] = loss_value_list

    return tosave

