import os
import sys
sys.path.insert(0, './')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from external.auto_attack.autopgd_pt import APGDAttack
from external.auto_attack.fab_pt import FABAttack
from external.auto_attack.square_pt import SquareAttack

h_message = '''
>>> PGD(step_size, threshold, iter_num, order = np.inf)
>>> APGD(threshold, iter_num, rho, loss_type, alpha = 0.75, order = np.inf)
>>> Square(threshold, window_size_factor, iter_num, order = np.inf)
'''

def parse_attacker(name, **kwargs):

    if name.lower() in ['h', 'help']:
        print(h_message)
        exit(0)
    elif name.lower() in ['pgd',]:
        return PGDForTest(**kwargs)
    elif name.lower() in ['apgd',]:
        return APGD(**kwargs)
    elif name.lower() in ['square',]:
        return Square(**kwargs)
    else:
        raise ValueError('Unrecognized name of the attacker: %s' % name)

def project(ori_pt, threshold, order = np.inf):
    '''
    Project the data into a norm ball

    >>> ori_pt: the original point
    >>> threshold: maximum norms allowed
    >>> order: norm used
    '''

    if order in [np.inf,]:
        prj_pt = torch.clamp(ori_pt, min = - threshold, max = threshold) 
    elif order in [2,]:
        ori_shape = ori_pt.size()
        pt_norm = torch.norm(ori_pt.view(ori_shape[0], -1), dim = 1, p = 2)
        pt_norm_clip = torch.clamp(pt_norm, max = threshold)
        prj_pt = ori_pt.view(ori_shape[0], -1) / (pt_norm.view(-1, 1) + 1e-8) * (pt_norm_clip.view(-1, 1) + 1e-8)
        prj_pt = prj_pt.view(ori_shape)
    else:
        raise ValueError('Invalid norms: %s' % order)

    return prj_pt

class PGD(object):

    def __init__(self, step_size, threshold, iter_num, order = np.inf):

        self.step_size = step_size if step_size < 1. else step_size / 255.
        self.threshold = threshold if threshold < 1. else threshold / 255.
        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf

        self.meta_threshold = self.threshold
        self.meta_step_size = self.step_size

        print('Create a PGD attack')
        print('step_size = %1.2e, threshold = %1.2e, iter_num = %d, order = %f' % (
            self.step_size, self.threshold, self.iter_num, self.order))

    def adjust_threshold(self, threshold):

        threshold = threshold if threshold < 1. else threshold / 255.

        self.step_size = self.meta_step_size * threshold / (self.meta_threshold + 1e-6)
        self.threshold = threshold

        print('Attacker adjusted, threshold = %1.2e, step_size = %1.2e' % (self.threshold, self.step_size))

    def attack(self, model, optim, data_batch, label_batch, criterion):

        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        if self.threshold < 1e-6:
            return data_batch, label_batch

        ori_batch = data_batch.detach()

        # Initial perturbation
        step_size = self.step_size
        noise = project(ori_pt = (torch.rand(data_batch.shape, device = device) * 2 - 1) * step_size, threshold = self.threshold, order = self.order)
        data_batch = torch.clamp(data_batch + noise, min = 0., max = 1.)
        data_batch = data_batch.detach().requires_grad_()

        for iter_idx in range(self.iter_num):

            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            _, prediction = logits.max(dim = 1)
            indicator_vec = (prediction == label_batch).float()

            loss.backward()
            grad = data_batch.grad.data

            step_size = self.step_size
            if self.order == np.inf:
                next_point = data_batch + step_size * torch.sign(grad)
            elif self.order == 2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = step_size * (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8)
                next_point = data_batch + perb.view(ori_shape)
            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = self.threshold, order = self.order)
            next_point = torch.clamp(next_point, min = 0., max = 1.)

            data_batch = next_point.detach().requires_grad_()

            model.zero_grad()

        return data_batch, label_batch

    def attack_list(self, model_list, optim, data_batch, label_batch, criterion):

        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        if self.threshold < 1e-6:
            return data_batch, label_batch

        ori_batch = data_batch.detach()

        # Initial perturbation
        step_size = self.step_size
        noise = project(ori_pt = (torch.rand(data_batch.shape, device = device) * 2 - 1) * step_size, threshold = self.threshold, order = self.order)
        data_batch = torch.clamp(data_batch + noise, min = 0., max = 1.)
        data_batch = data_batch.detach().requires_grad_()

        for iter_idx in range(self.iter_num):

            prob_sum = 0.
            for model in model_list:
                logits = model(data_batch)
                prob_sum = prob_sum + F.softmax(logits)

            prob = prob_sum / len(model_list)
            loss = - torch.log(prob).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1).mean()
            _, prediction = prob.max(dim = 1)
            indicator_vec = (prediction == label_batch).float()

            loss.backward()
            grad = data_batch.grad.data

            step_size = self.step_size

            if self.order == np.inf:
                next_point = data_batch + step_size * torch.sign(grad)
            elif self.order == 2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = step_size * (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8)
                next_point = data_batch + perb.view(ori_shape)
            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = self.threshold, order = self.order)
            next_point = torch.clamp(next_point, min = 0., max = 1.)

            data_batch = next_point.detach().requires_grad_()

            for model in model_list:
                model.zero_grad()

        return data_batch, label_batch

    def attack_curve(self, model, optim, data_batch, label_batch, criterion, coeffs):

        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        if self.threshold < 1e-6:
            return data_batch, label_batch

        ori_batch = data_batch.detach()

        # Initial perturbation
        step_size = self.step_size
        noise = project(ori_pt = (torch.rand(data_batch.shape, device = device) * 2 - 1) * step_size, threshold = self.threshold, order = self.order)
        data_batch = torch.clamp(data_batch + noise, min = 0., max = 1.)
        data_batch = data_batch.detach().requires_grad_()

        for iter_idx in range(self.iter_num):

            logits = model(data_batch, coeffs)
            loss = criterion(logits, label_batch)
            _, prediction = logits.max(dim = 1)
            indicator_vec = (prediction == label_batch).float()

            loss.backward()
            grad = data_batch.grad.data

            step_size = self.step_size
            if self.order == np.inf:
                next_point = data_batch + step_size * torch.sign(grad)
            elif self.order ==2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = step_size * (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8)
                next_point = data_batch + perb.view(ori_shape)
            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = self.threshold, order = self.order)
            next_point = torch.clamp(next_point, min = 0., max = 1.)

            data_batch = next_point.detach().requires_grad_()

            optim.zero_grad()

        return data_batch, label_batch

class APGD(object):

    def __init__(self, threshold, iter_num, rho, loss_type = 'ce', alpha = 0.75, order = np.inf):

        self.order = order if order > 0 else np.inf
        self.threshold = threshold if threshold < 1. or self.order != np.inf else threshold / 255.
        self.step_size = self.threshold * 2
        self.iter_num = int(iter_num)
        self.rho = rho
        self.alpha = alpha
        self.loss_type = loss_type

        self.meta_threshold = self.threshold
        self.meta_step_size = self.step_size

        print('Create a Auto-PGD attacker')
        print('step_size = %1.2e, threshold = %1.2e, iter_num = %d, rho = %.4f, alpha = %.4f, order = %f' % (
            self.step_size, self.threshold, self.iter_num, self.rho, self.alpha, self.order))
        print('loss type = %s' % self.loss_type)

    def adjust_threshold(self, threshold):

        threshold = threshold if threshold < 1. or self.order != np.inf  else threshold / 255.

        self.step_size = self.meta_step_size * threshold / (self.meta_threshold + 1e-6)
        self.threshold = threshold

    def attack(self, model, optim, data_batch, label_batch, criterion):

        norm = {np.inf: 'Linf', 2: 'L2'}[self.order]

        attacker = APGDAttack(model, n_restarts = 5, n_iter = self.iter_num, verbose=False, eps = self.threshold,
            norm = norm, eot_iter = 1, rho = self.rho, seed = time.time(), loss = self.loss_type, device = data_batch.device)

        _, adv_data_batch = attacker.perturb(data_batch, label_batch, cheap = True)

        return adv_data_batch.detach(), label_batch

class Square(object):

    def __init__(self, threshold, window_size_factor, iter_num, order = np.inf):

        self.order = order if order > 0 else np.inf
        self.threshold = threshold if threshold < 1. or self.order != np.inf else threshold / 255.
        self.window_size_factor = window_size_factor
        self.iter_num = int(iter_num)

        print('Create a Square attacker')
        print('threshold = %1.2e, window_size_factor = %d, iter_num = %d, order = %s' % (
            self.threshold, self.window_size_factor, self.iter_num, self.order))

    def adjust_threshold(self, threshold):

        threshold = threshold if threshold < 1. or self.order != np.inf else threshold / 255.
        self.threshold = threshold
        
    def attack(self, model, optim, data_batch, label_batch, criterion):

        norm = {np.inf: 'Linf', 2: 'L2'}[self.order]

        attacker = SquareAttack(model, p_init = 0.8, n_queries = self.iter_num, eps = self.threshold, norm = norm,
            n_restarts = 1, seed = time.time(), verbose = False, device = data_batch.device, resc_schedule = False)

        adv_data_batch = attacker.perturb(data_batch, label_batch)

        return adv_data_batch.detach(), label_batch
