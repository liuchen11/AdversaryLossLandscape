import torch
import torch.nn as nn

from collections import OrderedDict

from .models import CurveBatchNorm2d

## Tensor Operation
def group_add(x1_list, mul1, x2_list, mul2):
    '''
    >>> group summation: x1 * mul1 + x2 * mul2
    '''

    return [x1 * mul1 + x2 * mul2 for x1, x2 in zip(x1_list, x2_list)]

def group_product(x1_list, x2_list):
    '''
    >>> x1_list, x2_list: the list of tensors to be multiplied

    >>> group dot product
    '''

    return sum([torch.sum(x1 * x2) for x1, x2 in zip(x1_list, x2_list)])

def group_normalize(v_list):
    '''
    >>> normalize the tensor list to make them joint l2 norm be 1
    '''

    summation = group_product(v_list, v_list)
    summation = summation ** 0.5
    v_list = [v / (summation + 1e-6) for v in v_list]

    return v_list

def get_param(model):
    '''
    >>> return the parameter list
    '''

    return [param for param in model.parameters()]

def get_param_grad(model):
    '''
    >>> return the parameter and gradient list
    '''
    param_list = []
    grad_list = []
    for param in model.parameters():
        if param.grad is None:
            continue
        param_list.append(param)
        grad_list.append(param.grad)
    return param_list, grad_list

## Model Operation
def distance_between_ckpts(ckpt1, ckpt2):
    '''
    >>> Calculate the distance ckpt2 - ckpt1
    '''

    assert len(ckpt1) == len(ckpt2), 'The length of ckpt1 should be the same as ckpt2'
    key_list = ckpt1.keys()

    distance_dict = OrderedDict()
    for key in key_list:
        param1 = ckpt1[key]
        param2 = ckpt2[key]
        distance_dict[key] = param2.data - param1.data

    return distance_dict

## Update BN
def reset_bn_stats(module):
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.reset_running_stats()

def get_bn_momenta(module, momenta):
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        momenta[module] = module.momentum

def set_bn_momenta(module, momenta):
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.momentum = momenta[module]

def update_bn(model, loader, attacker, criterion, use_gpu):

    device = torch.device('cpu' if not use_gpu else 'cuda:0')

    model.train()
    momenta = {}
    model.apply(reset_bn_stats)
    model.apply(lambda module: get_bn_momenta(module, momenta))
    instance_num = 0

    for idx, (data_batch, label_batch) in enumerate(loader, 0):

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        if attacker != None:
            optimizer = torch.optim.SGD(model.parameters(), lr = 1.)
            data_batch, label_batch = attacker.attack(model, optimizer, data_batch, label_batch, criterion)

        batch_size = data_batch.data.size(0)
        momentum = batch_size / (instance_num + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(data_batch)
        instance_num += batch_size

    model.apply(lambda module: set_bn_momenta(module, momenta))

## Update BN in Curve Model
def reset_bn_stats_curve(module):
    if isinstance(module, CurveBatchNorm2d):
        module.reset_running_stats()

def get_bn_momenta_curve(module, momenta):
    if isinstance(module, CurveBatchNorm2d):
        momenta[module] = module.momentum

def set_bn_momenta_curve(module, momenta):
    if isinstance(module, CurveBatchNorm2d):
        module.momentum = momenta[module]    

def update_bn_curve(model, coeffs, loader, attacker, criterion, use_gpu):

    device = torch.device('cpu' if not use_gpu else 'cuda:0')

    model.train()
    momenta = {}
    model.apply(reset_bn_stats_curve)
    model.apply(lambda module: get_bn_momenta_curve(module, momenta))
    instance_num = 0

    for idx, (data_batch, label_batch) in enumerate(loader, 0):

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        if attacker != None:
            optimizer = torch.optim.SGD(model.parameters(), lr = 1.)
            data_batch, label_batch = attacker.attack_curve(model, optimizer, data_batch, label_batch, criterion, coeffs)

        batch_size = data_batch.data.size(0)
        momentum = batch_size / (instance_num + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(data_batch, coeffs)
        instance_num += batch_size

    model.apply(lambda module: set_bn_momenta_curve(module, momenta))


