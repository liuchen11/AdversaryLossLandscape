import sys
import math
import pickle
import numpy as np

import torch
import torch.nn as nn

from .utility import group_add, group_product, group_normalize, get_param, get_param_grad

def calc_hessian_eigen_full_dataset(model, loader, criterion, tosave, out_file,  use_gpu, attacker = None, topk = 1, max_iter = 50, tol = 1e-3):
    '''
    >>> calculate the top eigenvalues of model parameters

    >>> model: the model studied
    >>> criterion: the loss function used
    >>> use_gpu: Boolean, whether or not to use GPU
    >>> attacker: Attacker
    >>> topk: Int, the number of top eigenvalues and eigen vector calculated
    >>> max_iter: Int, the maximum iterations allowed
    >>> tol: float, the precision tolerence
    '''

    device = torch.device('cuda:0' if use_gpu == True else 'cpu')

    # Dataset
    data_batch_list = []
    label_batch_list = []
    batch_size = None
    for data_batch, label_batch in loader:
        data_batch = data_batch.cuda() if use_gpu else data_batch
        label_batch = label_batch.cuda() if use_gpu else label_batch
        if batch_size is None:
            batch_size = data_batch.size(0)
        if data_batch.size(0) < batch_size:
            continue
        if attacker != None:
            optim = torch.optim.SGD(model.parameters(), lr = 1.)
            data_batch, label_batch = attacker.attack(model, optim, data_batch, label_batch, criterion)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)

    eigenvalue_list = []
    eigenvec_list = []
    model.eval()
    for eigen_idx in range(topk):

        print('>>> Eigen Index: %d / %d' % (eigen_idx, topk))
        eigenvalue = None

        param_list = get_param(model)
        v_list = [torch.randn(p.size()).to(device) for p in param_list]
        v_list = group_normalize(v_list)

        for iter_idx in range(max_iter):

            if eigenvalue is None:
                print('Iter Index: %d / %d' % (iter_idx, max_iter))
            else:
                print('Iter Index: %d / %d --> %.4f' % (iter_idx, max_iter, eigenvalue))
            Hv_sum = [torch.zeros(p.size()).to(device) for p in param_list]
            counter = 0
            for idx, (data_batch, label_batch) in enumerate(zip(data_batch_list, label_batch_list)):

                sys.stdout.write('Instance Idx: %d\r' % idx)
                model.zero_grad()
                logits = model(data_batch)
                loss = criterion(logits, label_batch)
                loss.backward(create_graph = True)

                param_list, grad_list = get_param_grad(model)
                Hv = torch.autograd.grad(grad_list, param_list, grad_outputs = v_list, only_inputs = True, retain_graph = False)

                Hv_sum = [Hv_sum_item + Hv_item for Hv_sum_item, Hv_item in zip(Hv_sum, Hv)]
                for value, vector in zip(eigenvalue_list, eigenvec_list):
                    inner_prod = group_product(vector, v_list).data.cpu().item()
                    Hv_sum = group_add(Hv_sum, 1., vector, - value * float(inner_prod))
                counter += 1

            eigenvalue_next = group_product(Hv_sum, v_list).data.cpu().item() / float(counter)
            v_list = group_normalize(Hv_sum)

            if eigenvalue != None and abs((eigenvalue_next - eigenvalue) / eigenvalue) < tol:
                break
            else:
                eigenvalue = eigenvalue_next
        print('')
        print('Eigenvalue %d = %.4f' % (eigen_idx, eigenvalue_next))
        eigenvalue_list.append(eigenvalue_next)
        eigenvec_list.append(v_list)

        # Convert to Numpy
        eigenvec_list_tosave = [None,] * len(eigenvec_list)
        for eigen_idx, eigenvec in enumerate(eigenvec_list):
            eigenvec_list_tosave[eigen_idx] = [v.data.cpu().numpy() for v in eigenvec]

        tosave['eigenvalue_list'] = eigenvalue_list
        tosave['eigenvec_list'] = eigenvec_list_tosave

        if out_file != None:
            pickle.dump(tosave, open(out_file, 'wb'))

    return eigenvalue_list, eigenvec_list

