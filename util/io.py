import os
import sys
import copy
import pickle
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

def eigenvec2ckpt(model, eigen_info_file, index, use_gpu):

    device = torch.device('cuda:0' if use_gpu == True else 'cpu')

    eigen_info = pickle.load(open(eigen_info_file, 'rb'))
    eigenvec = eigen_info['eigenvec_list'][index]

    vec = OrderedDict()
    for idx, (name, param) in enumerate(model.named_parameters()):
        tensor = torch.from_numpy(eigenvec[idx]).float().to(device)
        vec[name] = tensor

    saved_name = ''.join(eigen_info_file.rsplit('pkl', 1)) + '_vec_%d' % index + '.ckpt'
    torch.save(vec, saved_name)

    return vec

