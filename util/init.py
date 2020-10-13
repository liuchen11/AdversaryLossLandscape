import torch
import torch.nn as nn

init_func = {
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'uniform': nn.init.uniform_,
    'normal': nn.init.normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
}

def init(model, init_type, param_type = ['4d',]):

    if init_type == None:

        return

    for param in model.parameters():

        if param.dim() == 4 and '4d' in param_type:
            init_func[init_type](param)

        if param.dim() == 3 and '3d' in param_type:
            init_func[init_type](param)

        if param.dim() == 2 and '2d' in param_type:
            init_func[init_type](param)

        if param.dim() == 1 and '1d' in param_type:
            init_func[init_type](param)
