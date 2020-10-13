import os
import sys
sys.path.insert(0, './')
import argparse
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_file1', type = str, default = None,
        help = 'The checkpoint1, default = None.')
    parser.add_argument('--ckpt_file2', type = str, default = None,
        help = 'The checkpoint2, default = None.')

    args = parser.parse_args()

    ckpt1 = torch.load(args.ckpt_file1)
    ckpt2 = torch.load(args.ckpt_file2)

    keys1 = ckpt1.keys()
    keys2 = ckpt2.keys()

    # Check 
    [ckpt2[key] for key in keys1]
    [ckpt1[key] for key in keys2]

    distance = 0.
    for key in keys1:
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            continue
        param1 = ckpt1[key].view(-1).data.cpu().numpy()
        param2 = ckpt2[key].view(-1).data.cpu().numpy()
        distance = distance + np.linalg.norm(param1 - param2) ** 2
    distance = distance ** 0.5

    print('Distance = %.4f' % distance)
