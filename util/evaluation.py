import torch
import torch.nn as nn

class AverageCalculator(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0.
        self.average = 0.

    def update(self, value, weight = 1):
        self.value = value
        self.sum += value * weight
        self.count += weight
        self.average = self.sum / self.count

def accuracy(logits, label_batch, topk = 1, show_full_list = False):
    '''
    >>> calculate the top k accuracy for a mini_batch
    '''
    maxk = max(topk) if isinstance(topk, (tuple, list)) else topk
    batch_size = label_batch.size(0)

    _, prediction = logits.topk(maxk, 1, True, True)
    prediction = prediction.t()

    correct_mask = prediction.eq(label_batch.view(1, -1).expand_as(prediction))

    if isinstance(topk, (list, tuple)):
        full_list = [correct_mask[:k].view(-1).float() for k in topk]
        if show_full_list == False:
            return [item.sum(0).mul_(1.0 / batch_size) for item in full_list]
        else:
            return [item.sum(0).mul_(1.0 / batch_size) for item in full_list], full_list
    else:
        full_list = correct_mask[:topk].view(-1).float()
        if show_full_list == False:
            return full_list.sum(0).mul_(1.0 / batch_size)
        else:
            return full_list.sum(0).mul_(1.0 / batch_size), full_list
