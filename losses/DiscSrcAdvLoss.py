import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import ipdb


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)
   

class DiscAdvLossForSource_PartialDA(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(DiscAdvLossForSource_PartialDA, self).__init__(weight, size_average)

    def forward(self, input, target, class_weight):
        _assert_no_grad(target)
        batch_size = target.size(0)
        
        prob = F.softmax(input, dim=1)
        class_weight = Variable(class_weight)
        
        loss = 0
        for i in range(batch_size):
            if (prob.data[i, target.data[i]] != 1) and (prob.data[i, target.data[i]] != 0):
                loss += class_weight[target[i]] * (- prob[i, target.data[i]].log().mul(1 - prob[i, -1]) - (1 - prob[i, target.data[i]]).log().mul(prob[i, -1]))
            elif prob.data[i, target.data[i]] == 1:
                loss += class_weight[target[i]] * (- prob[i, target.data[i]].log().mul(1 - prob[i, -1]) - (1 - (1 - 1e-6) * prob[i, target.data[i]]).log().mul(prob[i, -1]))
            elif prob.data[i, target.data[i]] == 0:
                loss += class_weight[target[i]] * (- (prob[i, target.data[i]] + 1e-6).log().mul(1 - prob[i, -1]) - (1 - prob[i, target.data[i]]).log().mul(prob[i, -1]))
            
        loss /= batch_size
        
        return loss


