import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
        

class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class EMLossForTarget(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(EMLossForTarget, self).__init__(weight, size_average)

    def forward(self, input):
        batch_size = input.size(0)
        prob = F.softmax(input[:, :-1], dim=1)
        
        if (prob.data.cpu() == 0).sum() != 0:
            weight = torch.FloatTensor(prob.size()).fill_(0)
            weight[prob.data.cpu() == 0] = 1e-6
            weight = Variable(weight).cuda()
            
            loss = - (prob + weight).log().mul(prob).sum(1).mean()
        else:
            loss = - prob.log().mul(prob).sum(1).mean()
        
        return loss
        

