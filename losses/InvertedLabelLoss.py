import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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


class AdvLossForTarget_max(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(AdvLossForTarget_max, self).__init__(weight, size_average)

    def forward(self, input, target):
        _assert_no_grad(target)
        batch_size = target.size(0)
        
        prob = F.softmax(input, dim=1)
        
        if (prob.data[:, -1] == 1).sum() != 0:
            temp = torch.FloatTensor(batch_size).fill_(1)
            temp[prob.data.cpu()[:, -1]==1] = (1 - 1e-6)
            soft_weight_var = Variable(temp).cuda()
            loss = (1 - prob[:, -1] * soft_weight_var).log().mean()
        else:
            loss = (1 - prob[:, -1]).log().mean()
        
        return loss


class DiscAdvLossForTarget_max(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, nClass = 10):
        super(DiscAdvLossForTarget_max, self).__init__(weight, size_average)
        self.nClass = nClass

    def forward(self, input, target):
        _assert_no_grad(target)
        batch_size = target.size(0)
        
        loss = 0
        prob = F.softmax(input[:, :-1], dim=1)
        for i in range(self.nClass):
            prob_c = F.softmax(torch.cat([input[:, i].unsqueeze(1), input[:, -1].unsqueeze(1)], dim=1), dim=1)       
            if (prob_c.data[:, -1] == 1).sum() != 0:
                temp = torch.FloatTensor(batch_size).fill_(1)
                temp[prob_c.data.cpu()[:, -1]==1] = (1 - 1e-6)
                soft_weight_var = Variable(temp).cuda()
                loss += (prob[:, i] * ((1 - prob_c[:, -1] * soft_weight_var).log())).mean()
            else:
                loss += (prob[:, i] * ((1 - prob_c[:, -1]).log())).mean()
            
        
        return loss


  