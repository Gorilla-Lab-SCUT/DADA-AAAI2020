import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


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


class AdvLossForTarget_min(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(AdvLossForTarget_min, self).__init__(weight, size_average)

    def forward(self, input, target):
        _assert_no_grad(target)
        batch_size = target.size(0)
        
        prob = F.softmax(input, dim=1)

        if (prob.data[:, -1] == 0).sum() != 0:
            temp = torch.FloatTensor(batch_size).fill_(0)
            temp[prob.data.cpu()[:, -1]==0] = 1e-6
            soft_weight_var = Variable(temp).cuda()
            loss = (- ((prob[:, -1] + soft_weight_var).log()).mean())
        else:
            loss = (- (prob[:, -1].log()).mean())
                    
        return loss


class DiscAdvLossForTarget_min(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, nClass = 10):
        super(DiscAdvLossForTarget_min, self).__init__(weight, size_average)
        self.nClass = nClass

    def forward(self, input, target):
        _assert_no_grad(target)
        batch_size = target.size(0)
        
        prob = F.softmax(input[:, :-1], dim=1)
        loss = 0
        for i in range(self.nClass):
            prob_c = F.softmax(torch.cat([input[:, i].unsqueeze(1), input[:, -1].unsqueeze(1)], dim=1), dim=1)          
            if (prob_c.data[:, -1] == 0).sum() != 0:
                temp = torch.FloatTensor(batch_size).fill_(0)
                temp[prob_c.data.cpu()[:, -1]==0] = 1e-6
                soft_weight_var = Variable(temp).cuda()
                loss += (- (prob[:, i] * ((prob_c[:, -1] + soft_weight_var).log())).mean())
            else:
                loss += (- (prob[:, i] * (prob_c[:, -1].log())).mean())
            
        
        return loss
        

