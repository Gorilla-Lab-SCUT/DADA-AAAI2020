import time
import torch
import os
import ipdb
import math
import torch.nn as nn


def train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch, model, source_adv_loss, target_adv_min_loss, target_adv_max_loss, target_em_loss, optimizer, test_interval, epoch, current_epoch, epoch_count_dataset, class_weight, layer_index, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_min = AverageMeter()
    losses_max = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    
    if args.train_by_iter:
        this_loop = epoch
    else:
        this_loop = current_epoch
    lam = 2 / (1 + math.exp(-1 * 10 * this_loop / args.epochs)) - 1
    adjust_learning_rate(optimizer, this_loop, args)
    
    new_epoch_flag = False
    
    end = time.time()
    try:
        (input_source, target_source) = source_train_loader_batch.__next__()[1]
    except StopIteration:
        source_train_loader_batch = enumerate(source_train_loader)
        if epoch_count_dataset == 'source':
            new_epoch_flag = True
            current_epoch = current_epoch + 1
        (input_source, target_source) = source_train_loader_batch.__next__()[1]

    try:
        (input_target, _) = target_train_loader_batch.__next__()[1]
    except StopIteration:
        target_train_loader_batch = enumerate(target_train_loader)
        if epoch_count_dataset == 'target':
            new_epoch_flag = True
            current_epoch = current_epoch + 1
        (input_target, _) = target_train_loader_batch.__next__()[1]
    data_time.update(time.time() - end)

    # prepare input and target
    target_source = target_source.cuda(async=True)
    input_source_var = torch.autograd.Variable(input_source)
    target_source_var = torch.autograd.Variable(target_source)
        
    target_target = torch.LongTensor(input_target.size(0)).fill_(args.num_classes_s).cuda(async=True)
    input_target_var = torch.autograd.Variable(input_target)
    target_target_var = torch.autograd.Variable(target_target)
    
    # forward
    output_source = model(input_source_var)
    output_target = model(input_target_var)
    
    # compute loss
    if args.convex_combine:
        adv_loss_src = source_adv_loss(output_source, target_source_var, lam * class_weight + (1 - lam) * 1)
    else:
        adv_loss_src = source_adv_loss(output_source, target_source_var, class_weight)
        
    adv_min_loss_tar = target_adv_min_loss(output_target, target_target_var) #ce loss
    adv_max_loss_tar = target_adv_max_loss(output_target, target_target_var) #il loss
    em_loss_tar = target_em_loss(output_target) #em loss

    if args.lam:
        loss_min = lam * (adv_loss_src + adv_min_loss_tar) + em_loss_tar
        loss_max = lam * (adv_loss_src + adv_max_loss_tar) - em_loss_tar
    else:
        loss_min = adv_loss_src + adv_min_loss_tar + em_loss_tar
        loss_max = adv_loss_src + adv_max_loss_tar - em_loss_tar
    
    # mesure accuracy and record
    prec1, prec5 = accuracy(output_source.data[:, :-1], target_source, topk=(1,5))
    losses_min.update(loss_min.data.item(), input_source.size(0))
    losses_max.update(loss_max.data.item(), input_source.size(0))
    top1.update(prec1[0], input_source.size(0))
    top5.update(prec5[0], input_source.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss_min.backward(retain_graph=True)
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_classifier = temp_grad
    
    optimizer.zero_grad()
    loss_max.backward()
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_featureExtractor = temp_grad
    
    # update parameters
    count = 0
    for param in model.parameters():
        temp_grad = param.grad.data.clone()
        temp_grad.zero_()
        if count < layer_index:
            temp_grad = temp_grad - grad_for_featureExtractor[count]
        else:
            temp_grad = temp_grad + grad_for_classifier[count]
        temp_grad = temp_grad
        param.grad.data = temp_grad
        count = count + 1
    optimizer.step()
    
    batch_time.update(time.time() - end)
    if epoch % int(test_interval / 2) == 0:
        print('Train: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss_min {loss_min.val:.4f} ({loss_min.avg:.4f})\t'
              'Loss_max {loss_max.val:.4f} ({loss_max.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
               this_loop, args.epochs, batch_time=batch_time, data_time=data_time, 
               loss_min=losses_min, loss_max=losses_max, top1=top1, top5=top5))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write("\n")
        log.write("Train: %d, loss_min: %4f, loss_max: %4f, Top1 acc: %3f, Top5 acc: %3f" % (this_loop, losses_min.avg, losses_max.avg, top1.avg, top5.avg))
        log.close()
    
    return source_train_loader_batch, target_train_loader_batch, current_epoch


def validate(target_val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    
    softmax = nn.Softmax(dim=1)
    count = 0
    class_weight = torch.cuda.FloatTensor(args.num_classes_s).fill_(0)

    end = time.time()
    total_vector = torch.FloatTensor(args.num_classes_s).fill_(0)
    correct_vector = torch.FloatTensor(args.num_classes_s).fill_(0)
    for i, (input, target) in enumerate(target_val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output[:, :-1], target_var)
        
        output_prob = softmax(output[:, :-1])
        output_prob_data = output_prob.data.clone()
        count += output_prob_data.size(0)
        class_weight += output_prob_data.sum(0)

        # measure accuracy and record
        prec1, prec5 = accuracy(output.data[:, :-1], target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        total_vector, correct_vector = accuracy_for_each_class(output.data[:, :-1], target, total_vector, correct_vector)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(target_val_loader), batch_time=batch_time, 
                   loss=losses, top1=top1, top5=top5))
    
    correct_vector = correct_vector[total_vector != 0]
    total_vector = total_vector[total_vector != 0]
    acc_for_each_class = 100.0 * correct_vector / total_vector
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n")
    log.write("                                    Test: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" %\
              (epoch, losses.avg, top1.avg, top5.avg))
    log.write("\nAcc. for each class: ")
    for i in range(acc_for_each_class.size(0)):
        if i == 0:
            log.write("%dst: %3f" % (i+1, acc_for_each_class[i]))
        elif i == 1:
            log.write(",  %dnd: %3f" % (i+1, acc_for_each_class[i]))
        elif i == 2:
            log.write(", %drd: %3f" % (i+1, acc_for_each_class[i]))
        else:
            log.write(", %dth: %3f" % (i+1, acc_for_each_class[i]))
    log.write("\nAvg. over all classes: %3f" % acc_for_each_class.mean())
    log.close()
    
    class_weight /= count
    class_weight /= max(class_weight)
    
    return top1.avg.cpu(), class_weight


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    # annealing strategy
    lr = args.lr * 10 / pow((1 + 10 * epoch / args.epochs), 0.75)
    lr_pretrain = args.lr / pow((1 + 10 * epoch / args.epochs), 0.75)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            param_group['lr'] = lr_pretrain
        else:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[target[i]] += 1
        correct_vector[torch.LongTensor([target[i]])] += correct[i]
    
    return total_vector, correct_vector
    
    
