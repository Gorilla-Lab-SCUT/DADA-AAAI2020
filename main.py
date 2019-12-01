##############################################################################
#
# All the codes about the model construction should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# All the codes about the loss functions should be kept in the folder ./losses/
# All the source pre-trained checkpoints should be kept in the folder ./checkpoints/
# The file ./opts.py stores the options
# The file ./trainer.py stores the training and test strategy
# The file ./main.py should be simple
# The file ./run_visda_partial.sh stores the running commands
#
##############################################################################
import json
import os
import shutil
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import numpy as np
import random
from data.prepare_data import generate_dataloader  # Prepare dataloader
from models.resnet import resnet  # Construct ResNet model
from opts import opts  # The options for the project
from trainer import train  # The training process
from trainer import validate # The test process
from losses.DiscSrcAdvLoss import DiscAdvLossForSource_PartialDA #The source discriminative adversarial loss for partial domain adaptation
from losses.CrossEntropyLoss import AdvLossForTarget_min #The target adversarial loss in minimization
from losses.CrossEntropyLoss import DiscAdvLossForTarget_min #The target discriminative adversarial loss in minimization
from losses.InvertedLabelLoss import AdvLossForTarget_max #The target adversarial loss in maximization
from losses.InvertedLabelLoss import DiscAdvLossForTarget_max #The target discriminative adversarial loss in maximization
from losses.EntropyMinimizationLoss import EMLossForTarget #The entropy minimization loss
import ipdb #Debug


best_prec1 = 0

def main():
    global args, best_prec1
    args = opts()
    if args.arch.find('resnet') != -1:
        model = resnet(args)
    else:
        raise ValueError('Unavailable model architecture!!!')
    # define-multi GPU
    model = torch.nn.DataParallel(model).cuda()
    print(model)
    # define loss function (criterion) and optimizer
    source_adv_loss = DiscAdvLossForSource_PartialDA().cuda()
    if args.disc_tar:
        target_adv_min_loss = DiscAdvLossForTarget_min(nClass=args.num_classes_s).cuda()
        target_adv_max_loss = DiscAdvLossForTarget_max(nClass=args.num_classes_s).cuda()
    else:
        target_adv_min_loss = AdvLossForTarget_min().cuda()
        target_adv_max_loss = AdvLossForTarget_max().cuda()
    target_em_loss = EMLossForTarget().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    
    np.random.seed(1)  # fix the test data.
    random.seed(1)
    
    # apply different learning rates to different layers
    if args.arch.find('resnet') != -1:
        if args.arch.find('50') != -1:
            layer_index = 159
        elif args.arch.find('101') != -1:
            layer_index = 312
        elif args.arch.find('152') != -1:
            layer_index = 465
        else:
            raise ValueError('Undefined layer index!!!')
        optimizer = torch.optim.SGD([
            {'params': model.module.conv1.parameters(), 'name': 'pre-trained'},
            {'params': model.module.bn1.parameters(), 'name': 'pre-trained'},
            {'params': model.module.layer1.parameters(), 'name': 'pre-trained'},
            {'params': model.module.layer2.parameters(), 'name': 'pre-trained'},
            {'params': model.module.layer3.parameters(), 'name': 'pre-trained'},
            {'params': model.module.layer4.parameters(), 'name': 'pre-trained'},
            {'params': model.module.fc.parameters(), 'name': 'pre-trained'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, 
                                    nesterov=False
                                    )
    else:
        raise ValueError('Unavailable model architecture!!!')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> Loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> Loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from is not existed', args.resume)
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    source_train_loader, target_train_loader, source_val_loader, target_val_loader = generate_dataloader(args)
    #test only
    if args.test_only:
        validate(target_val_loader, model, criterion, -1, args)
        return
    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()
    
    current_epoch = 0
    print('Begin training')
    epoch_count_dataset = 'target'
    batch_number_t = len(target_train_loader)
    batch_number = batch_number_t
    batch_number_s = len(source_train_loader)    
    if batch_number_s > batch_number_t:
        epoch_count_dataset = 'source'
        batch_number = batch_number_s    
    if args.train_by_iter:
        num_iter_total = args.epochs
    else:
        num_iter_total = args.epochs * batch_number
    test_interval = int(num_iter_total / args.test_time)
    source_train_loader_batch = enumerate(source_train_loader)
    target_train_loader_batch = enumerate(target_train_loader)
    class_weight = torch.cuda.FloatTensor(args.num_classes_s).fill_(1)
    for epoch in range(args.start_epoch, num_iter_total):
        # train for one epoch
        source_train_loader_batch, target_train_loader_batch, current_epoch = train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch, model, source_adv_loss, target_adv_min_loss, target_adv_max_loss, target_em_loss, optimizer, test_interval, epoch, current_epoch, epoch_count_dataset, class_weight, layer_index, args)
        # evaluate on the val data
        if (epoch + 1) % test_interval == 0:
            prec1, class_weight = validate(target_val_loader, model, criterion, current_epoch, args)
            print('Class weight: ', class_weight)
            # record the best top-1 precision and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                log = open(os.path.join(args.log, 'log.txt'), 'a')
                log.write('\nBest accuracy till now: %3f' % (best_prec1))
                log.close()
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args)
        # early stop
        if args.train_by_iter:
            this_loop = epoch
        else:
            this_loop = current_epoch
        if this_loop > args.stop_epoch:
            break
        
    print(' * best_prec1: %3f' % best_prec1)
    # best result and end time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n * best_prec1: %3f' % best_prec1)
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()


def save_checkpoint(state, is_best, args):
    filename = 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
    

