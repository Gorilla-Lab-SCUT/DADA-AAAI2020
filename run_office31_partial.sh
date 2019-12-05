#!/bin/bash

python main.py  --epochs 200  --source-train-data-path /sharedData/Office31/  --source-domain amazon  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain webcam  --num-classes-s 31  --print-freq 4  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_a2w_200epoch_resnet50/checkpoint.pth.tar  --log office31 


python main.py --epochs 200 --source-train-data-path /sharedData/Office31/  --source-domain dslr  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain webcam  --num-classes-s 31  --print-freq 2  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_d2w_200epoch_resnet50/checkpoint.pth.tar  --log office31 


python main.py  --epochs 200  --source-train-data-path /sharedData/Office31/  --source-domain webcam  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain dslr  --num-classes-s 31  --print-freq 2  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_w2d_200epoch_resnet50/checkpoint.pth.tar  --log office31 


python main.py  --epochs 200  --source-train-data-path /sharedData/Office31/  --source-domain amazon  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain dslr  --num-classes-s 31  --print-freq 4  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_a2d_200epoch_resnet50/checkpoint.pth.tar  --log office31 


python main.py --epochs 200 --source-train-data-path /sharedData/Office31/  --source-domain dslr  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain amazon  --num-classes-s 31  --print-freq 2  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_d2a_200epoch_resnet50/checkpoint.pth.tar  --log office31 


python main.py  --epochs 200  --source-train-data-path /sharedData/Office31/  --source-domain webcam  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain amazon  --num-classes-s 31  --print-freq 2  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_w2a_200epoch_resnet50/checkpoint.pth.tar  --log office31 



python main.py  --epochs 200  --source-train-data-path /sharedData/Office31/  --source-domain amazon  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain webcam  --num-classes-s 31  --print-freq 4  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_a2w_200epoch_resnet50/checkpoint.pth.tar  --log office31_2  --convex-combine 


python main.py --epochs 200 --source-train-data-path /sharedData/Office31/  --source-domain dslr  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain webcam  --num-classes-s 31  --print-freq 2  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_d2w_200epoch_resnet50/checkpoint.pth.tar  --log office31_2  --convex-combine 


python main.py  --epochs 200  --source-train-data-path /sharedData/Office31/  --source-domain webcam  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain dslr  --num-classes-s 31  --print-freq 2  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_w2d_200epoch_resnet50/checkpoint.pth.tar  --log office31_2  --convex-combine 


python main.py  --epochs 200  --source-train-data-path /sharedData/Office31/  --source-domain amazon  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain dslr  --num-classes-s 31  --print-freq 4  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_a2d_200epoch_resnet50/checkpoint.pth.tar  --log office31_2  --convex-combine 


python main.py --epochs 200 --source-train-data-path /sharedData/Office31/  --source-domain dslr  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain amazon  --num-classes-s 31  --print-freq 2  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_d2a_200epoch_resnet50/checkpoint.pth.tar  --log office31_2  --convex-combine 


python main.py  --epochs 200  --source-train-data-path /sharedData/Office31/  --source-domain webcam  --target-train-data-path /sharedData/office_caltech_10/  --target-test-data-path /sharedData/office_caltech_10/  --target-domain amazon  --num-classes-s 31  --print-freq 2  --lr 0.0001  --weight-decay 1e-4  --batch-size 128  --workers 4  --test-time 200  --stop-epoch 200  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint checkpoints/baseline_office31_w2a_200epoch_resnet50/checkpoint.pth.tar  --log office31_2  --convex-combine 


