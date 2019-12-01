#!/bin/bash 

python main.py  --train-by-iter  --epochs 20000  --source-train-data-path /sharedData/visDA/  --source-domain visda_train  --target-train-data-path /sharedData/visDA/  --target-test-data-path /sharedData/visDA/  --target-domain visda_validation_first6classes  --num-classes-s 12  --print-freq 50  --lr 0.0001  --weight-decay 1e-4  --no-da  --batch-size 128  --workers 4  --test-time 4000  --stop-epoch 1000  --lam  --disc-tar  --arch resnet50  --pretrained-checkpoint ./checkpoints/baseline_visda_train2val_20epoch_resnet50/model_best.pth.tar  --log visda


