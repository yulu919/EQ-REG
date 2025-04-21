#!/bin/bash


################# train 
## cnn R100L Training
# python main.py  --save RCDNet_syn --model RCDNet --scale 2 \
#     --epochs 100 --batch_size 16 --patch_size 64 --data_train RainHeavy \
#     --n_threads 0 --data_test RainHeavyTest --stage 17 --lr_decay 25 \
#     --gamma 0.2 --num_M 32 --num_Z 32 --data_range 1-200/1-100 \
#     --loss 1*MSE --save_models --device 1

# EQ-REG R100L  MyTraining
# python main_loss.py  --save RCDNet_syn_equi --model RCDNet_loss_xnet --scale 2 \
#     --epochs 100 --batch_size 16 --patch_size 64 --data_train RainHeavy \
#     --n_threads 0 --data_test RainHeavyTest --stage 17 --lr_decay 25 \
#     --gamma 0.2 --num_M 32 --num_Z 33 --data_range 1-200/1-100 \
#     --loss 1*MSE --save_models --device 1

## fconv R100L Training
# python main.py  --save RCDNet_syn_fconv --model RCDNet_fconv --scale 2 \
#     --epochs 100 --batch_size 16 --patch_size 64 --data_train RainHeavy \
#     --n_threads 0 --data_test RainHeavyTest --stage 17 --lr_decay 25 \
#     --gamma 0.2 --num_M 32 --num_Z 32 --data_range 1-200/1-100 \
#     --loss 1*MSE --save_models --device 1

## GCNN R100L Training
# python main.py  --save RCDNet_syn_gcnn --model RCDNet_gcnn --scale 2 \
#     --epochs 100 --batch_size 16 --patch_size 64 --data_train RainHeavy \
#     --n_threads 0 --data_test RainHeavyTest --stage 17 --lr_decay 25 \
#     --gamma 0.2 --num_M 32 --num_Z 32 --data_range 1-200/1-100 \
#     --loss 1*MSE --save_models --device 7

# ## pdoe R100L Training
# python main.py  --save RCDNet_syn_pdoe --model RCDNet_pdoe --scale 2 \
#     --epochs 100 --batch_size 16 --patch_size 64 --data_train RainHeavy \
#     --n_threads 0 --data_test RainHeavyTest --stage 17 --lr_decay 25 \
#     --gamma 0.2 --num_M 32 --num_Z 32 --data_range 1-200/1-100 \
#     --loss 1*MSE --save_models --device 7

## e2cnn R100L Training
# python main.py  --save RCDNet_syn_e2cnn --model RCDNet_e2cnn --scale 2 \
#     --epochs 100 --batch_size 16 --patch_size 64 --data_train RainHeavy \
#     --n_threads 0 --data_test RainHeavyTest --stage 17 --lr_decay 25 \
#     --gamma 0.2 --num_M 32 --num_Z 32 --data_range 1-200/1-100 \
#     --loss 1*MSE --save_models --device 1




################# test
## conv R100L Testing
# python main.py --data_test RainHeavyTest --ext img --scale 2  \
#     --data_range 1-200/1-100 --pre_train ../experiment/RCDNet_syn/model/model_best.pt \
#     --model RCDNet --test_only --save_results --save RCDNet_syn --device 1


# # EQ-REG R100L MyTesting
# python main.py --data_test RainHeavyTest --ext img --scale 2 --num_Z 33 \
#     --data_range 1-200/1-100 --pre_train ../experiment/RCDNet_loss/model/model_best.pt \
#     --model RCDNet --test_only --save_results --save RCDNet_loss --device 3

# fconv R100L Testing
# python main.py --data_test RainHeavyTest  --ext img --scale 2  --data_range 1-200/1-100 \
#     --pre_train ../experiment/RCDNet_syn_fconv_ks3_4/model/model_best.pt --model RCDNet_fconv \
#     --test_only --save_results --save RCDNet_syn_fconv_ks3_4 --device 3



