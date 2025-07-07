#!/bin/bash

# ----------- test -------0--
# Training settings
export CUDA_VISIBLE_DEVICES=1


python main_dunhuang_test.py \
    --input_dir "opt/dataset/DunHuang" \
    --input_val_dir "opt/dataset/DunHuang" \
    --output_dir "checkpoints" \
     --train_data "DunHuang" \
    --test_data "DunHuang_20_mask" \
    --test_list "test_pair_mask.lst" \
    --is_testing true \
    --predict_all false \
    --up_scale false \
    --resume false \
    --checkpoint_data "20/20_model.pth" \
    --test_img_width 720 \
    --test_img_height 720 \
    --res_dir "result" \
    --use_gpu 0 \
    --log_interval_vis 200 \
    --show_log 20 \
    --epochs 30 \
    --lr 1e-3 \
    --lrs 8e-5 \
    --wd 2e-4 \
    --adjust_lr 4 \
    --workers 2 \
    --tensorboard true \
    --img_width 720 \
    --img_height 720 \