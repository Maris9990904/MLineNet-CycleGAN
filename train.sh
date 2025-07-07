export CUDA_VISIBLE_DEVICES=2
#!/bin/bash

# ----------- test -------0--
# Training settings

python main_TEED.py \
    --input_dir "opt/dataset/DunHuang" \
    --input_val_dir "opt/dataset/DunHuang" \
    --output_dir "checkpoints" \
    --train_data "DunHuang" \
    --train_list "augmented_train_pair.lst" \
    --resume false \
    --checkpoint_data "29/29_model.pth" \
    --log_interval_vis 200 \
    --epochs 30 \
    --lr 1e-3 \
    --lrs 8e-5 \
    --wd 2e-4 \
    --adjust_lr 4 \
    --workers 2 \
    --tensorboard true \
    --img_width 720 \
    --img_height 720 \

