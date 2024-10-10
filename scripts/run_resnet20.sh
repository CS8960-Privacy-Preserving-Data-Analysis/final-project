#!/bin/bash

# Define variables for model and privacy parameters
model=resnet20
noise_multiplier=1.1
max_grad_norm=1.0
target_epsilon=3.0
delta=1e-5

# Log the command for debugging
echo "python -u src/trainer.py \
  --arch=$model \
  --save-dir=save_$model \
  --noise-multiplier=$noise_multiplier \
  --max-grad-norm=$max_grad_norm \
  --target-epsilon=$target_epsilon \
  --delta=$delta |& tee -a log_$model"

# Run the command
python -u ../src/trainer.py \
  --arch=$model \
  --save-dir=save_$model \
  --noise-multiplier=$noise_multiplier \
  --max-grad-norm=$max_grad_norm \
  --target-epsilon=$target_epsilon \
  --delta=$delta |& tee -a log_$model
