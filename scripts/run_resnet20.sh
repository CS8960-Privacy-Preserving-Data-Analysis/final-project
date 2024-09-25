#!/bin/bash

model=resnet20
echo "python -u src/trainer.py --arch=$model --save-dir=save_$model |& tee -a log_$model"
python -u ../src/trainer.py --arch=$model --save-dir=save_$model |& tee -a log_$model
