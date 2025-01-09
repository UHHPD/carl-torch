#!/bin/bash
whoami
pwd
which python
. ~/.bashrc
which python
# source CARLENV/bin/activate
conda activate CARLENV
which python
cp -r ~/carl-torch/ml .
echo "Training with $@"
python train.py $@
echo "done"

