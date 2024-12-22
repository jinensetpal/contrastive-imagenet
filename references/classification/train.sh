#!/usr/bin/bash

#SBATCH --account=bbjr-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --gpus-per-node=4
#SBATCH --mem=96G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

source /u/cliu2/.bashrc
cd /work/hdd/bbjr/cliu2/vision/references/classification
ac

OMP_NUM_THREADS=16 \
torchrun --standalone --nnodes=1 --nproc-per-node=gpu train.py --model resnet50 --batch-size 256 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 \
--train-crop-size 224 --model-ema --val-resize-size 224 --ra-sampler --ra-reps 4 \
--data-path /work/hdd/bbjr/cliu2/contrastive-optimization/data/imagenet/images
