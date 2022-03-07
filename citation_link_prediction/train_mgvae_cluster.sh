#!/bin/bash

program=train_mgvae_cluster
dir=./$program/
mkdir $dir
kl_loss=0
device=cuda

n_clusters=2
n_levels=7

for dataset in citeseer
do
for seed in 1 2 3 4 5 6
do
name=${program}.dataset.${dataset}.kl_loss.${kl_loss}.seed.${seed}.n_levels.${n_levels}.n_clusters.${n_clusters}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --kl_loss=$kl_loss --seed=$seed --n_clusters=$n_clusters --n_levels=$n_levels --device=$device
done
done
