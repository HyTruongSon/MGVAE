#!/bin/bash

program=train_mgvae_2nd_cluster
dir=./$program/
mkdir $dir
kl_loss=0
device=cuda

num_layers=3
n_clusters=2
n_levels=7

learning_rate=0.001

for dataset in citeseer cora
do
for seed in 1 2 3 4 5 6
do
name=${program}.dataset.${dataset}.kl_loss.${kl_loss}.seed.${seed}.num_layers.${num_layers}.n_clusters.${n_clusters}.n_levels.${n_levels}.learning_rate.${learning_rate}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --kl_loss=$kl_loss --seed=$seed --num_layers=${num_layers} --n_clusters=$n_clusters --n_levels=$n_levels --learning_rate=$learning_rate --device=$device
done
done

