#!/bin/bash

program=train_mgvae_2nd
dir=./$program/
mkdir $dir
num_layers=2
kl_loss=0
device=cuda

n_clusters=4
n_levels=3
pca=1

for dataset in cora citeseer
do
for partition in kmeans spectral
do
for seed in 1 2 3 4 5 6
do
name=${program}.dataset.${dataset}.kl_loss.${kl_loss}.seed.${seed}.num_layers.${num_layers}.n_clusters.${n_clusters}.n_levels.${n_levels}.pca.${pca}.partition.${partition}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --kl_loss=$kl_loss --seed=$seed --num_layers=${num_layers} --n_clusters=$n_clusters --n_levels=$n_levels --pca=$pca --partition=$partition --device=$device
done
done
done
