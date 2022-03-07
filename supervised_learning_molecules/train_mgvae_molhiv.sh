#!/bin/bash

program=train_mgvae_molhiv
dir=./$program/
mkdir $dir

dataset=ogbg-molhiv
num_epoch=256
batch_size=20
learning_rate=0.001
seed=123456789
n_clusters=2
n_levels=3
n_layers=6
hidden_dim=256
z_dim=256
device=cpu

name=${program}.dataset.${dataset}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.n_clusters.${n_clusters}.n_levels.${n_levels}.n_layers.${n_layers}.hidden_dim.${hidden_dim}.z_dim.${z_dim}.l1_loss
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --n_clusters=$n_clusters --n_levels=$n_levels --n_layers=$n_layers --hidden_dim=$hidden_dim --z_dim=$z_dim --device=$device

