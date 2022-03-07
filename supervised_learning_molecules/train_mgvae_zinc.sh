#!/bin/bash

program=train_mgvae_zinc
dir=./$program/
mkdir $dir

dataset=ZINC_12k
learning_target=logp
num_epoch=512
batch_size=20
learning_rate=0.002
seed=123456789
n_clusters=2
n_levels=2
n_layers=4
hidden_dim=128
z_dim=128
device=cpu

name=${program}.dataset.${dataset}.learning_target.${learning_target}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.n_clusters.${n_clusters}.n_levels.${n_levels}.n_layers.${n_layers}.hidden_dim.${hidden_dim}.z_dim.${z_dim}.l1_loss
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --learning_target=$learning_target --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --n_clusters=$n_clusters --n_levels=$n_levels --n_layers=$n_layers --hidden_dim=$hidden_dim --z_dim=$z_dim --device=$device

