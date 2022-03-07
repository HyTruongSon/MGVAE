#!/bin/bash

program=train_mgvae_simple
dir=./$program/
mkdir $dir
cd $dir
mkdir visualization
cd ..

num_epoch=32
batch_size=20
learning_rate=0.001
kl_loss=1
seed=123456789
cluster_height=2
cluster_width=2
n_levels=2
n_layers=4
Lambda=1
hidden_dim=256
z_dim=256
device=cuda

name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.kl_loss.${kl_loss}.seed.${seed}.cluster_height.${cluster_height}.cluster_width.${cluster_width}.n_levels.${n_levels}.n_layers.${n_layers}.Lambda.${Lambda}.hidden_dim.${hidden_dim}.z_dim.${z_dim}
python3 $program.py --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --kl_loss=$kl_loss --seed=$seed --cluster_height=$cluster_height --cluster_width=$cluster_width --n_levels=$n_levels --n_layers=$n_layers --Lambda=$Lambda --hidden_dim=$hidden_dim --z_dim=$z_dim --device=$device

