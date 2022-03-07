#!/bin/bash

program=train_mgvae_conv
dataset=cifar10
dir=./$program-$dataset/
rm -rf $dir
mkdir $dir
cd $dir
mkdir visualization
mkdir generation_32x32
mkdir generation_16x16
mkdir generation_8x8
cd ..

num_epoch=1024
batch_size=20
learning_rate=0.001
kl_loss=1
seed=123456789
cluster_height=2
cluster_width=2
n_levels=2
n_layers=6
Lambda=1
hidden_dim=256
z_dim=256
device=cuda

name=${program}.${dataset}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.kl_loss.${kl_loss}.seed.${seed}.cluster_height.${cluster_height}.cluster_width.${cluster_width}.n_levels.${n_levels}.n_layers.${n_layers}.Lambda.${Lambda}.hidden_dim.${hidden_dim}.z_dim.${z_dim}
python3 $program.py --dir=$dir --dataset=$dataset --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --kl_loss=$kl_loss --seed=$seed --cluster_height=$cluster_height --cluster_width=$cluster_width --n_levels=$n_levels --n_layers=$n_layers --Lambda=$Lambda --hidden_dim=$hidden_dim --z_dim=$z_dim --device=$device

