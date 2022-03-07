#!/bin/bash

program=train_mgvae_2nd
dir=./$program/
mkdir $dir

graph_type=citeseer_small
num_epoch=128
batch_size=20
learning_rate=0.001
kl_loss=1
n_clusters=2
n_levels=2
n_layers=10
Lambda=0.01
hidden_dim=256
z_dim=256
magic_number=64
device=cuda

for seed in 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
name=${program}.graph_type.${graph_type}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.kl_loss.${kl_loss}.seed.${seed}.n_clusters.${n_clusters}.n_levels.${n_levels}.n_layers.${n_layers}.Lambda.${Lambda}.hidden_dim.${hidden_dim}.z_dim.${z_dim}.magic_number.${magic_number}
python3 $program.py --dir=$dir --name=$name --graph_type=$graph_type --magic_number=$magic_number --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --kl_loss=$kl_loss --seed=$seed --n_clusters=$n_clusters --n_levels=$n_levels --n_layers=$n_layers --Lambda=$Lambda --hidden_dim=$hidden_dim --z_dim=$z_dim --device=$device
done
