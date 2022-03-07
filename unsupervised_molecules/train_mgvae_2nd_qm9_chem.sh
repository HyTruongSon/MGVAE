#!/bin/bash

program=train_mgvae_2nd_qm9_chem
dir=./$program/
mkdir $dir

dataset=QM9_smiles
num_epoch=256
batch_size=20
learning_rate=0.001
kl_loss=1
seed=123456789
n_clusters=2
n_levels=2
n_layers=6
Lambda=0.01
hidden_dim=256
z_dim=256
magic_number=64
device=cpu

name=${program}.dataset.${dataset}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.kl_loss.${kl_loss}.seed.${seed}.n_clusters.${n_clusters}.n_levels.${n_levels}.n_layers.${n_layers}.Lambda.${Lambda}.hidden_dim.${hidden_dim}.z_dim.${z_dim}.magic_number.${magic_number}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --magic_number=$magic_number --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --kl_loss=$kl_loss --seed=$seed --n_clusters=$n_clusters --n_levels=$n_levels --n_layers=$n_layers --Lambda=$Lambda --hidden_dim=$hidden_dim --z_dim=$z_dim --device=$device

