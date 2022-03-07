#!/bin/bash

program=mlp_regression_zinc
dir=./$program/
mkdir $dir

dataset=ZINC_12k
features_fn=train_mgvae_2nd_zinc/train_mgvae_2nd_zinc.dataset.ZINC_12k.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.n_clusters.2.n_levels.2.n_layers.4.Lambda.0.01.hidden_dim.256.z_dim.256.magic_number.64.epoch.7.feature
num_epoch=128
batch_size=20
learning_rate=0.001
seed=123456789
device=cuda

learning_target=logp
name=${program}.dataset.${dataset}.learning_target.${learning_target}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --learning_target=${learning_target} --features_fn=$features_fn --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --device=$device
