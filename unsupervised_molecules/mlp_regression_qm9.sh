#!/bin/bash

program=mlp_regression_qm9
dir=./$program/
mkdir $dir

dataset=QM9_smiles
features_fn=train_mgvae_2nd_qm9_chem/train_mgvae_2nd_qm9_chem.dataset.QM9_smiles.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.n_clusters.2.n_levels.2.n_layers.6.Lambda.0.01.hidden_dim.256.z_dim.256.magic_number.64.feature
num_epoch=256
batch_size=20
learning_rate=0.01
seed=123456789
device=cuda

for learning_target in U U0 gap HOMO LUMO mu omega1 alpha Cv G H R2 ZPVE
do
name=${program}.dataset.${dataset}.learning_target.${learning_target}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --features_fn=$features_fn --learning_target=$learning_target --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --device=$device
done
