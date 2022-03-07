#!/bin/bash

program=train_vgae
dir=./$program/
mkdir $dir
device=cuda

for dataset in citeseer cora
do
for seed in 1 2 3 4 5 6 7 8 9 10
do
name=${program}.dataset.${dataset}.seed.${seed}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --seed=$seed --device=$device
done
done
