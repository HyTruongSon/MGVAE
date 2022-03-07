program=train_mgvae_core
dir=${program}
mkdir ${program}

for dataset in citeseer
do
	for num_layers in 3 2
	do
		for kl_loss in 1
		do
			for seed in 1 2 3 4 5
			do
				name=${program}.dataset.${dataset}.num_layers.${num_layers}.kl_loss.${kl_loss}.seed.${seed}
				python3 ${program}.py --dir=$dir --name=$name --dataset=$dataset --num_layers=$num_layers --kl_loss=$kl_loss --seed=${seed}
			done
		done
	done
done
