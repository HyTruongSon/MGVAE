# To run everything and reproduce the whole table in the paper
# Remark: There are 5 different models, 2 different datasets and 10 random seeds.
sh total_run.sh

# To run MGVAE + Sn (second order) with learnable clustering
sh train_mgvae_2d_cluster.sh

# To run MGVAE + Sn (second order) with K-Means or Spectral clustering
sh train_mgvae_2d.sh

# To run MGVAE with learnable clustering
sh train_mgvae_cluster.sh

# To run MGVAE with K-Means or Spectral clustering
sh train_mgvae.sh

# To run the baseline of VGAE
sh train_vgae.sh
