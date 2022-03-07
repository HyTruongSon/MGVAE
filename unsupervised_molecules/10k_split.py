import numpy as np
np.random.seed(12345678)

dataset='train_mgvae_2nd_zinc'

train_features_fn='train_mgvae_2nd_zinc.dataset.zinc.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.n_clusters.2.n_levels.2.n_layers.4.Lambda.0.01.hidden_dim.256.z_dim.256.magic_number.32.epoch.1.split.train.feature'

train_target_fn='train_mgvae_2nd_zinc.dataset.zinc.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.n_clusters.2.n_levels.2.n_layers.4.Lambda.0.01.hidden_dim.256.z_dim.256.magic_number.32.epoch.1.split.train.target'

val_features_fn='train_mgvae_2nd_zinc.dataset.zinc.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.n_clusters.2.n_levels.2.n_layers.4.Lambda.0.01.hidden_dim.256.z_dim.256.magic_number.32.epoch.1.split.val.feature'

val_target_fn='train_mgvae_2nd_zinc.dataset.zinc.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.n_clusters.2.n_levels.2.n_layers.4.Lambda.0.01.hidden_dim.256.z_dim.256.magic_number.32.epoch.1.split.val.target'

test_features_fn='train_mgvae_2nd_zinc.dataset.zinc.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.n_clusters.2.n_levels.2.n_layers.4.Lambda.0.01.hidden_dim.256.z_dim.256.magic_number.32.epoch.1.split.test.feature'

test_target_fn='train_mgvae_2nd_zinc.dataset.zinc.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.n_clusters.2.n_levels.2.n_layers.4.Lambda.0.01.hidden_dim.256.z_dim.256.magic_number.32.epoch.1.split.test.target'

train_features = np.loadtxt(dataset + '/' + train_features_fn)
train_target = np.loadtxt(dataset + '/' + train_target_fn)
print('Done loading train')

val_features = np.loadtxt(dataset + '/' + val_features_fn)
val_target = np.loadtxt(dataset + '/' + val_target_fn)
print('Done loading val')

test_features = np.loadtxt(dataset + '/' + test_features_fn)
test_target = np.loadtxt(dataset + '/' + test_target_fn)
print('Done loading test')

features = np.concatenate([train_features, val_features, test_features], axis = 0)
target = np.concatenate([train_target, val_target, test_target], axis = 0)
print('Done concatentation')

num_train = 10000
num_total = features.shape[0]

perm = np.random.permutation(num_total)
train_idx = perm[:num_train]
test_idx = perm[num_train:]

train_features = features[train_idx]
train_target = target[train_idx]

test_features = features[test_idx]
test_target = target[test_idx]

np.savetxt(dataset + '/10k.train.feature', train_features)
np.savetxt(dataset + '/10k.train.target', train_target)
np.savetxt(dataset + '/10k.test.feature', test_features)
np.savetxt(dataset + '/10k.test.target', test_target)
print('Done')






