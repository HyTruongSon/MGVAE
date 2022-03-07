from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np

feature = np.loadtxt('train_mgvae/train_mgvae.dataset.QM9_smiles.num_epoch.2048.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.n_clusters.2.n_levels.3.n_layers.3.Lambda.0.01.hidden_dim.128.z_dim.128.epoch.7.feature')
target = 'mu'

y = np.loadtxt('QM9_smiles/targets/' + target)
y = y[1:]
MIN = np.min(y)
MAX = np.max(y)
y = (y - MIN) / (MAX - MIN)
N = y.shape[0]
perm = np.random.permutation(N)
n_test = 13046
n_train = N - n_test
train_idx = perm[:n_train]
test_idx = perm[n_train:]
X_train = feature[train_idx]
y_train = y[train_idx]
X_test = feature[test_idx]
y_test = y[test_idx]
print('Done loading data')

reg = LinearRegression().fit(X_train, y_train)
predict_train = reg.predict(X_train)
predict_test = reg.predict(X_test)
mae_train = np.mean(np.abs(predict_train - y_train)) * (MAX - MIN)
mae_test = np.mean(np.abs(predict_test - y_test)) * (MAX - MIN)
print('Train MAE:', mae_train)
print('Test MAE:', mae_test)
