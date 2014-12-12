from __future__ import division
import numpy as np

n_samples = input('sample size: ')
n_features = input('number of features: ')

while True:
    w = np.random.uniform(-1, 1, n_features+1)
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    sigmoid = lambda x: 1. / (1 + np.exp(np.dot(x, w)))
    y = np.asarray(sigmoid(np.append(X, np.ones((n_samples, 1)), 1)) > 0.5, dtype=int)

    n_pos = sum(y)
    pr = n_pos / n_samples

    data = np.append(X, y[:, np.newaxis], 1)
    np.savetxt('data_s{}_f{}.csv'.format(n_samples, n_features), data, delimiter=',', fmt='%1.4f')
    np.savetxt('para_s{}_f{}.csv'.format(n_samples, n_features), w, delimiter=',', fmt='%1.8f')
    break

