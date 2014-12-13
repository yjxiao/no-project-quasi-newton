import numpy as np
from optimize.log_reg import Logistic_Regressor
import time, csv
import sys

X = []
y = []
filename = sys.argv[1]

with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        X.append(map(float, row[:-1]))
        y.append(float(row[-1]))

X = np.asarray(X)
y = np.asarray(y)
m = len(y)
idx = np.arange(m)
np.random.shuffle(idx)
X_train = X[idx]
y_train = y[idx]


options = dict(output=True, disp=False)

for M in [1, 5, 20, 100]:
    clf = Logistic_Regressor(solver='lbfgs')
    print '--------------------'
    print M
    options = dict(output=True, disp=False, maxlen=M)
    t0 = time.time()
    clf.train(X_train, y_train, options)
    dt = time.time() - t0
    with open('lbfgs_memo_{}_{}.txt'.format(M, filename.split('.')[0]), 'w') as f:
        f.write('total running time: {}'.format(dt))
        f.write('optimal w:\n{}'.format(clf.w))

