import numpy as np
from optimize.minimize import bfgs, dfp, _norm
from optimize.log_reg import Logistic_Regressor
from sklearn.linear_model import LogisticRegression
import time, csv

X = []
y = []
with open('data_s6_f50.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        X.append(map(float, row[:-1]))
        y.append(float(row[-1]))

X = np.asarray(X)
y = np.asarray(y)
m = len(y)
idx = np.arange(m)
split = int(m*0.8)
#np.random.shuffle(idx)
X_train = X[idx[:split]]
y_train = y[idx[:split]]
X_test = X[idx[split:]]
y_test = y[idx[split:]]

options = dict(disp=False)

for solver in ['dfp', 'bfgs', 'lbfgs']:
    clf = Logistic_Regressor(solver=solver)
    print '--------------------'
    print solver
    t0 = time.time()
    clf.train(X_train, y_train, options)
    dt = time.time() - t0
    print '||g|| at termination: {}'.format(_norm(clf.gradient))
    print 'training time: {}s'.format(dt)

