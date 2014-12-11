from __future__ import division, print_function, absolute_import

import numpy as np
from .minimize import bfgs, dfp, lbfgs

class Logistic_Regressor():
    """ logistic regression classifier

    This class implement L2 regularized logistic regression and solve the problem in primer form

    Parameters
    ----------
    C : float, default=1.0
        Regularization term, smaller C specify stronger regularization
    
    solver : {'bfgs', 'dfp', 'lbfgs'}
        Method used to solve the optimization problem

    Attributes
    ----------
    w : array, shape (n_features+1,)
        Coefficients of the features in the decision function

    gradient : array
        Gradient value at the time of termination

    inv_hessian : array
        Approximation of the inverse hessian from Quasi Newton

    """

    def __init__(self, C=1.0, solver='bfgs'):
        
        self.C = C
        self.solver = solver.lower()
        self.w = None
        self.inv_hessian = None
        self.gradient = None
        
    def train(self, X, y, options=None):
        """ Fit the model given training data

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training samples, each has n_features features

        y : array, shape (n_samples,)
            Target values

        options : dict
            Other options controlling the behavior of the solver, pass directly to the solver as **options
        
        Returns
        ----------
        self

        """

        if options is None:
            options = dict()
            
        _X = np.copy(np.asarray(X))
        _y = np.copy(np.asarray(y))
        _y[_y==0] = -1
        m, n = X.shape
        _X = np.append(_X, np.ones((m, 1)), 1)

        if self.w is None:
            self.w = np.zeros(n+1)
        else:
            print('Seems the regressor has already been trained')
            return self

        if self.solver == 'bfgs':
            opt = bfgs
        elif self.solver == 'dfp':
            opt = dfp
        elif self.solver == 'lbfgs':
            opt = lbfgs
            
        func = lambda w: 0.5 * np.dot(w, w) + self.C * np.sum(np.log(1+np.exp(-np.dot(_X, w)*_y)))
        exp_margin = lambda w: 1 - 1. / (1 + np.exp(-np.dot(_X, w)*_y))
        fprime = lambda w: w - self.C * np.dot(_X.transpose(), exp_margin(w)*_y)

        results = opt(func, fprime, self.w, **options)
        self.w = results['x_star']
        self.gradient = results['gradient']
        if self.solver != 'lbfgs':
            self.inv_hessian = results['inv_hessian']

        return self
    
    def predict_proba(self, X):
        """ """
        _X = np.copy(np.asarray(X))
        if len(_X.shape) == 1:
            _X = np.append(_X, 1)
        else:
            m = _X.shape[0]
            _X = np.append(_X, np.ones((m, 1)), 1)
        
        if self.w is None:
            print('Please train the model first!')
            return None
        else:
            return 1. / (1 + np.exp(-np.dot(_X, self.w)))

    def predict(self, X, threshold=0.5):
        """ """
        return np.asarray(self.predict_proba(X) > threshold, dtype=int)

    def score(self, X, y, verbose=False):
        """ """
        m = len(y)
        n_pos = sum(y==1)
        y_pred = self.predict(X)
        n_pos_pred = sum(y_pred)
        precision = 0
        recall = 0
        if n_pos_pred > 0:
            precision = np.sum(y_pred[y_pred==y]==1) / n_pos_pred
        if n_pos > 0:
            recall = np.sum(y[y_pred==y]==1) / n_pos
        accuracy = sum(y_pred == y) / m

        if verbose:
            print('Sample size: {}'.format(m))
            print('Number of positives in the sample: {}'.format(n_pos))
            print('Number of predicted positives: {}'.format(n_pos_pred))            
            print('    Precision is: {:0.4f}'.format(precision))
            print('    Recall is {:0.4f}'.format(recall))
            print('    Prediction accuracy: {:0.4f}'.format(accuracy))            

        return dict(precision=precision, recall=recall, accuracy=accuracy)
