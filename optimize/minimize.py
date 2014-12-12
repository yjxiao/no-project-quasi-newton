from __future__ import division, print_function, absolute_import

import numpy as np
from collections import deque

from .exceptions import LineSearchError

__all__ = ['bfgs', 'dfp', 'lbfgs']


def _norm(x, ord=2):
    """ Calculate vector norm """
    if ord == np.Inf:
        return np.amax(np.abs(x))
    elif ord == -np.Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x) ** ord) ** (1. / ord)

    
def _zoom(a_lo, a_hi, xk, pk, fk, gk, fk_lo, fun, fp, c1, c2):
    """ Part of the line search algorithm

    See Nocedal and Wright, 'Numerical Optimization' Second Edition, Algorithm 3.6 (zoom)
    """
    a, b = a_lo, a_hi
    maxiter = 20
    for i in xrange(maxiter):
        a_j = (a + b) / 2
        fk_alphaj = fun(xk+a_j*pk)
        if (fk_alphaj > fk + c1*a_j*np.dot(pk, gk)) or (fk_alphaj >= fk_lo):
            b = a_j
        else:
            derphi_alphaj = np.dot(pk, fp(xk+a_j*pk))
            if np.abs(derphi_alphaj) <= -c2*np.dot(pk, gk):
                a_star = a_j
                break
            if derphi_alphaj * (b - a) >= 0:
                b = a
            a = a_j
            fk_lo = fk_alphaj
    else:
        raise LineSearchError('Line search failed, probably reached optimum')

    return a_star
                
                    
def _line_search(fun, fp, xk, gk, pk, c1=1e-4, c2=0.9, maxiter=12):
    """ Line search method to find alpha satisfying strong Wolfe condition

    See Nocedal and Wright, 'Numerical Optimization' Second Edition, Algorithm 3.5 (Line Search Algorithm)

    """
    alpha1 = 1
    alpha0 = 0
    
    fk = fun(xk)
    fk_alpha0 = fk
    gk = fp(xk)
    
    for k in xrange(1, maxiter+1):
        fk_alpha1 = fun(xk+alpha1*pk)
        if (fk_alpha1 > fk + c1*alpha1*np.dot(pk, gk)) or ((fk_alpha1 >= fk_alpha0) and (k > 1)):
            alpha_star = _zoom(alpha0, alpha1, xk, pk, fk, gk, fk_alpha0, fun, fp, c1, c2)
            break
        gk_alpha1 = fp(xk+alpha1*pk)
        if np.abs(np.dot(pk, gk_alpha1)) <= np.abs(c2*np.dot(pk, gk)):
            alpha_star = alpha1
            break
        if np.dot(pk, gk_alpha1) >= 0:
            alpha_star = _zoom(alpha1, alpha0, xk, pk, fk, gk, fk_alpha1, fun, fp, c1, c2)
            break
        alpha0 = alpha1
        alpha1 = alpha1 * 2
        fk_alpha0 = fk_alpha1
    else:
        alpha_star = alpha1

    return alpha_star


def bfgs(fun, fp, x0, norm=2, maxiter=None, tol=1e-6, adjust_init_h=False, disp=True):
    """ Implement Quasi Newton method with BFGS update """

    x0 = np.asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 50

    N = len(x0)
    xk = x0
    gk = fp(x0)
    Hk = np.eye(N)
    fk = fun(x0)
    
    # heuristic trick to set initial H
    if adjust_init_h:
        pk = -gk
        alpha_k = _line_search(fun, fp, xk, gk, pk)
        sk = alpha_k * pk
        yk = fp(xk+sk) - gk
        Hk = np.dot(yk, sk) / np.dot(yk, yk) * np.eye(N)
    
    for k in xrange(maxiter):
        pk = - np.dot(Hk, gk)    # compute descent direction

        # Try line search
        try:
            alpha_k = _line_search(fun, fp, xk, gk, pk)
        except LineSearchError as e:
            # line search failed
            print('Line search failed: {}'.format(e))
            break
        
        sk = alpha_k * pk
        xk = xk + sk
        gk_new = fp(xk)
        yk = gk_new - gk
        gk = gk_new
        fk = fun(xk)

        # break if norm less than tol
        gknorm = _norm(gk, norm)
        if gknorm < tol:
            break

        # update approximation of inverse Hessian using Sherman-Morrison formula
        rhok = np.dot(sk, yk)
        a1 = (rhok+np.dot(yk, np.dot(Hk, yk))) / rhok**2
        a2 = 1. / rhok
        Hk = Hk + a1 * (sk[:,np.newaxis]*sk[np.newaxis,:]) - a2 * (np.dot(Hk, yk[:,np.newaxis]*sk[np.newaxis,:])+np.dot((sk[:,np.newaxis]*yk[np.newaxis,:]), Hk))

        if disp:
            print('----------------------------------------------------')
            print('    Current iteration: {:d}'.format(k+1))
            print('    Current function value: {:f}'.format(fk))
            print('    Current gradient norm: {:0.8f}'.format(gknorm))
            print('    Alpha value this iteration: {:f}'.format(alpha_k))

    print('Done. Total iterations: {}'.format(k))
    
    return dict(fmin=fk, x_star=xk, gradient=gk, inv_hessian=Hk)


def dfp(fun, fp, x0, norm=2, maxiter=None, tol=1e-6, disp=True):
    """ implement quasi newton method with DFP update """

    x0 = np.asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 100

    N = len(x0)
    xk = x0
    gk = fp(x0)
    fk = fun(x0)
    Hk = np.eye(N)

    for k in xrange(maxiter):
        pk = - np.dot(Hk, gk)    # compute descent direction
        
        # try line search
        try:
            alpha_k = _line_search(fun, fp, xk, gk, pk)
        except Exception as e:
            # line search failed
            print('Line search failed: {}'.format(e))
            break

        sk = alpha_k * pk
        xk = xk + sk
        gk_new = fp(xk)
        yk = gk_new - gk
        gk = gk_new
        fk = fun(xk)
        
        # break if norm less than tol
        gknorm = _norm(gk, norm)
        if gknorm < tol:
            break

        # update approx of inverse Hessian using Sherman-Morrison formula
        rhok = np.dot(Hk, yk)
        a1 = (sk[:,np.newaxis]*sk[np.newaxis,:]) / np.dot(yk, sk)
        a2 = (rhok[:,np.newaxis]*rhok[np.newaxis,:]) / np.dot(yk, rhok)
        Hk = Hk + a1 - a2

        if disp:
            print('----------------------------------------------------')
            print('    Current iteration: {:d}'.format(k+1))
            print('    Current function value: {:f}'.format(fk))
            print('    Current gradient norm: {:f}'.format(gknorm))
            print('    Alpha value this iteration: {:f}'.format(alpha_k))

    print('Done. Total iterations: {}'.format(k))
    return dict(fmin=fk, x_star=xk, gradient=gk, inv_hessian=Hk)


def _reconstruct(H0, gk, s, y):
    """ L-BFGS two-loop recursion described in Nocedal &. Wright, Numerical Optimization 

    Part of the L-BFGS algorithm, should not call individually 

    """
    q = gk
    m = len(s)
    alpha = deque(maxlen=m)
    for i in xrange(m-1, -1, -1):
        si = s[i]
        yi = y[i]
        rhoi = 1. / np.dot(si, yi)
        alphai = rhoi * np.dot(si, q)
        alpha.appendleft(alphai)
        q = q - alphai * yi

    r = np.dot(H0, q)
    for i in xrange(m):
        si = s[i]
        yi = y[i]
        rhoi = 1. / np.dot(yi, si)
        beta = rhoi * np.dot(yi, r)
        r = r + (alpha[i]-beta) * si

    return -r    # r = np.dot(Hk, gk), need to return negative

    
def lbfgs(fun, fp, x0, norm=2, maxiter=None, maxlen=7, tol=1e-6, disp=True):
    """ Limited memory BFGS algorithm """
    x0 = np.asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 50

    N = len(x0)
    xk = x0
    gk = fp(x0)
    gknorm = _norm(gk)
    Hk = np.eye(N)
    fk = fun(x0)
    s = deque(maxlen=maxlen)
    y = deque(maxlen=maxlen)
    
    # calculate first pair
    alpha_k = _line_search(fun, fp, xk, gk, -gk)
    sk = - alpha_k * gk
    yk = fp(xk+sk) - gk
    s.append(sk)
    y.append(yk)
    
    k = 0
    while gknorm > tol:

        Hk = np.dot(yk, sk) / np.dot(yk, yk) * np.eye(N)
        pk = _reconstruct(Hk, gk, s, y)    # compute descent direction by reconstruct Hk from historic s and y

        # Try line search
        try:
            alpha_k = _line_search(fun, fp, xk, gk, pk)
        except LineSearchError as e:
            # line search failed
            print('Line search failed: {}'.format(e))
            break
        
        sk = alpha_k * pk
        xk = xk + sk
        gk_new = fp(xk)
        yk = gk_new - gk
        gk = gk_new
        gknorm = _norm(gk)
        fk = fun(xk)
        
        # add new sk, yk
        s.append(sk)
        y.append(yk)

        # display iteration info
        if disp:
            print('----------------------------------------------------')
            print('    Current iteration: {:d}'.format(k+1))
            print('    Current function value: {:f}'.format(fk))
            print('    Current gradient norm: {:0.8f}'.format(gknorm))
            print('    Alpha value this iteration: {:f}'.format(alpha_k))

        # break if maxiter reached
        k += 1
        if maxiter < k:
            break

    print('Done. Total iterations: {}'.format(k))
    
    return dict(fmin=fk, x_star=xk, gradient=gk)


    
        
