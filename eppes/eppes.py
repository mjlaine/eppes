# EPPES algorithms in python
# -*- coding: utf-8; -*-
"""EPPES module


Ensemble prediction and estimation. See [1]_ for details.

References
----------

.. [1] M. Laine, A. Solonen, H. Haario, H. Järvinen, "Ensemble
       prediction and parameter estimation system: the method", Quarterly
       Journal of the Royal Meteorological Society, 138(663),
       pp. 289-297, 2012.

"""

from __future__ import division, print_function

import numpy as np
from scipy import linalg
from scipy.stats import rankdata, chi2


# from typing import Dict, Tuple, Sequence, Optional

# from scipy.linalg import blas

# def resample(theta,weights):
#    return sample

def wmean(theta, weights):
    """Weighted mean."""
    return np.average(theta, weights=weights)


def invertmat_sym(a):
    """Invert symmetric positive definite matrix."""
    return linalg.solve(a, np.eye(a.shape[0]), assume_a='pos')


# this should be multiplying symmetric matrix a with vector x
def matmuls(a, x):
    """Matrix vector multiplication."""
    return np.dot(a, x)


#    return blas.dsymv(1.0,a,x)

def _mean1(x):
    """Mean along the rows of a matrix."""
    return np.mean(x, axis=0)


def _cov1(x, mu):
    """Covariance using fixed mu as a center."""
    y = x - mu
    return np.dot(y.T, y) / x.shape[0]


def eppesupdate(theta, mu, w, sig, n, w00=None, maxn=0, maxsteprel=None):
    """Eppes parameter update function.
    
    Use as
       mu,w,sig,n = eppesupdate(theta,mu,w,sig,n):
       :param theta: The ensemble of parameter values to updatate the parameters
       :param mu: Mean parameter
       :param w: The accuracy of mu as a covariance matrix
       :param sig: Covariance parameter
       :param n: The accuracy of sig as imaginary observations
       :param w00: Optimal minimal w00
       :param maxn: Optimal maximum n
       :param maxsteprel: Optimal maximum relative change in mu
       :return: mu_new, w_new, sig_new, n_new updated parameters

    """
    iw = invertmat_sym(w)
    isig = invertmat_sym(sig)
    # W and n do not depend on the ensemble
    w_new = invertmat_sym(iw + isig)
    n_new = n + 1
    mu_new = matmuls(w_new, (matmuls(iw, mu) + matmuls(isig, _mean1(theta))))

    # allow only change of relative size maxsteprel, if mu is positive
    if maxsteprel is not None and maxsteprel > 0.0:
        np.copyto(mu_new, (1.0 + np.sign(mu_new - mu) * maxsteprel) * mu,
                  where=np.logical_and(mu > 0.0, np.abs((mu_new - mu) / (np.abs(mu) + 0.0001)) > maxsteprel))

    sig_new = (n * sig + _cov1(theta, mu_new)) / n_new

    if maxn > 0:
        n_new = max(n_new, maxn)
    if w00 is not None:
        w_new = w_new + w00

    return mu_new, w_new, sig_new, n_new


def propose(nsample, mu, sig, bounds=None, maxtry=1000):
    """Propose new sample given mu, sig and bounds.
    :param nsample:
    :param mu:
    :param sig:
    :param bounds:
    :param maxtry:
    :return:
    """
    x = np.random.multivariate_normal(mu, sig, size=nsample)
    # no check on bounds
    if bounds is not None:
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            pass
        if np.any((mu <= bounds[:, 0]) | (mu >= bounds[:, 1])):
            pass
        itry = 0
        bad = np.any((x < bounds[:, 0]) | (x > bounds[:, 1]), axis=1)
        while np.any(bad):
            itry += 1
            x[bad, :] = np.random.multivariate_normal(mu, sig, size=sum(bad))
            bad = np.any((x < bounds[:, 0]) | (x > bounds[:, 1]), axis=1)
            if (itry > maxtry) & any(bad):
                # uniform for mu μ±2*σ or [min,max]
                ll = np.max(np.array([[bounds[:, 0]], [mu - 2 * np.sqrt(np.diag(sig))]]), axis=0)
                uu = np.min(np.array([[bounds[:, 1]], [mu + 2 * np.sqrt(np.diag(sig))]]), axis=0)
                # make sure we are still within bounds
                # can happen if mu not in [min,max]
                np.copyto(ll, bounds[:, 0], where=ll > bounds[:, 1])
                np.copyto(uu, bounds[:, 1], where=uu < bounds[:, 0])
                x[bad, :] = np.random.uniform(low=ll, high=uu, size=(sum(bad), x.shape[1]))
                break

    return x


# def logresample(theta: np.ndarray, scores: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
def logresample(theta: np.ndarray, scores: np.ndarray):
    u"""resample theta with weight proportional to exp(-0.5*scores)

    Parameters
    ----------

    theta : array
       Array of values to be resampled
    scores : array
       Score weights of same size as theta

    Returns
    -------

    sample : array
       Resampled theta.
    wout : array
       2d array with weights in the first column and number of times the row was
       sampled in the second column.
       

    """
    nsample = theta.shape[0]
    npar = theta.shape[1]

    if nsample != scores.shape[0]:
        print('weight vector size  and sample size do not match')
        return

    wout = np.zeros((nsample, 2))
    w = np.zeros(nsample)
    sample = np.zeros((nsample, npar))
    #  !! calculate cumulative sum of the actual weights = exp(-0.5*scores)
    w[0] = 1.0 / np.sum(np.exp(-0.5 * (scores - scores[0])))
    wout[0, 0] = w[0]
    for i in range(1, nsample - 1):
        w[i] = w[i - 1] + 1.0 / np.sum(np.exp(-0.5 * (scores - scores[i])))
        wout[i, 0] = w[i] - w[i - 1]
    w[nsample - 1] = 1.0
    wout[nsample - 1, 0] = w[nsample - 1] - w[nsample - 2]

    wout[:, 1] = 0.0  # initialize

    # !! sample according to weights
    for i in range(nsample):
        u = np.random.rand(1)
        ind = 0
        while u >= w[ind]:
            ind += 1
            if ind == nsample - 1:
                break  # ! should not happen
        sample[i, :] = theta[ind, :]
        wout[ind, 1] = wout[ind, 1] + 1.0

    return sample, wout


def eppesstep(sample, scores, mu, w, sig, n, bounds=None):
    """ perform one eppes step """
    theta, wout = logresample(sample, scores)
    mu, w, sig, n = eppesupdate(theta, mu, w, sig, n)
    nsample = sample.shape[0]
    newsample = propose(nsample, mu, sig, bounds=bounds)
    return newsample, wout, mu, w, sig, n


def rankscores(scores, scale=None, center=None, maxrank=2.0):
    """ rank the scores, center and scale """
    r = np.empty(np.shape(scores))
    nrow = np.shape(r)[0]
    ncol = np.shape(r)[1]
    maxrank = np.resize(maxrank, ncol)  #
    if scale is None:
        #        scale = np.double(nrow-1)/2.0/maxrank # between -2..2
        scale = np.resize(np.array([np.double(nrow)]), ncol)
    else:
        scale = np.resize(scale, ncol)
    if center is None:
        #       center = (nrow+1.0)/2.0
        #       center = -np.double(nrow)
        center = 0.5
    for i in range(ncol):
        r[:, i] = (rankdata(scores[:, i], method='average') - center) / scale[i]
    r = chi2.ppf(r, nrow)
    return r


def combinescores(scores, method='amean'):
    ncol = np.shape(scores)[1]
    if ncol == 1:
        return scores
    if method == 'amean':
        return np.mean(scores, axis=1)
    elif method == 'gmean':
        return np.product(scores ** (1.0 / ncol), axis=1)
    else:  # default is mean
        return np.mean(scores, axis=1)


def tomatrix(x):
    """ ensure that we have 2d array """
    if np.size(np.shape(x)) == 1:
        x.shape = (x.shape[0], 1)


# do not use this
def logpropose(nsample, mu, sig, bounds=[], maxtry=1000):
    """Propose new sample given sample mu, sig and bounds, using logN """
    mus = log(mu)
    sigs = sig / np.outer(mu, mu)
    x = np.exp(np.random.multivariate_normal(mus, sigs, size=nsample))
    # no check on bounds
    if len(bounds):
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            pass
        if np.any((mu <= bounds[:, 0]) | (mu >= bounds[:, 1])):
            pass
        itry = 0
        bad = np.any((x < bounds[:, 0]) | (x > bounds[:, 1]), axis=1)
        while np.any(bad):
            itry += 1
            x[bad, :] = np.exp(np.random.multivariate_normal(mus, sigs, size=sum(bad)))
            bad = np.any((x < bounds[:, 0]) | (x > bounds[:, 1]), axis=1)
            if (itry > maxtry) & any(bad):
                # uniform for mu μ±2*σ or [min,max]
                ll = np.max(np.array([[bounds[:, 0]], [mu - 2 * np.sqrt(np.diag(sig))]]), axis=0)
                uu = np.min(np.array([[bounds[:, 1]], [mu + 2 * np.sqrt(np.diag(sig))]]), axis=0)
                # make sure we are still within bounds
                # can happen if mu not in [min,max]
                np.copyto(ll, bounds[:, 0], where=ll > bounds[:, 1])
                np.copyto(uu, bounds[:, 1], where=uu < bounds[:, 0])
                x[bad, :] = np.random.uniform(low=ll, high=uu, size=(sum(bad), x.shape[1]))
                break

    return x
