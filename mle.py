#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  mle.py
#
#  Functions for maximum likelihood estimation.
#
#   * Author: Everybody is an author!
#   * Creation date: 1 March 2023
# -----------------------------------------------------------------------------

import sys
import logging

import numpy as np
import numpy.ma as ma
from scipy.stats import norm, crystalball
from scipy.special import loggamma, erf
from iminuit import Minuit

logger = logging.getLogger('xcf-spectral-analysis')

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------

#-----------------------------------------------------------------------
# normal distribution (Gaussian function)
#-----------------------------------------------------------------------
def normal(x, mu, sigma, a):
    """
    normal(x, mu, sigma, a)

    Parameters
    ----------
    x     : float or ndarray
    mu    : float
    sigma : float
    a     : float

    Returns
    -------
    f : float or ndarray

    """
    x = np.asarray(x)
    f = a * norm.pdf(x, mu, sigma) * sigma * np.sqrt(2*np.pi)
    return f

#-----------------------------------------------------------------------
# Crystal Ball function
# [ SLAC-R-236 (1980), SLAC-R-255 (1982), DESY F31-86-02 (1986) ]
#-----------------------------------------------------------------------
def crystal_ball(x, mu, sigma, beta, m, a):
    """
    crystal_ball(x, mu, sigma, beta, m, a)

    Parameters
    ----------
    x     : float or array_like of floats
    mu    : float
    sigma : float
    beta  : float
    m     : float
    a     : float

    Returns
    -------
    f : float or ndarray

    """
    x = np.asarray(x)
    c = m / np.abs(beta) / (m-1) * np.exp(-beta*beta/2)
    d = np.sqrt(np.pi/2) * (1+erf(np.abs(beta)/np.sqrt(2)))
    n = 1 / sigma / (c+d)
    f = a * crystalball.pdf(x, beta, m, mu, sigma) / n
    return f

# bimodal Crystal Ball function
def bimodal_crystal_ball(x, mu1, sigma1, beta1, m1, a1, mu2, sigma2,
                         beta2, m2, a2):
    """
    bimodal_crystal_ball(x, mu1, sigma1, beta1, m1, a1, mu2, sigma2,
                         beta2, m2, a2)

    Parameters
    ----------
    x      : float or array_like of floats
    mu1    : float
    sigma1 : float
    beta1  : float
    m1     : float
    a1     : float
    mu2    : float
    sigma2 : float
    beta2  : float
    m2     : float
    a2     : float

    Returns
    -------
    f : float or ndarray

    """
    f = crystal_ball(x, mu1, sigma1, beta1, m1, a1) + \
        crystal_ball(x, mu2, sigma2, beta2, m2, a2)
    return f

#-----------------------------------------------------------------------
# Novosibirsk function
# [ https://doi.org/10.1016/S0168-9002(99)00992-4 ]
#-----------------------------------------------------------------------
def novosibirsk(x, mu, sigma, tau):
    """
    novosibirsk(x, mu, sigma, tau)

    Parameters
    ----------
    x     : float or array_like of floats
    mu    : float
    sigma : float
    tau   : float

    Returns
    -------
    f : float or ndarray

    """
    x = np.asarray(x)
    low = 1e-7
    arg = 1 - (x-mu) * tau / sigma
    if np.isscalar(tau):
        if np.abs(tau) < low:
            return norm.pdf(x, mu, sigma) * sigma * np.sqrt(2*np.pi)
    if np.isscalar(arg):
        if arg < low:
            return 0.0
    else:
        flag = np.abs(tau) < low
    log = np.log(arg)
    xi = 2*np.sqrt(np.log(4))
    width_zero = ( 2.0 / xi ) * np.arcsinh( tau * xi * 0.5 )
    width_zero2 = width_zero * width_zero
    exponent = ( -0.5 / (width_zero2) * log * log ) - ( width_zero2 * 0.5 )
    f = np.exp(exponent)
    return f

#------------------------------------------------------------------------------
# likelihood function
#------------------------------------------------------------------------------
def make_nll(pdf, bin_edges, counts):
    """
    make_nll(pdf, bin_edges, counts)

    Closure of a negative log-likelihood function for some pdf (or pmf)
    under the assumption that the probability of measuring a single
    value is given by a Poisson probability mass function.

    Parameters
    ----------
    pdf : function
        Probability density function (or probability mass function).
    bin_edges : array_like of floats
        The bin edges along the first dimension.
    counts : array_like of floats
        Single-dimensional histogram.

    Returns
    -------
    nll : function

    """
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    def nll(parameters):
        mask = counts < 1e-9
        x = ma.masked_array(bin_centers, mask)
        z = ma.masked_array(counts, mask)
        f = pdf(x, *parameters)
        return np.sum(f - z*np.log(f) + loggamma(z+1))
    return nll

#------------------------------------------------------------------------------
# minimization routine
#------------------------------------------------------------------------------
def minimize(bins, counts, fcn, parameters, names, errors, limits,
             start=0, stop=-1, errordef=Minuit.LIKELIHOOD,
             migrad_ncall=1000000):
    """
    minimize(bins, counts, fcn, parameters, names, errors, limits,
             start=0, stop=-1, errordef=Minuit.LIKELIHOOD,
             migrad_ncall=1000000)

    Runs minimization routine using Minuit2.

    Parameters
    ----------
    bins : array_like of floats
        The bin edges along the first dimension.
    counts : array_like of floats
        Single-dimensional histogram.
    fcn : function
        Probability density function (or probability mass function).
    parameters : array_like of floats
    names : array_like of strings
        Names of parameters.
    errors : array_like of floats
    limits : array_like of floats
    start : int
    stop : int
    errordef : float
    migrad_ncall : int

    Returns
    -------
    output : dictionary

    """

    #--------------------------------------------------------------------------
    # select training sample
    #--------------------------------------------------------------------------

    flag = np.zeros(bins.shape, dtype=bool)
    flag[start:stop+1] = True

    bins_train = bins[flag]
    counts_train = counts[flag[:-1]][:-1]

    #-------------------------------------------------------------------
    # run minuit
    #-------------------------------------------------------------------

    # construct minuit object
    minuit = Minuit(
        make_nll(fcn, bins_train, counts_train), parameters, name=names)

    # set step sizes for minuit's numerical gradient estimation
    # minuit.errors = (1e-5, 1e-5, 1e-1)
    minuit.errors = errors

    # set limits for each parameter
    # minuit.limits = [ None, (0, None), (0, None) ]
    minuit.limits = limits

    # set errordef
    # for a negative log-likelihood (NLL) cost function
    # minuit.errordef = Minuit.LIKELIHOOD  # == 0.5
    # for a least-squares cost function
    # minuit.errordef = Minuit.LEAST_SQUARES  # == 1
    minuit.errordef = errordef

    # run migrad minimizer
    minuit.migrad(ncall=migrad_ncall)

    # print estimated parameters
    logger.debug('minuit.values:\n{}'.format(minuit.values))

    # run hesse algorithm to compute asymptotic errors
    minuit.hesse()

    # print estimated errors on estimated parameters
    logger.debug('minuit.errors:\n{}'.format(minuit.errors))

    # print estimated errors on estimated parameters
    logger.debug('minuit.covariance:\n{}'.format(minuit.covariance))

    # run minos algorithm to compute confidence intervals
    minuit.minos()

    # print parameters
    logger.debug('minuit.params:\n{}'.format(minuit.params))

    # print estimated errors on estimated parameters
    logger.debug('minuit.merrors:\n{}'.format(minuit.merrors))

    output = {
            'counts'       : counts,
            'bins'         : bins,
            'counts_train' : counts_train,
            'bins_train'   : bins_train,
            'minuit'       : minuit,
            'values'       : minuit.values,
            'errors'       : minuit.errors,
            'covariance'   : minuit.covariance,
            'params'       : minuit.params,
            'merrors'      : minuit.merrors,
        }

    return output

#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------
if __name__ == '__main__':

    print('hello, world!')

