#!/usr/bin/env python

""" Authors: David A. Mely <david_mely@brown.edu>
             Thomas Serre <thomas_serre@brown.edu>
"""

from __future__ import absolute_import
try:
    import colored_traceback.always
except ImportError:
    pass

from .contextual_CUDA import _DEFAULT_VERBOSE
from .contextual_CUDA import _DEFAULT_KEEPTIME
from .contextual_CUDA import _DEFAULT_MAXITER
from .contextual_CUDA import _DEFAULT_STEPSIZE
from .contextual_CUDA import _DEFAULT_FLOATX
from .contextual_CUDA import _DEFAULT_PARAMETERS

import itertools as it
from copy import deepcopy
import numpy as np
from numpy import r_, s_
import scipy as sp
from scipy import stats
from scipy import ndimage
from scipy.ndimage.filters import uniform_filter1d as u1d
from scipy.ndimage.filters import maximum_filter1d as m1d
from scipy.ndimage.filters import gaussian_filter1d as g1d
from hmax.models.hnorm.computation import recursive_pool
from hmax.models.hnorm.computation import hwrectify
from hmax.models.hnorm.computation import softplus
from hmax.tools.utils import pb, mul


#------------------------------------------------------------------------------#
def to4(inarray, axis):
    """ Docstring for to4 """

    initshape = inarray.shape
    outarray = inarray.swapaxes(-3, axis)
    n = outarray.shape[-3]
    h = outarray.shape[-2]
    w = outarray.shape[-1]
    m = mul(outarray.shape) // (n * h * w )
    outarray.shape = (m, n, h, w)

    return outarray, initshape, n

#------------------------------------------------------------------------------#
def from4(inarray, axis, keeptime, size):
    """ Docstring for from4 """

    assert(axis >= 0)
    niters = len(inarray)
    outarray = sp.array(inarray)
    outarray = outarray.swapaxes(axis + int(keeptime), -3)
    outarray.shape = (niters,) * int(keeptime) + size

    return outarray

#------------------------------------------------------------------------------#
def run_reference(afferent, p=_DEFAULT_PARAMETERS, axis=-3,
    maxiter=_DEFAULT_MAXITER, h=_DEFAULT_STEPSIZE, keeptime=_DEFAULT_KEEPTIME,
    verbose=_DEFAULT_VERBOSE):
    """ Integrate with Forward Euler method with integration step size h
    """

    ######################################
    # re-arrange array into canonical form
    ######################################
    axis = axis % afferent.ndim
    O, initsz, nunits = to4(afferent, axis=axis)
    I, O_t, I_t = O.copy(), [], []
    if keeptime: O_t.append(O); I_t.append(I)

    ############
    # parameters
    ############
    p = _DEFAULT_PARAMETERS if p is None else p
    sigma, tau = p['sigma'], p['tau']
    epsilon, eta  = p['epsilon'], p['eta']
    ssc, sss = p['ssc'], p['sss']
    gamma, alpha, mu = p['gamma'], p['alpha'], p['mu']
    delta, beta, nu = p['delta'], p['beta'], p['nu']
    xi, zeta, omega = p['xi'], p['zeta'], p['omega']

    ##############################################
    # make sure pool sizes, input sizes make sense
    ##############################################
    assert sss < afferent.shape[-2]
    assert sss < afferent.shape[-1]
    assert ssc < sss
    assert sss % 2 == 1
    assert ssc % 2 == 1

    tuned_pooling_method = 'mean'# 'max'
    untuned_pooling_method = 'mean'# 'max'

    #################################
    # tuned summation: center pooling
    #################################
    zeta = 1.0 # because here unlike the GPU implementation we exclude center
               # and we pulled the default parameters from GPU implementation
    pool_P = {
        'type':             'pool',
        'mode':             tuned_pooling_method,
        'size':             (1, 1, ssc, ssc),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   (1, 1),
        'subpool':          {'type': None},
    }

    ####################################################
    # untuned suppression: reduction across feature axis
    ####################################################
    pool_U = {
        'type':             'pool',
        'mode':             untuned_pooling_method,
        'size':             (1, -1, 1, 1),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   None,
    }

    #####################################
    # tuned suppression: surround pooling
    #####################################
    pool_T = {
        'type':             'pool',
        'mode':             tuned_pooling_method,
        'size':             (1, 1, sss, sss),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   (ssc, ssc),
        'subpool':          {'type': None},
    }

    ########################
    # untuned summation: cRF
    ########################
    V = sp.linspace(0.0, 1.0, nunits)
    W = stats.norm.pdf(V, loc=V[nunits//2], scale=omega)
    W /= W.sum()
    pool_Q = {
        'type':     'conv',
        'fb':       W,
        'padding':  'wrap',
        'im_dims':  'ndhw',
        'fb_dims':  'd',
        'corr':     False }

    ###################
    # pooling functions
    ###################
    untuned_suppression = lambda arr: recursive_pool(arr,
        params=pool_U, keyname='subpool', verbose=False)
    tuned_suppression = lambda arr: recursive_pool(arr,
        params=pool_T, keyname='subpool', verbose=False)
    tuned_summation = lambda arr: recursive_pool(arr,
        params=pool_P, keyname='subpool', verbose=False)
    untuned_summation = lambda arr: recursive_pool(arr,
        params=pool_Q, keyname='subpool', verbose=False)

    relu = lambda x: hwrectify(x, '+')
    # relu = lambda x: softplus(x, 10.0)

    ###################################################
    # iterate lateral connections and store time frames
    ###################################################
    if verbose: pbar = pb(maxiter, 'Integrating [HOST]')
    for i in range(maxiter):
        U = untuned_suppression(O)
        T = tuned_suppression(O)
        P = tuned_summation(I)
        Q = untuned_summation(I)

        I_summand = relu(xi * afferent
            - (alpha * I + mu) * U
            - (beta * I + nu) * T)
        I = (1. - epsilon**2 * h/eta) * I + h/eta * I_summand

        O_summand = relu(zeta * I
            + gamma * P
            + delta * Q)
        O = (1. - sigma**2 * h/tau) * O + h/tau * O_summand

        if keeptime: I_t.append(I); O_t.append(O)
        if verbose: pbar.update(i)
    if verbose: pbar.finish()

    ################
    # postprocessing
    ################
    out_I = from4(I_t if keeptime else I,
        axis=axis, keeptime=keeptime, size=initsz)
    out_O = from4(O_t if keeptime else O,
        axis=axis, keeptime=keeptime, size=initsz)
    afferent.shape = initsz

    return out_I, out_O

#------------------------------------------------------------------------------#
def run_inter3(afferent, p=_DEFAULT_PARAMETERS, axis=-3, maxiter=_DEFAULT_MAXITER,
    h=_DEFAULT_STEPSIZE, keeptime=_DEFAULT_KEEPTIME, verbose=_DEFAULT_VERBOSE):
    """ Integrate with Forward Euler method with integration step size h
    """

    ######################################
    # re-arrange array into canonical form
    ######################################
    axis = axis % afferent.ndim
    O, initsz, nunits = to4(afferent, axis=axis)
    I, O_t, I_t = O.copy(), [], []
    Z = sp.zeros_like(I)
    if keeptime: O_t.append(O); I_t.append(I)

    ############
    # parameters
    ############
    p = _DEFAULT_PARAMETERS if p is None else p
    sigma, tau = p['sigma'], p['tau']
    epsilon, eta  = p['epsilon'], p['eta']
    ssc, sss = p['ssc'], p['sss']
    gamma, alpha, mu = p['gamma'], p['alpha'], p['mu']
    delta, beta, nu = p['delta'], p['beta'], p['nu']
    xi, zeta, omega = p['xi'], p['zeta'], p['omega']

    ##############################################
    # make sure pool sizes, input sizes make sense
    ##############################################
    assert sss < afferent.shape[-2]
    assert sss < afferent.shape[-1]
    assert ssc < sss
    assert sss % 2 == 1
    assert ssc % 2 == 1

    tuned_pooling_method = 'mean'# 'max'
    untuned_pooling_method = 'mean'# 'max'

    #################################
    # tuned summation: center pooling
    #################################
    zeta = 1.0 # because here unlike the GPU implementation we exclude center
               # and we pulled the default parameters from GPU implementation
    pool_P = {
        'type':             'pool',
        'mode':             tuned_pooling_method,
        'size':             (1, 1, ssc, ssc),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   (1, 1),
        'subpool':          {'type': None},
    }

    ####################################################
    # untuned suppression: reduction across feature axis
    ####################################################
    pool_U = {
        'type':             'pool',
        'mode':             untuned_pooling_method,
        'size':             (1, -1, 1, 1),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   None,
    }

    #####################################
    # tuned suppression: surround pooling
    #####################################
    pool_T = {
        'type':             'pool',
        'mode':             tuned_pooling_method,
        'size':             (1, 1, sss, sss),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   None,#(ssc, ssc),
        'subpool':          {'type': None},
    }

    ########################
    # untuned summation: cRF
    ########################
    V = sp.linspace(0.0, 1.0, nunits)
    W = stats.norm.pdf(V, loc=V[nunits//2], scale=omega)
    W /= W.sum()
    pool_Q = {
        'type':     'conv',
        'fb':       W,
        'padding':  'wrap',
        'im_dims':  'ndhw',
        'fb_dims':  'd',
        'corr':     False }

    ###################
    # pooling functions
    ###################
    untuned_suppression = lambda arr: recursive_pool(arr,
        params=pool_U, keyname='subpool', verbose=False)
    tuned_suppression = lambda arr: recursive_pool(arr,
        params=pool_T, keyname='subpool', verbose=False)
    tuned_summation = lambda arr: recursive_pool(arr,
        params=pool_P, keyname='subpool', verbose=False)
    untuned_summation = lambda arr: recursive_pool(arr,
        params=pool_Q, keyname='subpool', verbose=False)

    relu = lambda x: hwrectify(x, '+')
    # relu = lambda x: softplus(x, 10.0)

    ###################################################
    # iterate lateral connections and store time frames
    ###################################################

    a = sigma
    b = sp.sqrt(1)
    c = eta/5.

    if verbose: pbar = pb(maxiter, 'Integrating [HOST]')
    for i in range(maxiter):
        U = untuned_suppression(O)
        T = tuned_suppression(O)
        P = tuned_summation(I)
        Q = untuned_summation(I)

        Z_summand = relu((alpha * I + mu) * U + (beta * I + nu) * T - b**2)
        Z = (1. - a**2 * h/c) * Z + h/c * Z_summand

        I_summand = relu(xi * afferent - Z)
        I = (1. - epsilon**2 * h/eta) * I + h/eta * I_summand

        O_summand = relu(zeta * I
            + gamma * P
            + delta * Q)
        O = (1. - sigma**2 * h/tau) * O + h/tau * O_summand

        if keeptime: I_t.append(I); O_t.append(O)
        if verbose: pbar.update(i)
    if verbose: pbar.finish()

    ################
    # postprocessing
    ################
    out_I = from4(I_t if keeptime else I,
        axis=axis, keeptime=keeptime, size=initsz)
    out_O = from4(O_t if keeptime else O,
        axis=axis, keeptime=keeptime, size=initsz)
    afferent.shape = initsz

    return out_I, out_O

#------------------------------------------------------------------------------#
from .contextual_CUDA_experimental import _DEFAULT_MAXITER as _DEFAULT_MAXITER_new
from .contextual_CUDA_experimental import _DEFAULT_STEPSIZE as _DEFAULT_STEPSIZE_new
from .contextual_CUDA_experimental import _DEFAULT_PARAMETERS as _DEFAULT_PARAMETERS_new

def run(afferent, p=_DEFAULT_PARAMETERS_new, axis=-3, maxiter=_DEFAULT_MAXITER_new,
    h=_DEFAULT_STEPSIZE_new, keeptime=_DEFAULT_KEEPTIME, verbose=_DEFAULT_VERBOSE):
    """ Integrate with Forward Euler method with integration step size h
    """

    ######################################
    # re-arrange array into canonical form
    ######################################
    axis = axis % afferent.ndim
    Pyr, initsz, nunits = to4(afferent, axis=axis)
    Int = Pyr.copy()
    Pyr_t = []
    Int_t = []

    if keeptime:
        Pyr_t.append(Pyr); Int_t.append(Int)

    ############
    # parameters
    ############
    p = _DEFAULT_PARAMETERS if p is None else p
    sigma, tau = p['sigma'], p['tau']
    epsilon, eta  = p['epsilon'], p['eta']
    ssc, sss = p['ssc'], p['sss']
    gamma, alpha, mu = p['gamma'], p['alpha'], p['mu']
    delta, beta, nu = p['delta'], p['beta'], p['nu']
    xi, zeta, omega = p['xi'], p['zeta'], p['omega']
    phi, psi = p['phi'], p['psi']

    ##############################################
    # make sure pool sizes, input sizes make sense
    ##############################################
    assert sss < afferent.shape[-2]
    assert sss < afferent.shape[-1]
    assert ssc < sss
    assert sss % 2 == 1
    assert ssc % 2 == 1

    tuned_pooling_method = 'mean'# 'max'
    untuned_pooling_method = 'mean'# 'max'

    #################################
    # tuned summation: center pooling
    #################################
    pool_P = {
        'type':             'pool',
        'mode':             tuned_pooling_method,
        'size':             (1, 1, ssc, ssc),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   (1, 1),
        'subpool':          {'type': None},
    }

    ####################################################
    # untuned suppression: reduction across feature axis
    ####################################################
    pool_U = {
        'type':             'pool',
        'mode':             untuned_pooling_method,
        'size':             (1, -1, 1, 1),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   None,
    }

    #####################################
    # tuned suppression: surround pooling
    #####################################
    pool_T = {
        'type':             'pool',
        'mode':             tuned_pooling_method,
        'size':             (1, 1, sss, sss),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   (1, 1),#(ssc, ssc),
        'subpool':          {'type': None},
    }

    ########################
    # untuned summation: cRF
    ########################
    V = sp.linspace(0.0, 1.0, nunits)
    W = stats.norm.pdf(V, loc=V[nunits//2], scale=omega)
    W /= W.sum()
    pool_Q = {
        'type':     'conv',
        'fb':       W,
        'padding':  'wrap',
        'im_dims':  'ndhw',
        'fb_dims':  'd',
        'corr':     False }

    ###################
    # pooling functions
    ###################
    untuned_suppression = lambda arr: recursive_pool(arr,
        params=pool_U, keyname='subpool', verbose=False)
    tuned_suppression = lambda arr: recursive_pool(arr,
        params=pool_T, keyname='subpool', verbose=False)
    tuned_summation = lambda arr: recursive_pool(arr,
        params=pool_P, keyname='subpool', verbose=False)
    untuned_summation = lambda arr: recursive_pool(arr,
        params=pool_Q, keyname='subpool', verbose=False)

    relu = lambda x: hwrectify(x, '+')
    # relu = lambda x: softplus(x, 10.0)

    ###################################################
    # iterate lateral connections and store time frames
    ###################################################
    if verbose: pbar = pb(maxiter, 'Integrating [HOST]')
    for i in range(maxiter):
        U = untuned_suppression(Pyr)
        T = tuned_suppression(Pyr)
        P = tuned_summation(Pyr)
        Q = untuned_summation(Pyr)
        Int_summand = relu(zeta * Pyr \
            + alpha * U \
            + beta * T \
            - psi ** 2)
        Int = (1. - epsilon**2 * h/eta) * Int + h/eta * Int_summand

        Pyr_summand = relu(xi * afferent #+ 0.25 * tuned_summation(afferent) \
            + gamma * P \
            + delta * Q \
            - (mu * Pyr + nu) * Int \
            - phi ** 2)
        Pyr = (1. - sigma**2 * h/tau) * Pyr + h/tau * Pyr_summand

        if keeptime: Int_t.append(Int); Pyr_t.append(Pyr)
        if verbose: pbar.update(i)
    if verbose: pbar.finish()

    ################
    # postprocessing
    ################
    out_Int = from4(Int_t if keeptime else Int,
        axis=axis, keeptime=keeptime, size=initsz)
    out_Pyr = from4(Pyr_t if keeptime else Pyr,
        axis=axis, keeptime=keeptime, size=initsz)
    afferent.shape = initsz

    return out_Int, out_Pyr

#------------------------------------------------------------------------------#
def run_all_interneurons(afferent, axis=-3,maxiter=50, h=1., keeptime=True,
    verbose=True):
    """ Integrate with Forward Euler method with integration step size h
    """

    ######################################
    # re-arrange array into canonical form
    ######################################
    axis = axis % afferent.ndim
    Pyr, initsz, nunits = to4(afferent, axis=axis)
    Pyr_t = []
    # base_shape = Pyr.shape
    # intr_shape = Pyr.shape
    # intr_shape[-3] = 1
    # Som, Som_t = sp.zeros(base_shape), []
    # Pvb, Pvb_t = sp.zeros(intr_shape), []
    # Vip, Vip_t = sp.zeros(intr_shape), []

    if keeptime:
        Pyr_t.append(Pyr)
        # Som_t.append(Som)
        # Pvb_t.append(Pvb)
        # Vip_t.append(Vip)

    ############
    # parameters
    ############

    ssc = 9
    sss = 29
    tau = 5.00
    sigma = 1.00
    omega = 0.15
    k_FF_Pyr = 2.00
    k_SE_Pyr = 1.90
    k_SI_Pyr = 2.00
    k_HE_Pyr = 1.00
    k_HI_Pyr = 3.00

    ##############################################
    # make sure pool sizes, input sizes make sense
    ##############################################
    assert sss < afferent.shape[-2]
    assert sss < afferent.shape[-1]
    assert ssc < sss
    assert sss % 2 == 1
    assert ssc % 2 == 1

    tuned_pooling_method = 'mean'# 'max'
    untuned_pooling_method = 'mean'# 'max'

    #################################
    # tuned summation: center pooling
    #################################
    pool_P = {
        'type':             'pool',
        'mode':             tuned_pooling_method,
        'size':             (1, 1, ssc, ssc),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   (1, 1),
        'subpool':          {'type': None},
    }

    ####################################################
    # untuned suppression: reduction across feature axis
    ####################################################
    pool_U = {
        'type':             'pool',
        'mode':             untuned_pooling_method,
        'size':             (1, -1, 1, 1),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   None,
    }

    #####################################
    # tuned suppression: surround pooling
    #####################################
    pool_T = {
        'type':             'pool',
        'mode':             tuned_pooling_method,
        'size':             (1, 1, sss, sss),
        'padding':          'reflect',
        'stride_size':      None,
        'keepdims':         True,
        'exclude_center':   None,#(ssc, ssc),
        'subpool':          {'type': None},
    }

    ########################
    # untuned summation: cRF
    ########################
    V = sp.linspace(0.0, 1.0, nunits)
    W = stats.norm.pdf(V, loc=V[nunits//2], scale=omega)
    W /= W.sum()
    pool_Q = {
        'type':     'conv',
        'fb':       W,
        'padding':  'wrap',
        'im_dims':  'ndhw',
        'fb_dims':  'd',
        'corr':     False }

    ###################
    # pooling functions
    ###################
    untuned_suppression = lambda arr: recursive_pool(arr,
        params=pool_U, keyname='subpool', verbose=False)
    tuned_suppression = lambda arr: recursive_pool(arr,
        params=pool_T, keyname='subpool', verbose=False)
    tuned_summation = lambda arr: recursive_pool(arr,
        params=pool_P, keyname='subpool', verbose=False)
    untuned_summation = lambda arr: recursive_pool(arr,
        params=pool_Q, keyname='subpool', verbose=False)
    relu = lambda x: hwrectify(x, '+')

    ###################################################
    # iterate lateral connections and store time frames
    ###################################################

    if verbose: pbar = pb(maxiter, 'Integrating [HOST]')
    for i in range(maxiter):
        U = untuned_suppression(Pyr)
        T = tuned_suppression(Pyr)
        P = tuned_summation(Pyr)
        Q = untuned_summation(Pyr)

        Pyr_dendritic = relu(k_FF_Pyr * afferent \
            + k_HE_Pyr * Q \
            + k_SE_Pyr * P \
            - k_SI_Pyr * T)
        Pyr_summand = relu(Pyr_dendritic \
            - k_HI_Pyr * U)
        Pyr = (1. - sigma**2 * h/tau) * Pyr + h/tau * Pyr_summand

        if keeptime: Pyr_t.append(Pyr)
        if verbose: pbar.update(i)
    if verbose: pbar.finish()

    ################
    # postprocessing
    ################
    out_Pyr = from4(Pyr_t if keeptime else Pyr,
        axis=axis, keeptime=keeptime, size=initsz)
    afferent.shape = initsz

    return out_Pyr

#------------------------------------------------------------------------------#
