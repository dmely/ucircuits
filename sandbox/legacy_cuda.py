#!/usr/bin/env python

""" Authors: David A. Mely  <david_mely@brown.edu>
             Thomas Serre   <thomas_serre@brown.edu>
"""

from __future__ import absolute_import
try:
    import colored_traceback.always
except ImportError:
    pass
import os
import ctypes
from copy import deepcopy
from functools import wraps
from collections import namedtuple
import numpy as np
import scipy as sp
from scipy import stats
from scipy import array as ar
from hmax.tools.utils import pb
from hmax.tools.utils import ifloor, iceil, iround
from pycuda import driver as cuda_driver
from pycuda import gpuarray
from pycuda import cumath
from pycuda.compiler import SourceModule
from Cheetah.Template import Template
from hmax.backend._cudnn_ import libcudnn as cudnn
from skcuda import cublas

_DEFAULT_GPU = 1
_DEFAULT_FLOATX = 'float32'
_DEFAULT_VERBOSE = True
_DEFAULT_KEEPTIME = False   # whether to keep time series of X and Y
_DEFAULT_KEEPVARS = False   # whether to keep time series of accessory vars
_DEFAULT_MAXITER = 100      # (comfortable value: 200)
_DEFAULT_STEPSIZE = 3.      # (comfortable value: 1.)

_PARAMETER_SET_VERSION = 'paper'
if _PARAMETER_SET_VERSION == 'paper':
    _SRF = 1
    _SSN = 9
    _SSF = 29
    _K_P = 1.00
    _K_T = 1.00
elif _PARAMETER_SET_VERSION == 'v2':
    _SRF = 3
    _SSN = 9
    _SSF = 21
    _K_P = 1.00
    _K_T = 1.50
else:
    raise ValueError('Invalid value for _PARAMETER_SET_VERSION')

_DEFAULT_PARAMETERS = {
    'tau':          6.00,        # X: time constant
    'sigma':        0.50,        # X: saturation/decay constant
    'eta':          6.00,        # Y: time constant
    'epsilon':      0.50,        # Y: saturation/decay constant
    'xi':           4.50,        # L -> X baseline afferent strength
    'zeta':         0.00,        # X -> Y supplementary afferent excitation
    'gamma':        1.00 * _K_P, # P strength [tuned summation]
    'delta':        1.00,        # Q strength [untuned summation]
    'alpha':        1.00,        # U strength (linear)
    'mu':           1.00,        # U strength (constant)
    'beta':         3.00 * _K_T, # T strength (linear)
    'nu':           0.30 * _K_T, # T strength (constant)
    'srf':          _SRF,        # extent of cRF (i.e., minimal response field)
    'ssn':          _SSN,        # extent of near surround
    'ssf':          _SSF,        # extent of far surround
    'omega':        0.15,        # spread of weights for supp. aff. exc.
    'continuous':   True,        # feature space is continuously parametrized?
}

CircuitParameters = namedtuple('CircuitParameters',
    _DEFAULT_PARAMETERS.keys(), verbose=False, rename=False)
_DEFAULT_PARAMETERS_TEMPLATE = deepcopy(_DEFAULT_PARAMETERS)

if _DEFAULT_FLOATX == 'float32':
    from skcuda.cublas import cublasSaxpy as cuaxpy
    from skcuda.cublas import cublasScopy as cucopy
    from skcuda.cublas import cublasSscal as cuscal
elif _DEFAULT_FLOATX == 'float64':
    from skcuda.cublas import cublasDaxpy as cuaxpy
    from skcuda.cublas import cublasDcopy as cucopy
    from skcuda.cublas import cublasDscal as cuscal


#------------------------------------------------------------------------------#
def _array2gpu(host_array, dtype=_DEFAULT_FLOATX):
    """ Send array from host to GPU and enforce dtype and C-order """
    if not isinstance(host_array, gpuarray.GPUArray):
        return gpuarray.to_gpu(sp.array(host_array, order='C', dtype=dtype))
    else:
        return host_array

def _empty2gpu(array_shape, dtype=_DEFAULT_FLOATX):
    """ Create empty GPUArray """
    return gpuarray.empty(array_shape, order='C', dtype=dtype)

def _gpuarray2ptr(gpu_array):
    """ Pointer to GPUArray data for cuDNN """
    return ctypes.c_void_p(int(gpu_array.gpudata))

def _sgw(k, s):
    """ Shifted histogram of Gaussian weights, centered appropriately """
    x = sp.linspace(0.0, 1.0, k)
    if s == sp.inf:
        w = sp.ones((k,)) / float(k)
    else:
        w = stats.norm.pdf(x, loc=x[k//2], scale=s)
    return sp.roll(w / w.sum(), shift=int(sp.ceil(k/2.0)))

def _sdw(k, s):
    """ Shifted histogram of discontinuous weights, centered appropriately """
    g1 = _sgw(k=k, s=s).max()
    g2 = (1.0 - g1) / (k - 1)
    return sp.array([g1] + [g2] * (k- 1))

'''
def _cg2d(sz, exclude_center=True):
    x, y = sp.mgrid[-1.0:1.0:sz*1j, -1.0:1.0:sz*1j]
    w = sp.exp(-(x**2 + y**2)/0.5)
    if exclude_center:
        w[sz//2, sz//2] = 0.0
    w /= w.sum()
    return w

def _ag2d(sz):
    x, y = sp.mgrid[-1.0:1.0:sz*1j, -1.0:1.0:sz*1j]
    r = sp.sqrt(x**2 + y**2)
    w = sp.exp(-(sp.log(r) - sp.log(0.5))**2/0.25)
    w /= w.sum()
    return w
'''

class ContextualCircuit(object):
    """ Returns a simulator for the Mely-Serre circuit that runs on CUDA. """

    def __init__(self, input_shape=None, i_gpu=_DEFAULT_GPU, cudaDevice=None,
        cudaContext=None, cudnnContext=None, cublasContext=None, verbose=None,
        maxiter=None, keeptime=None, keepvars=None, stepsize=None, parameters=None):
        """ Initialize contexts if needed """

        self.i_gpu = i_gpu
        self.floatX = _DEFAULT_FLOATX
        self.verbose = verbose
        self.maxiter = maxiter
        self.keeptime = keeptime
        self.keepvars = keepvars
        self.stepsize = stepsize
        self.input_shape = input_shape
        self.cudaDevice = cudaDevice
        self.cudaContext = cudaContext
        self.cudnnContext = cudnnContext
        self.cublasContext = cublasContext
        ############################################################
        # define circuit parameters as a constant named tuple rather
        # than a dict to avoid inappropriate modification by user
        ############################################################
        try:
            for pkey, pval in parameters.iteritems():
                _DEFAULT_PARAMETERS_TEMPLATE[pkey] = pval
        except AttributeError:
            pass
        finally:
            self.parameters = CircuitParameters(
                **_DEFAULT_PARAMETERS_TEMPLATE)

        if self.verbose is None:
            self.verbose = _DEFAULT_VERBOSE
        if self.maxiter is None:
            self.maxiter = _DEFAULT_MAXITER
        if self.keeptime is None:
            self.keeptime = _DEFAULT_KEEPTIME
        if self.keepvars is None:
            self.keepvars = _DEFAULT_KEEPVARS
        if self.stepsize is None:
            self.stepsize = _DEFAULT_STEPSIZE

        if self.input_shape is not None:
            self._sanity_check()

        if self.cudaDevice is None:
            cuda_driver.init()
            self.cudaDevice = cuda_driver.Device(self.i_gpu)
        if self.cudaContext is None:
            self.cudaContext = self.cudaDevice.make_context()
        if self.cudnnContext is None:
            self.cudnnContext = cudnn.cudnnCreate()
        if self.cublasContext is None:
            self.cublasContext = cublas.cublasCreate()

        # if input shape is known, initialize now
        if self.input_shape is not None:
            self._prepare_kernels()
            self._prepare_tensors()

    #------------------------------------------------------------#
    def __del__(self):
        """ Make sure contexts are released """
        # apparently the self.release() method is not a good idea
        # let's use the with context statement instead
        # http://eli. thegreenplace.net/2009/06/12/
        # safely-using-destructors-in-python
        return # self.release()

    #------------------------------------------------------------#
    def __enter__(self):
        """ Nothing to do here """
        return self

    #------------------------------------------------------------#
    def __exit__(self, type, value, traceback):
        """ When leaving, clean up the GPU """
        self.release()

    #------------------------------------------------------------#
    def _prepare_kernels(self):
        """ Prepare the kernels and load the input """

        # compile template
        n, k, h, w = self.input_shape
        templ_pars = {'N': n, 'K': k, 'H': h, 'W': w}
        templ_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'bcastmul.cu')
        cuda_code = Template(
            file=templ_path, searchList=[templ_pars])
        compiled_cuda = SourceModule(cuda_code)
        self._bcastbias_cuda = compiled_cuda.get_function(
            'bcastbias_inplace')
        self._bcastmul_cuda = compiled_cuda.get_function(
            'bcastmul_inplace')
        self._pwisemul_cuda = compiled_cuda.get_function(
            'pwisemul_inplace')

        # determine block, grid size
        max_block_size = 32#min(2**ifloor(sp.log2(min(h, w))), 32)#32
        grid_z = h//max_block_size + int(bool(h % max_block_size))
        grid_z *= w//max_block_size + int(bool(w % max_block_size))
        self._bl = (max_block_size, max_block_size, 1)
        self._gr = (n, k, grid_z)

    #------------------------------------------------------------#
    def _prepare_tensors(self):
        """ Allocate buffer space on the GPU, etc. """

        n, k, h, w = self.input_shape
        SRF = self.parameters.srf
        SSN = self.parameters.ssn
        SSF = self.parameters.ssf
        OMEGA = self.parameters.omega
        ISCONTINUOUS = self.parameters.continuous

        # computation parameters
        ########################
        self._cudnn_tensor_format = cudnn.cudnnTensorFormat[
            'CUDNN_TENSOR_NCHW']
        self._cudnn_pooling_mode = cudnn.cudnnPoolingMode[
            'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING']
        self._cudnn_relu_act = cudnn.cudnnActivationMode[
            'CUDNN_ACTIVATION_RELU']
        self._cudnn_data_type = cudnn.cudnnDataType[
            'CUDNN_DATA_FLOAT'] if self.floatX == 'float32' \
                else cudnn.cudnnDataType['CUDNN_DATA_DOUBLE']
        self._cudnn_conv_mode = cudnn.cudnnConvolutionMode[
            'CUDNN_CROSS_CORRELATION']
        # self._cudnn_conv_pref = cudnn.cudnnConvolutionFwdPreference[
        #     'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST']
        self._cudnn_conv_pref = cudnn.cudnnConvolutionFwdPreference[
            'CUDNN_CONVOLUTION_FWD_NO_WORKSPACE']

        # constant tensors
        ##################
        self._gpu_negMU = _array2gpu([-1.0 * self.parameters.mu])
        self._gpu_negNU = _array2gpu([-1.0 * self.parameters.nu])

        # return tensors
        ################
        self.X = _empty2gpu((n, k, h, w))
        self.Y = _empty2gpu((n, k, h, w))
        self._desc_X = cudnn.cudnnCreateTensorDescriptor()
        self._desc_Y = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(self._desc_X,
            self._cudnn_tensor_format, self._cudnn_data_type, n, k, h, w)
        cudnn.cudnnSetTensor4dDescriptor(self._desc_Y,
            self._cudnn_tensor_format, self._cudnn_data_type, n, k, h, w)

        if self.keeptime:
            self.X_t = sp.zeros((self.maxiter+1, n, k, h, w))
            self.Y_t = sp.zeros((self.maxiter+1, n, k, h, w))
        if self.keepvars:
            self.T_t = sp.zeros((self.maxiter+1, n, k, h, w))
            self.U_t = sp.zeros((self.maxiter+1, n, k, h, w))
            self.P_t = sp.zeros((self.maxiter+1, n, k, h, w))
            self.Q_t = sp.zeros((self.maxiter+1, n, k, h, w))

        # buffer tensors
        ################
        self._gpu_buf1 = _empty2gpu((n, k, h, w))
        self._gpu_buf2 = _empty2gpu((n, k, h, w))
        self._gpu_buf3 = _empty2gpu((n, k, h, w))
        self._gpu_buf4 = _empty2gpu((n, 1, h, w))
        self._desc_buf1 = cudnn.cudnnCreateTensorDescriptor()
        self._desc_buf2 = cudnn.cudnnCreateTensorDescriptor()
        self._desc_buf3 = cudnn.cudnnCreateTensorDescriptor()
        self._desc_buf4 = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(
            self._desc_buf1,
            self._cudnn_tensor_format,
            self._cudnn_data_type, n, k, h, w)
        cudnn.cudnnSetTensor4dDescriptor(
            self._desc_buf2,
            self._cudnn_tensor_format,
            self._cudnn_data_type, n, k, h, w)
        cudnn.cudnnSetTensor4dDescriptor(
            self._desc_buf3,
            self._cudnn_tensor_format,
            self._cudnn_data_type, n, k, h, w)
        cudnn.cudnnSetTensor4dDescriptor(
            self._desc_buf4,
            self._cudnn_tensor_format,
            self._cudnn_data_type, n, 1, h, w)

        # broadly-tuned summation
        #########################
        if self.parameters.omega:
            weights = _sgw(k=k, s=OMEGA) \
                if ISCONTINUOUS else _sdw(k=k, s=OMEGA)
            q_array = sp.array([sp.roll(weights,
                shift=shift) for shift in range(k)])
            q_array.shape = (k, k, 1, 1)
            self._gpu_q = _array2gpu(q_array)
            self._desc_q = cudnn.cudnnCreateFilterDescriptor()
            self._desc_Q = cudnn.cudnnCreateConvolutionDescriptor()
            cudnn.cudnnSetFilter4dDescriptor(
                self._desc_q, self._cudnn_data_type, k, k, 1, 1)
            cudnn.cudnnSetConvolution2dDescriptor(
                self._desc_Q, 0, 0, 1, 1, 1, 1, self._cudnn_conv_mode)
            self._algo_q = cudnn.cudnnGetConvolutionForwardAlgorithm(
                self.cudnnContext, self._desc_X, self._desc_q, self._desc_Q,
                self._desc_buf1, self._cudnn_conv_pref, 0)

            assert cudnn.cudnnGetConvolution2dForwardOutputDim(
                self._desc_Q, self._desc_X, self._desc_q) == (n, k, h, w)

        # untuned suppression: reduction across feature axis
        ####################################################
        self._gpu_u = _array2gpu(1.0/k * sp.ones((1, k, 1, 1)))
        self._desc_u = cudnn.cudnnCreateFilterDescriptor()
        self._desc_U = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetFilter4dDescriptor(
            self._desc_u, self._cudnn_data_type, 1, k, 1, 1)
        cudnn.cudnnSetConvolution2dDescriptor(
            self._desc_U, 0, 0, 1, 1, 1, 1, self._cudnn_conv_mode)
        self._algo_u = cudnn.cudnnGetConvolutionForwardAlgorithm(
            self.cudnnContext, self._desc_Y, self._desc_u, self._desc_U,
            self._desc_buf4, self._cudnn_conv_pref, 0)

        assert cudnn.cudnnGetConvolution2dForwardOutputDim(
            self._desc_U, self._desc_Y, self._desc_u) == (n, 1, h, w)

        # tuned summation: pooling in h, w dimensions
        #############################################
        SSN_ = 2 * ifloor(SSN/2.0) + 1
        p_array = sp.zeros((k, k, SSN_, SSN_))
        # Uniform weights
        #----------------
        for pdx in range(k):
            p_array[pdx, pdx, :SSN, :SSN] = 1.0
        p_array[:, :, # exclude classical receptive field!
            SSN//2-ifloor(SRF/2.0):SSN//2+iceil(SRF/2.0),
            SSN//2-ifloor(SRF/2.0):SSN//2+iceil(SRF/2.0)] = 0.0
        p_array /= SSN**2 - SRF**2 # normalize to get true average
        # Gaussian weights
        #-----------------
        # for pdx in range(k):
        #     p_array[pdx, pdx] = _cg2d(SSN)

        self._gpu_p = _array2gpu(p_array)
        self._desc_p = cudnn.cudnnCreateFilterDescriptor()
        self._desc_P = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetFilter4dDescriptor(
            self._desc_p, self._cudnn_data_type, k, k, SSN_, SSN_)
        cudnn.cudnnSetConvolution2dDescriptor(
            self._desc_P, SSN_//2, SSN_//2, 1, 1, 1, 1, self._cudnn_conv_mode)
        self._algo_p = cudnn.cudnnGetConvolutionForwardAlgorithm(
            self.cudnnContext, self._desc_X, self._desc_p,
            self._desc_P, self._desc_buf1, self._cudnn_conv_pref, 0)
        # self._algo_p = self._cudnn_conv_pref

        assert cudnn.cudnnGetConvolution2dForwardOutputDim(
            self._desc_P, self._desc_X, self._desc_p) == (n, k, h, w)

        # tuned suppression: pooling in h, w dimensions
        ###############################################
        SSF_ = 2 * ifloor(SSF/2.0) + 1
        t_array = sp.zeros((k, k, SSF_, SSF_))
        # Uniform weights
        #----------------
        for tdx in range(k):
            t_array[tdx, tdx, :SSF, :SSF] = 1.0
        t_array[:, :, # exclude near surround!
            SSF//2-ifloor(SSN/2.0):SSF//2+iceil(SSN/2.0),
            SSF//2-ifloor(SSN/2.0):SSF//2+iceil(SSN/2.0)] = 0.0
        t_array /= SSF**2 - SSN**2 # normalize to get true average
        # Gaussian weights
        #-----------------
        # for tdx in range(k):
        #     t_array[tdx, tdx] = _cg2d(SSF)

        self._gpu_t = _array2gpu(t_array)
        self._desc_t = cudnn.cudnnCreateFilterDescriptor()
        self._desc_T = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetFilter4dDescriptor(
            self._desc_t, self._cudnn_data_type, k, k, SSF_, SSF_)
        cudnn.cudnnSetConvolution2dDescriptor(
            self._desc_T, SSF_//2, SSF_//2, 1, 1, 1, 1, self._cudnn_conv_mode)
        # self._scal_T_alpha = 1.0/(SSF**2.0 - SSN**2.0)
        # self._scal_T_beta = -1.0/(SSF**2.0 - SSN**2.0)
        self._algo_t = cudnn.cudnnGetConvolutionForwardAlgorithm(
            self.cudnnContext, self._desc_Y, self._desc_t,
            self._desc_T, self._desc_buf3, self._cudnn_conv_pref, 0)
        # self._algo_t = self._cudnn_conv_pref

        assert cudnn.cudnnGetConvolution2dForwardOutputDim(
            self._desc_T, self._desc_Y, self._desc_t) == (n, k, h, w)

    #------------------------------------------------------------#
    def destroy_tensors(self):
        """ Clean-up buffer tensors that take up GPU memory """

        del self._gpu_buf1, self._gpu_buf2, self._gpu_buf3, \
            self._gpu_buf4, self.X, self.Y, \
            self._gpu_input, self._gpu_negMU, self._gpu_negNU, \
            self._gpu_u, self._gpu_t, self._gpu_p

        if self.parameters.omega:
            del self._gpu_q
            cudnn.cudnnDestroyPoolingDescriptor(self._desc_Q)
            cudnn.cudnnDestroyPoolingDescriptor(self._desc_q)

        cudnn.cudnnDestroyTensorDescriptor(self._desc_X)
        cudnn.cudnnDestroyTensorDescriptor(self._desc_Y)
        cudnn.cudnnDestroyTensorDescriptor(self._desc_buf1)
        cudnn.cudnnDestroyTensorDescriptor(self._desc_buf2)
        cudnn.cudnnDestroyTensorDescriptor(self._desc_buf3)
        cudnn.cudnnDestroyTensorDescriptor(self._desc_buf4)
        cudnn.cudnnDestroyPoolingDescriptor(self._desc_T)
        cudnn.cudnnDestroyPoolingDescriptor(self._desc_P)
        cudnn.cudnnDestroyPoolingDescriptor(self._desc_U)
        cudnn.cudnnDestroyPoolingDescriptor(self._desc_t)
        cudnn.cudnnDestroyPoolingDescriptor(self._desc_p)
        cudnn.cudnnDestroyPoolingDescriptor(self._desc_u)

    #------------------------------------------------------------#
    def destroy_contexts(self):
        """ Destroy context """

        cublas.cublasDestroy(self.cublasContext)
        cudnn.cudnnDestroy(self.cudnnContext)
        self.cudaContext.pop()

    #------------------------------------------------------------#
    def release(self):
        """ Destroy both tensors and context"""

        self.destroy_tensors()
        self.destroy_contexts()

    #------------------------------------------------------------#
    def _sanity_check(self):
        """ Make sure the input makes sense """

        try:
            n, k, h, w = self.input_shape
        except ValueError:
            raise ValueError('Input array must be 4-tensor')
        srf = self.parameters.srf
        ssn = self.parameters.ssn
        ssf = self.parameters.ssf

        assert ssf < h
        assert ssf < w
        assert srf < ssn < ssf
        assert self.maxiter > 0
        assert self.stepsize > 0

    #------------------------------------------------------------#
    def run(self, in_array, from_gpu=True):
        """ Do numerical integration
        """

        assert in_array.ndim == 4
        if self.input_shape is None:
            self.input_shape = in_array.shape
            self._sanity_check()
            self._prepare_kernels()
            self._prepare_tensors()

        SSN = self.parameters.ssn
        SSF = self.parameters.ssf
        ETA = self.parameters.eta
        TAU = self.parameters.tau
        EPSILON = self.parameters.epsilon
        SIGMA = self.parameters.sigma
        DELTA = self.parameters.delta
        GAMMA = self.parameters.gamma
        ALPHA = self.parameters.alpha
        BETA = self.parameters.beta
        ZETA = self.parameters.zeta
        OMEGA = self.parameters.omega
        XI = self.parameters.xi

        # load copies of input into GPU
        self._gpu_input = _array2gpu(in_array)

        cucopy(self.cublasContext, self._gpu_input.size,
            self._gpu_input.gpudata, 1, self.X.gpudata, 1)
        self.cudaContext.synchronize()

        cucopy(self.cublasContext, self._gpu_input.size,
            self._gpu_input.gpudata, 1, self.Y.gpudata, 1)
        self.cudaContext.synchronize()

        if self.keeptime:
            try:
                self.X_t[0] = in_array.get()
                self.Y_t[0] = in_array.get()
            except AttributeError:
                self.X_t[0] = in_array
                self.Y_t[0] = in_array

        # create a bunch of pointers for cuDNN
        X__ptr__ = _gpuarray2ptr(self.X)
        Y__ptr__ = _gpuarray2ptr(self.Y)
        u__ptr__ = _gpuarray2ptr(self._gpu_u)
        p__ptr__ = _gpuarray2ptr(self._gpu_p)
        t__ptr__ = _gpuarray2ptr(self._gpu_t)
        buf1__ptr__ = _gpuarray2ptr(self._gpu_buf1)
        buf2__ptr__ = _gpuarray2ptr(self._gpu_buf2)
        buf3__ptr__ = _gpuarray2ptr(self._gpu_buf3)
        buf4__ptr__ = _gpuarray2ptr(self._gpu_buf4)
        if self.parameters.omega:
            q__ptr__ = _gpuarray2ptr(self._gpu_q)

        if self.verbose: pbar = pb(self.maxiter,
            'Integrating [GPU:%i]' % (self.i_gpu,))
        for idx in range(self.maxiter):

            # [-(alpha*X+mu) -> B2] <<<PASS>>>
            ###########################################################
            cucopy(self.cublasContext, self.X.size,
                self.X.gpudata, 1, self._gpu_buf2.gpudata, 1)
            self.cudaContext.synchronize()

            cuscal(self.cublasContext, self._gpu_buf2.size,
                -ALPHA, self._gpu_buf2.gpudata, 1)
            self.cudaContext.synchronize()

            self._bcastbias_cuda(self._gpu_buf2, self._gpu_negMU,
                block=self._bl, grid=self._gr)
            self.cudaContext.synchronize()

            # [compute(U); U -> B4] <<<PASS:max|ERR|<1e-7>>>
            ###########################################################
            cudnn.cudnnConvolutionForward(self.cudnnContext, 1.0, self._desc_Y,
                Y__ptr__, self._desc_u, u__ptr__, self._desc_U, self._algo_u,
                None, 0, 0.0, self._desc_buf4, buf4__ptr__, self._cudnn_data_type)
            self.cudaContext.synchronize()

            # [B2 *= B4 := U] <<<PASS>>>
            ###########################################################
            self._bcastmul_cuda(self._gpu_buf2, self._gpu_buf4,
                block=self._bl, grid=self._gr)
            self.cudaContext.synchronize()

            if self.keepvars:
                self.U_t[idx] = -1.0 * self._gpu_buf2.get()

            # [XI * L -> B1] <<<PASS>>>
            ###########################################################
            cucopy(self.cublasContext, self._gpu_input.size,
                self._gpu_input.gpudata, 1, self._gpu_buf1.gpudata, 1)
            self.cudaContext.synchronize()

            cuscal(self.cublasContext, self._gpu_buf1.size,
                XI, self._gpu_buf1.gpudata, 1)
            self.cudaContext.synchronize()


            ###########################################################
            ###########################################################
            # import warnings
            # warnings.warn('Shunting inhibition introduced ' + \
            #     'as an experimental feature!!!')
            # cucopy(self.cublasContext,
            #     self.X.size,self.X.gpudata, 1, self._gpu_buf3.gpudata, 1)
            # self.cudaContext.synchronize()
            # self._pwisemul_cuda(self._gpu_buf3,
            #     self._gpu_input, block=self._bl, grid=self._gr)
            # self.cudaContext.synchronize()
            # cuscal(self.cublasContext,
            #     self._gpu_buf3.size, -XI*0.5, self._gpu_buf3.gpudata, 1)
            # self.cudaContext.synchronize()
            # cuaxpy(self.cublasContext, self._gpu_buf3.size, 1.0,
            #     self._gpu_buf3.gpudata, 1, self._gpu_buf1.gpudata, 1)
            # self.cudaContext.synchronize()
            ###########################################################
            ###########################################################


            # [B1 += B2 := -(alpha*X+mu).*U] <<<PASS>>>
            ###########################################################
            cuaxpy(self.cublasContext, self._gpu_buf2.size, 1.0,
                self._gpu_buf2.gpudata, 1, self._gpu_buf1.gpudata, 1)
            self.cudaContext.synchronize()

            # [-(beta*X+nu) -> B2] <<<PASS>>>
            ###########################################################
            cucopy(self.cublasContext, self.X.size,
                self.X.gpudata, 1, self._gpu_buf2.gpudata, 1)
            self.cudaContext.synchronize()

            cuscal(self.cublasContext, self._gpu_buf2.size,
                -BETA, self._gpu_buf2.gpudata, 1)
            self.cudaContext.synchronize()

            self._bcastbias_cuda(self._gpu_buf2, self._gpu_negNU,
                block=self._bl, grid=self._gr)
            self.cudaContext.synchronize()

            # [T<excluding_center> -> B3] <<<PASS:max|ERR|<1e-3,avg=1e-5>>>
            ###########################################################
            cudnn.cudnnConvolutionForward(self.cudnnContext, 1.0,
                self._desc_Y, Y__ptr__, self._desc_t, t__ptr__, self._desc_T,
                self._algo_t, None, 0, 0.0, self._desc_buf3,
                buf3__ptr__, self._cudnn_data_type)
            self.cudaContext.synchronize()

            # [B2 *= B3 := T] <<<PASS>>>
            ###########################################################
            self._pwisemul_cuda(self._gpu_buf2, self._gpu_buf3,
                block=self._bl, grid=self._gr)
            self.cudaContext.synchronize()

            if self.keepvars:
                self.T_t[idx] = -1.0 * self._gpu_buf2.get()

            # [B1 += B2 := -(beta*X+nu).*T] <<<PASS>>>
            ###########################################################
            cuaxpy(self.cublasContext, self._gpu_buf2.size, 1.0,
                self._gpu_buf2.gpudata, 1, self._gpu_buf1.gpudata, 1)
            self.cudaContext.synchronize()

            # [now B1 := X_summand; rectify(B1) -> B2] <<<PASS>>>
            ###########################################################
            cudnn.cudnnActivationForward(self.cudnnContext,
                self._cudnn_relu_act, 1.0, self._desc_buf1, buf1__ptr__,
                0.0, self._desc_buf2, buf2__ptr__)
            self.cudaContext.synchronize()

            # [B2 *= h/eta] <<<PASS>>>
            ###########################################################
            cuscal(self.cublasContext, self._gpu_buf2.size,
                self.stepsize * 1.0/ETA, self._gpu_buf2.gpudata, 1)
            self.cudaContext.synchronize()

            # [X *= (1-epsilon**2 * h/eta)] <<<PASS>>>
            ###########################################################
            cuscal(self.cublasContext, self.X.size,
                (1.0 - EPSILON**2 * self.stepsize * 1.0/ETA), self.X.gpudata, 1)
            self.cudaContext.synchronize()

            # [X += B2 := h/eta * X_summand] <<<PASS>>>
            ###########################################################
            cuaxpy(self.cublasContext, self._gpu_buf2.size, 1.0,
                self._gpu_buf2.gpudata, 1, self.X.gpudata, 1)
            self.cudaContext.synchronize()

            # [X done; X -> B1] <<<ASSUMED_PASS>>>
            ###########################################################
            cucopy(self.cublasContext, self.X.size,
                self.X.gpudata, 1, self._gpu_buf1.gpudata, 1)
            self.cudaContext.synchronize()

            # [B1 = zeta * B1 + gamma * P] <<<ASSUMED_PASS:max|ERR|<1e-7>>>
            ###########################################################
            cudnn.cudnnConvolutionForward(self.cudnnContext, GAMMA,
                self._desc_X, X__ptr__, self._desc_p, p__ptr__, self._desc_P,
                self._algo_p, None, 0, ZETA, self._desc_buf1, buf1__ptr__,
                self._cudnn_data_type)
            self.cudaContext.synchronize()

            if self.keepvars:
                self.Q_t[idx] = self._gpu_buf1.get()
                self.P_t[idx] = self.Q_t[idx] - ZETA*self.X.get()

            # [B1 = 1.0 * B1 + delta * Q] <<<ASSUMED_PASS>>>
            ###########################################################
            if self.parameters.omega:
                cudnn.cudnnConvolutionForward(self.cudnnContext, DELTA,
                    self._desc_X, X__ptr__, self._desc_q, q__ptr__,
                    self._desc_Q, self._algo_q, None, 0, 1.0,
                    self._desc_buf1, buf1__ptr__, self._cudnn_data_type)
                self.cudaContext.synchronize()

            if self.keepvars:
                self.Q_t[idx] = self._gpu_buf1.get() - self.Q_t[idx]

            # [rectify(B1) -> B2] <<<PASS>>>
            ###########################################################
            cudnn.cudnnActivationForward(self.cudnnContext,
                self._cudnn_relu_act, 1.0, self._desc_buf1, buf1__ptr__,
                0.0, self._desc_buf2, buf2__ptr__)
            self.cudaContext.synchronize()

            # [now B2 := Y_summand; B2 *= h/tau] <<<PASS>>>
            ###########################################################
            cuscal(self.cublasContext, self._gpu_buf2.size,
                self.stepsize * 1.0/TAU, self._gpu_buf2.gpudata, 1)
            self.cudaContext.synchronize()

            # [Y *= (1-sigma**2 * h/tau)] <<<PASS>>>
            ###########################################################
            cuscal(self.cublasContext, self.Y.size,
                (1.0 - SIGMA**2 * self.stepsize * 1.0/TAU), self.Y.gpudata, 1)
            self.cudaContext.synchronize()

            # [Y += B2 := h/tau *Y _summand; then Y is done] <<<PASS>>>
            ###########################################################
            cuaxpy(self.cublasContext, self._gpu_buf2.size, 1.0,
                self._gpu_buf2.gpudata, 1, self.Y.gpudata, 1)
            self.cudaContext.synchronize()

            if self.keeptime:
                self.X_t[idx + 1] = self.X.get()
                self.Y_t[idx + 1] = self.Y.get()

            if self.verbose: pbar.update(idx)
        if self.verbose: pbar.finish()

        if from_gpu:
            self.X = self.X.get()
            self.Y = self.Y.get()


#------------------------------------------------------------------------------#
'''
import caffe
class ContextualCircuitLayer(caffe.Layer):
    """ Implements a layer for Caffe """

    def setup(self, bottom, top):
        self.cx = ContextualCircuit(
            input_shape=bottom[0].data.shape,
            verbose=True, keeptime=False)

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        self.cx.run(bottom[0].data, from_gpu=False)
        top[0].data[:] = self.cx.Y.get()

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[:] = top[0].diff
'''

#------------------------------------------------------------------------------#
def accuracy_test():
    """ Test the accuracy against a reference implementation on host """

    import matplotlib
    import matplotlib.pyplot as plt
    from scipy import random
    from matplotlib import cm
    import colormaps as custom_colormaps
    # from contextual_CUDA_legacy import integrate_on_host
    from hmax.models.ucircuits.contextual import contextual_CPU
    cm.register_cmap(name='viridis', cmap=custom_colormaps.viridis)
    plt.rcParams['image.cmap'] = 'viridis'
    plt.rcParams['axes.grid'] = False

    sz = 64
    pad, mode = _DEFAULT_PARAMETERS['ssf']//2, 'reflect'
    in_array = sp.array(random.rand(
        10, 20, sz, sz), order='C', dtype=_DEFAULT_FLOATX)
    in_array_padded = np.pad(in_array, ((0, 0), (0, 0),
        (pad, pad), (pad, pad)), mode=mode)

    print "*" * 80
    print "Running CUDA 7.5 implementation"
    print "*" * 80
    cx = ContextualCircuit(input_shape=in_array_padded.shape)
    cx.run(in_array_padded, from_gpu=False)
    X = cx.X.get()[:, :, pad:pad+sz, pad:pad+sz]
    Y = cx.Y.get()[:, :, pad:pad+sz, pad:pad+sz]
    print "\n"

    # print "*" * 80
    # print "Running same algorithm on host"
    # print "*" * 80
    # X__cpu__, Y__cpu__ = integrate_on_host(
    #     in_array_padded, params=cx.parameters,
    #     maxiter=_DEFAULT_MAXITER, stepsize=_DEFAULT_STEPSIZE)
    # X__cpu__ = X__cpu__[:, :, pad:pad+sz, pad:pad+sz]
    # Y__cpu__ = Y__cpu__[:, :, pad:pad+sz, pad:pad+sz]
    # print "\n"

    print "*" * 80
    print "Running reference implementation"
    print "*" * 80
    X_gt, Y_gt = contextual_CPU.run(in_array, p=_DEFAULT_PARAMETERS,
        axis=1, maxiter=_DEFAULT_MAXITER, keeptime=False, verbose=True)
    print "\n"

    # X_gt, Y_gt = X__cpu__, Y__cpu__

    print "Max diff: X: ", sp.absolute(X - X_gt).max()
    print "Max diff: Y: ", sp.absolute(Y - Y_gt).max()
    print "Avg diff: X: ", sp.absolute(X - X_gt).mean()
    print "Avg diff: Y: ", sp.absolute(Y - Y_gt).mean()

    # ii = random.randint(0, in_array.shape[0])
    # jj = random.randint(0, in_array.shape[1])
    ii, jj = sp.unravel_index(sp.absolute(
        X - X_gt).max((-2, -1)).argmax(), X.shape[:2])

    fig, ax = plt.subplots(2, 3)
    xmin = min(X.min(), X_gt.min())
    xmax = max(X.max(), X_gt.max())
    ymin = min(Y.min(), Y_gt.min())
    ymax = max(Y.max(), Y_gt.max())

    mpp = ax[0, 0].matshow(X[ii, jj], cmap='viridis', vmin=xmin, vmax=xmax)
    ax[0, 0].set_title('X[%i, %i] (cuda)' % (ii, jj))
    plt.colorbar(mpp, ax=ax[0, 0]); plt.draw();

    mpp = ax[0, 1].matshow(X_gt[ii, jj], cmap='viridis', vmin=xmin, vmax=xmax)
    ax[0, 1].set_title('X[%i, %i] (python)' % (ii, jj))
    plt.colorbar(mpp, ax=ax[0, 1]); plt.draw();

    mpp = ax[0, 2].matshow(sp.absolute(X[ii, jj]-X_gt[ii, jj]), cmap='viridis')
    ax[0, 2].set_title('X[%i, %i] (ABSDIFF)' % (ii, jj))
    plt.colorbar(mpp, ax=ax[0, 2]); plt.draw()

    mpp = ax[1, 0].matshow(Y[ii, jj], cmap='viridis', vmin=ymin, vmax=ymax)
    ax[1, 0].set_title('Y[%i, %i] (cuda)' % (ii, jj))
    plt.colorbar(mpp, ax=ax[1, 0]); plt.draw();

    mpp = ax[1, 1].matshow(Y_gt[ii, jj], cmap='viridis', vmin=ymin, vmax=ymax)
    ax[1, 1].set_title('Y[%i, %i] (python)' % (ii, jj))
    plt.colorbar(mpp, ax=ax[1, 1]); plt.draw();

    mpp = ax[1, 2].matshow(sp.absolute(Y[ii, jj]-Y_gt[ii, jj]), cmap='viridis')
    ax[1, 2].set_title('Y[%i, %i] (ABSDIFF)' % (ii, jj))
    plt.colorbar(mpp, ax=ax[1, 2]); plt.draw()

    if not matplotlib.is_interactive():
        plt.show(block=True);

    fig, ax = plt.subplots(1, 4)
    mpp = ax[0].matshow(sp.absolute(X - X_gt).mean((0, 1)), cmap='viridis');
    ax[0].set_title('X - X_gt (avg)')
    plt.colorbar(mpp, ax=ax[0]); plt.draw();
    mpp = ax[1].matshow(sp.absolute(X - X_gt).max((0, 1)), cmap='viridis');
    ax[1].set_title('X - X_gt (max)')
    plt.colorbar(mpp, ax=ax[1]); plt.draw();
    mpp = ax[2].matshow(sp.absolute(Y - Y_gt).mean((0, 1)), cmap='viridis');
    ax[2].set_title('Y - Y_gt (avg)')
    plt.colorbar(mpp, ax=ax[2]); plt.draw();
    mpp = ax[3].matshow(sp.absolute(Y - Y_gt).max((0, 1)), cmap='viridis');
    ax[3].set_title('Y - Y_gt (max)')
    plt.colorbar(mpp, ax=ax[3]); plt.draw();

    if not matplotlib.is_interactive():
        plt.show(block=True);

    del cx
    return X, Y, X_gt, Y_gt

#------------------------------------------------------------------------------#
if __name__ == '__main__':
    accuracy_test()
