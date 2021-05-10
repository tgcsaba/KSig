import numpy as np
import cupy as cp

import warnings

from sklearn.base import BaseEstimator

from . import utils
from . import embeddings
from .embeddings.kernels import Kernel
from .algorithms import signature_kern

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SignatureKernel(Kernel):
    """Class for full-rank signature kernel."""
    
    def _check_base_kernel(self, base_kernel=None):
        if not isinstance(base_kernel, Kernel):
            raise ValueError('base_kernel must be an object derived from the class ksig.embeddings.kernels.Kernel.')
        self.base_kernel = base_kernel if base_kernel is not None else embeddings.kernels.LinearKernel()
    
    def _check_order(self, order):
        order = self.n_levels if order==-1 else order
        self.order = utils.check_positive_int(order, 'order')
        if self.order > self.n_levels:
            raise ValueError('The parameter order should be either be -1 or positive int that\'s <= \'n_levels\'')
    
    def __init__(self, n_levels, n_features=None, base_kernel=None, order=1, sigma=1.0, difference=True, normalization=False):
        self.n_levels = utils.check_positive_int(n_levels, 'n_levels')
        self.n_features = n_features        
        self._check_base_kernel(base_kernel)
        self._check_order(order)
        self.sigma = utils.check_positive_float(sigma, 'sigma')
        self.normalization = utils.check_bool(normalization, 'normalization')
        self.difference = utils.check_bool(difference, 'difference')
                
    def _check_n_features(self, X):
        if X.ndim == 2:
            if self.n_features is None:
                warnings.warn('The input array has ndim==2, but n_features was not passed during initialization.',
                              'Assuming inputs are univariate time series. It is recommended to pass an n_features parameter during init.')
                n_features = 1
                return n_features
            else:
                return self.n_features
        elif X.ndim == 3:
            if X.shape[-1] != self.n_features:
                raise ValueError('The last dimension of the input array does not match the n_features parameter passed during initialization.')
            else:
                return self.n_features
        else:
            raise ValueError('Only arrays with ndim==3 are supported.')
    
    def _compute_kernel(self, X, Y=None, diag=False):
        n_features = self._check_n_features(X)
        if Y is not None:
            n_features2 = self._check_n_features(Y)
            if n_features != n_features2:
                raise ValueError('The input arrays X and Y have different dimensionality along the last axis.')
        if diag:
            if Y is not None:
                raise ValueError('Diagonal mode does not support a 2nd input array.')
            M = self.base_kernel(X)
        else:
            M = self.base_kernel(X.reshape((-1, X.shape[-1]))).reshape((X.shape[0], X.shape[1], X.shape[0], X.shape[1])) if Y is None \
                else self.base_kernel(X.reshape((-1, X.shape[-1])), Y.reshape((-1, Y.shape[-1]))).reshape((X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]))
        K = signature_kern(M, self.n_levels, order=self.order, difference=self.difference, return_levels=self.normalization)
        if self.normalization:
            # this is where it gets a bit tricky, since depending on whether this is a symmetric covariance matrix (i.e. Y is None)
            # or a cross-covariance matrix, we may need to separately compute diagonal entries for the normalization
            if Y is None:
                KX_sqrt = utils.robust_sqrt(utils.matrix_diag(K))
                K /= KX_sqrt[..., :, None] * KX_sqrt[..., None, :]
            else:
                KX_sqrt = utils.robust_sqrt(signature_kern(self.base_kernel(X), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                KY_sqrt = utils.robust_sqrt(signature_kern(self.base_kernel(Y), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                K /= KX_sqrt[..., :, None] * KY_sqrt[..., None, :]
            K = cp.mean(K, axis=0)
        return self.sigma**2 * K
    
    def Kdiag(self, X):
        """Wrapper to compute the diagonal signature kernel entries for a given array of sequences X."""
        if self.normalization:
            return cp.full((X.shape[0],), self.sigma**2)
        else:
            return self._compute_kernel(X, diag=True)
    
    def K(self, X, Y=None):
        """Wrapper to compute the signature kernel matrix for a given array of sequences X (and potentially another array Y)."""
        return self._compute_kernel(X, Y)
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------