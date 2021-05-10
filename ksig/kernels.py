import numpy as np
import cupy as cp

from sklearn.base import BaseEstimator

from . import utils
from .embeddings.kernels import Kernel
from .algorithms import signature_kern

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SignatureKernel(Kernel):
    """Class for full-rank signature kernel."""
    
    def __init__(self, static_kernel, n_levels, order=1, sigma=1.0, normalization=True, difference=True):
        self.static_kernel = static_kernel
        self.n_levels = utils.check_positive_int(n_levels, 'n_levels')
        order = n_levels if order==-1 else order
        self.order = utils.check_positive_int(order, 'order')
        if self.order > self.n_levels:
            raise ValueError('The parameter \'order\' should be either be -1 or positive int that\'s <= \'n_levels\'')
        self.sigma = utils.check_positive_float(sigma, 'sigma')
        self.normalization = utils.check_bool(normalization, 'normalization')
        self.difference = utils.check_bool(difference, 'difference')
        
    def check_compatible_inputs(self, X, Y=None):
        """Checks if input array is a proper 3D array"""
        if X.ndim != 3:
            raise ValueError('The input \'X\' to the signature kernel should be a 3D array of shape (n_X, l_X, d).')
        if Y is not None:
            if Y.ndim != 3:
                raise ValueError('The input \'Y\' to the signature kernel should be a 3D array of shape (n_Y, l_Y, d).')
            if X.shape[-1] != Y.shape[-1]:
                raise ValueError(f'The last dimension of the input arrays \'X\' and \'Y\' should be the same, however',
                                 'they were \'d_X\' = {X.shape[-1]} and \'d_Y\' = {Y.shape[-1]}...')
                
    
    def compute_signature_kernel(self, X, Y=None, diag=False):
        self.check_compatible_inputs(X, Y)
        if diag:
            if Y is not None:
                raise ValueError('Diagonal mode does not support a 2nd input array.')
            M = self.static_kernel(X)
        else:
            M = self.static_kernel(X.reshape((-1, X.shape[-1]))).reshape((X.shape[0], X.shape[1], X.shape[0], X.shape[1])) if Y is None \
                else self.static_kernel(X.reshape((-1, X.shape[-1])), Y.reshape((-1, Y.shape[-1]))).reshape((X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]))
                
        K = signature_kern(M, self.n_levels, order=self.order, difference=self.difference, return_levels=self.normalization)
        if self.normalization:
            # this is where it gets a bit tricky, since depending on whether this is a symmetric covariance matrix (i.e. Y is None)
            # or a cross-covariance matrix, we may need to separately compute diagonal entries for the normalization
            if Y is None:
                KX_sqrt = utils.robust_sqrt(utils.matrix_diag(K))
                K /= KX_sqrt[..., :, None] * KX_sqrt[..., None, :]
            else:
                KX_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(X), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                KY_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(Y), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                K /= KX_sqrt[..., :, None] * KY_sqrt[..., None, :]
            K = cp.mean(K, axis=0)
        return self.sigma**2 * K
    
    def Kdiag(self, X):
        """Wrapper to compute the diagonal signature kernel entries for a given array of sequences X."""
        if self.normalization:
            return cp.full((X.shape[0],), self.sigma**2)
        else:
            return self.compute_signature_kernel(X, diag=True)
    
    def K(self, X, Y=None):
        """Wrapper to compute the signature kernel matrix for a given array of sequences X (and potentially another array Y)."""
        return self.compute_signature_kernel(X, Y)
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------