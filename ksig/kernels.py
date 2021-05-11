import numpy as np
import cupy as cp

import warnings

from . import utils
from . import static
from .static.kernels import Kernel
from .algorithms import signature_kern

from typing import Optional, Union

ArrayOnCPU = np.ndarray
ArrayOnGPU = cp.ndarray
ArrayOnCPUOrGPU = Union[cp.ndarray, np.ndarray]

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SignatureKernel(Kernel):
    """Class for full-rank signature kernel."""
    
    def __init__(self, n_levels : int, order : int = 1, sigma : float = 1.0, difference : bool = True, normalization : bool = False,
                 n_features : Optional[int] = None, static_kernel : Optional[Kernel] = None) -> None:
    
        self.n_levels = utils.check_positive_value(n_levels, 'n_levels')
        self.order = self.n_levels if order <= 0 or order >= self.n_levels else order
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.normalization = normalization
        self.difference = difference
        self.n_features = utils.check_positive_value(n_features, 'n_features') if n_features is not None else None        
        self.static_kernel = static_kernel or static.kernels.LinearKernel()
                
    def _validate_n_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
    
        n_features = self.n_features # default
        if X.ndim == 2 and n_features is None:
                warnings.warn('The input array has ndim==2, but n_features was not passed during initialization.',
                              'Assuming inputs are univariate time series. It is recommended to pass an n_features parameter during init.')
                n_features = 1
        elif X.ndim == 3:
            if n_features is None:
                n_features = X.shape[-1]
            elif X.shape[-1] != n_features:
                raise ValueError('The last dimension of the 3-dim input array does not match the n_features parameter passed during init.')
        else:
            raise ValueError('Only input sequence arrays with ndim==2 or ndim==3 are supported.')
        # reshape data to ndim==3
        X = X.reshape([X.shape[0], -1, n_features])
        
        return X
    
    def _compute_kernel(self, X : ArrayOnGPU, Y : Optional[ArrayOnGPU] = None, diag : bool = False) -> ArrayOnGPU:
        
        # check compatible inputs
        X = self._validate_n_features(X)
        if Y is not None:
            Y = self._validate_n_features(Y)
            if X.shape[-1] != Y.shape[-1]:
                raise ValueError('The input arrays X and Y have different dimensionality along the last axis.')
        
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
                K_X_sqrt = utils.robust_sqrt(utils.matrix_diag(K))
                K /= K_X_sqrt[..., :, None] * K_X_sqrt[..., None, :]
            else:
                K_X_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(X), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                K_Y_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(Y), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                K /= K_X_sqrt[..., :, None] * K_Y_sqrt[..., None, :]
            K = cp.mean(K, axis=0)
            
        return self.sigma**2 * K
    
    def _Kdiag(self, X : ArrayOnGPU) -> ArrayOnGPU:
        if self.normalization:
            return cp.full((X.shape[0],), self.sigma**2)
        else:
            return self._compute_kernel(X, diag=True)
    
    def _K(self, X : ArrayOnGPU, Y : Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
        return self._compute_kernel(X, Y)
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
