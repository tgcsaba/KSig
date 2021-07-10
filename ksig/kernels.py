from __future__ import annotations

from abc import ABCMeta, abstractmethod

import warnings

import numpy as np
import cupy as cp

from . import utils
from . import static
from .projections import RandomProjection
from .static.kernels import Kernel
from .static.features import LowRankFeatures
from .algorithms import signature_kern, signature_kern_low_rank

from typing import Optional
from .utils import ArrayOnCPU, ArrayOnGPU, ArrayOnCPUOrGPU

from sklearn.base import clone

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SignatureBase(Kernel, metaclass=ABCMeta):
    """Base class for signature kernels.
    
    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    
    @abstractmethod
    def __init__(self, n_levels : int = 5, order : int = 1, sigma : float = 1.0, difference : bool = True, normalization : int = 0, 
                 n_features : Optional[int] = None) -> None:
        self.n_levels = utils.check_positive_value(n_levels, 'n_levels')
        self.order = self.n_levels if order <= 0 or order >= self.n_levels else order
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.normalization = normalization
        self.difference = difference
        self.n_features = utils.check_positive_value(n_features, 'n_features') if n_features is not None else None   
    
    def _validate_data(self, X : ArrayOnCPUOrGPU, reset : Optional[bool] = False) -> ArrayOnCPUOrGPU:
        
        n_features = self.n_features_ if hasattr(self, 'n_features_') and self.n_features_ is not None else self.n_features # default
        if X.ndim == 2:
            if n_features is None or reset:
                warnings.warn('The input array has ndim==2. Assuming inputs are univariate time series.',
                              'It is recommended to pass an n_features parameter during init when using flattened arrays of ndim==2.')
                n_features = 1
        elif X.ndim == 3:
            if n_features is None or reset:
                n_features = X.shape[-1]
            elif X.shape[-1] != n_features:
                raise ValueError('The last dimension of the 3-dim input array does not match the saved n_features parameter ',
                                 '(either during init or during the last time fit was called).')
        else:
            raise ValueError('Only input sequence arrays with ndim==2 or ndim==3 are supported.')
        # reshape data to ndim==3
        X = X.reshape([X.shape[0], -1, n_features])
        if reset:
            self.n_features_ = n_features
        
        return X
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SignatureKernel(SignatureBase):
    """Class for full-rank signature kernel."""
    
    def __init__(self, n_levels : int = 4, order : int = 1, sigma : float = 1.0, difference : bool = True, normalization : int = 0,
                 n_features : Optional[int] = None, static_kernel : Optional[Kernel] = None) -> None:
    
        super().__init__(n_levels=n_levels, order=order, sigma=sigma, difference=difference, normalization=normalization, n_features=n_features)
        self.static_kernel = static_kernel or static.kernels.LinearKernel()
    
    def _compute_kernel(self, X : ArrayOnGPU, Y : Optional[ArrayOnGPU] = None, diag : bool = False) -> ArrayOnGPU:
        
        # check compatible inputs
        X = self._validate_data(X)
        if Y is not None:
            Y = self._validate_data(Y)
            if X.shape[-1] != Y.shape[-1]:
                raise ValueError('The input arrays X and Y have different dimensionality along the last axis.')
        
        if diag:
            if Y is not None:
                raise ValueError('Diagonal mode does not support a 2nd input array.')
            M = self.static_kernel(X)
        else:
            M = self.static_kernel(X.reshape((-1, X.shape[-1])), return_on_gpu=True).reshape((X.shape[0], X.shape[1], X.shape[0], X.shape[1])) if Y is None \
                else self.static_kernel(X.reshape((-1, X.shape[-1])), Y.reshape((-1, Y.shape[-1])), return_on_gpu=True).reshape((X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]))
                
        K = signature_kern(M, self.n_levels, order=self.order, difference=self.difference, return_levels=self.normalization==1)
        
        if self.normalization == 1:
            if Y is None:
                K_X_sqrt = utils.robust_sqrt(utils.matrix_diag(K))
                K /= K_X_sqrt[..., :, None] * K_X_sqrt[..., None, :]
            else:
                K_X_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(X), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                K_Y_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(Y), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                K /= K_X_sqrt[..., :, None] * K_Y_sqrt[..., None, :]
            K = cp.mean(K, axis=0)
        elif self.normalization == 2:
            if Y is None:
                K_X_sqrt = utils.robust_sqrt(utils.matrix_diag(K))
                K /= K_X_sqrt[:, None] * K_X_sqrt[None, :]
            else:
                K_X_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(X), self.n_levels, order=self.order, difference=self.difference))
                K_Y_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(Y), self.n_levels, order=self.order, difference=self.difference))
                K /= K_X_sqrt[:, None] * K_Y_sqrt[None, :]
            
        return self.sigma**2 * K
    
    def _Kdiag(self, X : ArrayOnGPU) -> ArrayOnGPU:
        if self.normalization != 0:
            return cp.full((X.shape[0],), self.sigma**2)
        else:
            return self._compute_kernel(X, diag=True)
    
    def _K(self, X : ArrayOnGPU, Y : Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
        return self._compute_kernel(X, Y)
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class LowRankSignatureKernel(SignatureBase, LowRankFeatures):
    """Class for low-rank signature kernel."""
    def __init__(self, n_levels : int = 4, order : int = 1, sigma : float = 1.0, difference : bool = True, normalization : bool = False, 
                 n_features : Optional[int] = None, static_features : Optional[LowRankFeatures] = None, projection : Optional[RandomProjection] = None) -> None:
    
        super().__init__(n_levels=n_levels, order=order, sigma=sigma, difference=difference, normalization=normalization, n_features=n_features)
        self.static_features = static_features
        self.projection = projection
        
    def _make_feature_components(self, X : ArrayOnCPUOrGPU) -> None:
        if self.static_features is not None:
            self.static_features_ = self.static_features.fit(X)
            U = self.static_features_.transform(X)
        else:
            U = X
        if self.projection is not None:
            self.projections_ = [clone(self.projection).fit(U)]
            V = self.projections_[0](U)
            self.projections_ += [clone(self.projection).fit(V, Z=U) for i in range(1, self.n_levels)]
        else:
            self.projections_ = None
    
    def _compute_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        U = self.static_features_.transform(X, return_on_gpu=True) if self.static_features is not None else X
        P = signature_kern_low_rank(U, self.n_levels, order=self.order, difference=self.difference, return_levels=self.normalization==1, projections=self.projections_)
        if self.normalization == 1:
            P_norms = [utils.robust_sqrt(utils.squared_norm(p, axis=-1)) for p in P]
            P = cp.concatenate([p / P_norms[i][..., None] for i, p in enumerate(P)], axis=-1) / cp.sqrt(self.n_levels + 1)
        elif self.normalization == 2:
            P_norms = utils.robust_sqrt(utils.squared_norm(p, axis=-1))
            P /= P_norms
        
        return self.sigma * P
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------