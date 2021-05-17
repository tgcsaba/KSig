from __future__ import annotations

from abc import ABCMeta, abstractmethod

import warnings

import numpy as np
import cupy as cp

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .kernels import Kernel

from .. import utils
from ..utils import ArrayOnCPU, ArrayOnGPU, ArrayOnCPUOrGPU, RandomStateOrSeed

from typing import Optional

class LowRankFeatures(Kernel, TransformerMixin, metaclass=ABCMeta):
    """Base class for Low-Rank Feature Maps.
    
    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    
    def __init__(self, n_components : int = 100, random_state : Optional[RandomStateOrSeed] = None) -> None:
        self.n_components = utils.check_positive_value(n_components, 'n_components')
        self.random_state = random_state
        
    def _check_n_features(self, X : ArrayOnCPUOrGPU, reset : bool = False) -> None:
        n_features = X.shape[-1]
        if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
            self.n_features_ = n_features
        elif n_features != self.n_features_:
            raise ValueError(f'Received data with a different number of features than at fit time. ({n_features} != {self.n_features_})')
        
    def _validate_data(self, X : ArrayOnCPUOrGPU, reset : bool = False) -> ArrayOnCPUOrGPU:
        self._check_n_features(X, reset=reset)
        return X
    
    @abstractmethod
    def _make_feature_components(self, X : ArrayOnCPUOrGPU) -> None:
        pass
        
    def fit(self, X: ArrayOnCPUOrGPU, y : Optional[ArrayOnCPUOrGPU] = None) -> LowRankFeatures:
        X = self._validate_data(X, reset=True)
        self._make_feature_components(X)
        return self
    
    @abstractmethod
    def _compute_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        pass
        
    def transform(self, X : ArrayOnCPUOrGPU, return_on_gpu : bool = False) -> ArrayOnCPUOrGPU:
        check_is_fitted(self)
        X = self._validate_data(X)
        X = cp.asarray(X)
        X_feat = self._compute_features(X)
        if not return_on_gpu:
            X_feat = cp.asnumpy(X_feat)
        return X_feat
        
    def _K(self, X : ArrayOnGPU, Y : Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
        X_feat = self._compute_features(X)
        Y_feat = self._compute_features(Y) if Y is not None else None
        return utils.matrix_mult(X_feat, Y_feat, transpose_Y=True)
    
    def _Kdiag(self, X : ArrayOnGPU) -> ArrayOnGPU:
        X_feat = self._compute_features(X)
        return utils.squared_norm(X_feat, axis=-1)
        
    def __call__(self, X : ArrayOnCPUOrGPU, Y : Optional[ArrayOnCPUOrGPU] = None, diag : bool = False, return_on_gpu : bool = False) -> ArrayOnCPUOrGPU:
        check_is_fitted(self)
        X = self._validate_data(X)
        if Y is not None:
            Y = self._validate_data(Y)
        return super().__call__(X, Y=Y, diag=diag, return_on_gpu=return_on_gpu)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        
class NystroemFeatures(LowRankFeatures):
    """Nystroem Low-Rank Feature Map."""
    
    def __init__(self, static_kernel : Kernel, n_components : int = 100, random_state : Optional[RandomStateOrSeed] = None) -> None:
        super().__init__(n_components=n_components, random_state=random_state)
        self.static_kernel = static_kernel
    
    def _make_feature_components(self, X : ArrayOnGPUOrCPU) -> None:
        X = X.reshape([-1, self.n_features_])
        n_samples = X.shape[0]
        if self.n_components > n_samples:
            warnings.warn('n_samples <= n_components, hence n_components was set to n_samples, which results in inefficient evaluation of the full kernel.')
            self.n_components_ = n_samples
        else:
            self.n_components_ = self.n_components
        random_state = utils.check_random_state(self.random_state)
        basis_inds = random_state.choice(n_samples, size=self.n_components_, replace=False)
        if isinstance(X, ArrayOnCPU):
            basis_inds = cp.asnumpy(basis_inds) # this is unideal but simplest with the current structure
        basis = cp.asarray(X[basis_inds])
        basis_K = self.static_kernel(basis, return_on_gpu=True)
        S, U = cp.linalg.eigh(basis_K)
        nonzero_eigs_mask = utils.robust_nonzero(S)
        self.normalization_ = U[..., :, nonzero_eigs_mask] / utils.robust_sqrt(S[..., nonzero_eigs_mask][..., None, :])
        self.components_ = basis
        self.component_idx_ = basis_inds
        
    def _compute_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        embedded = self.static_kernel(X.reshape([-1, self.n_features_]), self.components_, return_on_gpu=True)
        return utils.matrix_mult(embedded, self.normalization_).reshape(X.shape[:-1] + (-1,))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class RBFFourierFeatures(LowRankFeatures):
    """Random Fourier Features for the RBF kernel."""
    
    def __init__(self, sigma : float = 1., lengthscale : float = 1., n_components : int = 100, random_state : Optional[RandomStateOrSeed] = None) -> None:
        super().__init__(n_components=n_components, random_state=random_state)
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.lengthscale = utils.check_positive_value(lengthscale, 'lengthscale')
        
    def _make_feature_components(self, X : ArrayOnGPUOrCPU) -> None:
        self.n_components_ = self.n_components
        random_state = utils.check_random_state(self.random_state)
        self.random_weights_ = (cp.sqrt(2 / self.lengthscale**2) * random_state.normal(size=(self.n_features_, self.n_components_)))
        self.random_offset_ = random_state.uniform(0, 2 * cp.pi, size=self.n_components_)
    
    def _compute_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        projection = utils.matrix_mult(X.reshape([-1, self.n_features_]), self.random_weights_)
        projection += self.random_offset_[None, :]
        cp.cos(projection, projection)
        projection *= self.sigma * cp.sqrt(2.) / cp.sqrt(self.n_components)
        return projection.reshape(X.shape[:-1] + (-1,))
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------