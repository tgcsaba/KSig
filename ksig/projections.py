from __future__ import annotations

from abc import ABCMeta, abstractmethod

import warnings

import numpy as np
import cupy as cp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted 

from . import utils
from .utils import ArrayOnCPU, ArrayOnGPU, ArrayOnCPUOrGPU, RandomStateOrSeed

from typing import Optional, Tuple


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class RandomProjection(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Base class for projections.
    
    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    
    def __init__(self, n_components : int = 100, random_state : Optional[RandomStateOrSeed] = None) -> None:
        self.n_components = utils.check_positive_value(n_components, 'n_components')
        self.random_state = random_state
        
    def _check_batch_dim(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        if Z is not None and X.shape[:-1] != Z.shape[:-1]:
            raise ValueError(f'The arrays X and Z have different batch dimensions. ({X.shape[:-1]} != {Z.shape[:-1]})')
    
    def _check_n_features(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, reset : bool = False) -> None:
        n_features = X.shape[-1]
        if Z is not None:
            n_features *= Z.shape[-1]
        if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
            self.n_features_ = n_features
        elif n_features != self.n_features_:
            raise ValueError(f'Received data with a different number of features than at fit time. ({n_features} != {self.n_features_})')
        
    def _validate_data(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, reset : bool = False) -> Tuple[ArrayOnCPUOrGPU, Optional[ArrayOnCPUOrGPU]]:
        self._check_batch_dim(X, Z=Z)
        self._check_n_features(X, Z=Z, reset=reset)
        return X, Z
    
    def _check_n_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        if not hasattr(self, 'n_features_') or self.n_features_ is None:
            warnings.warn('Components could not be checked, because the number of features are as of yet unknown.')
        elif self.n_components > self.n_features_:
            warnings.warn(f'n_features < n_components ({self.n_features_} < {self.n_components}). The dimensionality of the problem will not be reduced.')
        self.n_components_ = self.n_components
    
    @abstractmethod
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        pass
            
    def fit(self, X : ArrayOnCPUOrGPU, y : Optional[ArrayOnCPUOrGPU] = None, Z : Optional[ArrayOnCPUOrGPU] = None) -> RandomProjection:
        X, Z = self._validate_data(X, Z=Z, reset=True)
        self._check_n_components(X, Z=Z)
        self._make_projection_components(X, Z=Z)
        return self
    
    @abstractmethod
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        pass
        
    @abstractmethod
    def _project_outer_prod(self, X : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        pass
        
    def _project(self, X : ArrayOnGPU, Z : Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
        if Z is None:
            return self._project_features(X)
        else:
            return self._project_outer_prod(X, Z)
        
    def transform(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, return_on_gpu : bool = False) -> ArrayOnCPUOrGPU:
        check_is_fitted(self)
        X, Z = self._validate_data(X, Z=Z)
        X = cp.asarray(X)
        Z = cp.asarray(Z) if Z is not None else None
        proj = self._project(X, Z=Z)
        if not return_on_gpu:
            proj = cp.asnumpy(proj)
        return proj
        
    def __call__(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, return_on_gpu : bool = False) -> ArrayOnCPUOrGPU:
        return self.transform(X, Z=Z, return_on_gpu=return_on_gpu)
            
    def fit_transform(self, X : ArrayOnCPUOrGPU, y : Optional[ArrayOnCPUOrGPU] = None, Z : Optional[ArrayOnCPUOrGPU] = None,
                      return_on_gpu : bool = False, **fit_params) -> ArrayOnCPUOrGPU:
        return self.fit(X, y=y, Z=Z, **fit_params).transform(X, Z=Z, return_on_gpu=return_on_gpu)
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class GaussianRandomProjection(RandomProjection):
    """Gaussian Random Projection."""
    
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        random_state = utils.check_random_state(self.random_state)
        self.components_ = random_state.normal(size=(self.n_components_, self.n_features_))
        self.scaling_ = 1. / cp.sqrt(self.n_components_)
        
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        return self.scaling_ * utils.matrix_mult(X.reshape((-1, self.n_features_)), self.components_, transpose_Y=True).reshape(X.shape[:-1] + (self.n_components_,))
        
    def _project_outer_prod(self, X : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        return self._project_features(utils.outer_prod(X, Z))
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SubsamplingRandomProjection(RandomProjection):
    """Subsampling-based Random Projection"""
    
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        random_state = utils.check_random_state(self.random_state)
        self.sampled_idx_ = random_state.choice(self.n_features_, size=self.n_components_, replace=False)
        self.scaling_ = cp.sqrt(self.n_features_ / self.n_components_)
        
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        return self.scaling_ * cp.take(X, self.sampled_idx_, axis=-1)
    
    def _project_outer_prod(self, X : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        X_proj, Z_proj = utils.subsample_outer_prod_comps(X, Z, self.sampled_idx_)
        return self.scaling_ * X_proj * Z_proj
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class VerySparseRandomProjection(RandomProjection):
    """Very Sparse Random Projection."""
    
    def _check_sparsity_mode(self, sparsity : str) -> None:
        if sparsity not in ['sqrt', 'log']:
            raise ValueError(f'Unknown sparsity mode ({sparsity}). Possible values are [\'sqrt\', \'log\'].')
        self.sparsity = sparsity
    
    def __init__(self, n_components : int = 100, sparsity : str = 'log', random_state : Optional[RandomStateOrSeed] = None) -> None:
        self._check_sparsity_mode(sparsity)
        super().__init__(n_components=n_components, random_state=random_state)
    
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        random_state = utils.check_random_state(self.random_state)
        if self.sparsity == 'log':
            prob_nonzero = cp.log(self.n_features_) / self.n_features_
        elif self.sparsity == 'sqrt':
            prob_nonzero = 1. / cp.sqrt(self.n_features_)
        components_full = utils.draw_rademacher_matrix((self.n_components_, self.n_features_), random_state=random_state) \
                          * utils.draw_bernoulli_matrix((self.n_components_, self.n_features_), prob_nonzero, random_state=random_state)
        self.sampled_idx_ = cp.where(cp.any(utils.robust_nonzero(components_full), axis=0))[0]
        self.n_sampled_ = self.sampled_idx_.shape[0]
        self.components_ = cp.squeeze(cp.take(components_full, self.sampled_idx_, axis=1))
        self.scaling_ = cp.sqrt(1. / (prob_nonzero * self.n_components_))
        
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        return self.scaling_ * utils.matrix_mult(
            cp.take(X, self.sampled_idx_, axis=-1).reshape((-1, self.n_sampled_)), self.components_, transpose_Y=True).reshape(X.shape[:-1] + (-1,))
    
    def _project_outer_prod(self, X : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        X_proj, Z_proj = utils.subsample_outer_prod_comps(X, Z, self.sampled_idx_)
        return self.scaling_ * utils.matrix_mult(
            cp.reshape(X_proj * Z_proj, (-1, self.n_sampled_)), self.components_, transpose_Y=True).reshape(X_proj.shape[:-1] + (-1,))
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class CountSketchRandomProjection(RandomProjection):
    """Count Sketch Random Projection."""
    
    def __init__(self, n_components : int = 100, random_state : Optional[RandomStateOrSeed] = None) -> None:
        super().__init__(n_components=n_components, random_state=random_state)
        
    def _check_n_features(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, reset : bool = False) -> None:
        n_features = Z.shape[-1] if Z is not None else X.shape[-1]
        if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
            self.n_features_ = n_features
        elif n_features != self.n_features_:
            raise ValueError(f'Received data with a different number of features than at fit time. ({n_features} != {self.n_features_})')
    
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        random_state = utils.check_random_state(self.random_state)
        self.hash_index_ = random_state.randint(self.n_components_, size=(self.n_features_))
        self.hash_bit_ = utils.draw_rademacher_matrix((self.n_features_,), random_state=random_state) 
        
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        return utils.compute_count_sketch(X, self.hash_index_, self.hash_bit_, n_components=self.n_components_)
    
    def _project_outer_prod(self, X_count_sketch : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        Z_count_sketch = utils.compute_count_sketch(Z, self.hash_index_, self.hash_bit_, n_components=self.n_components_)
        return utils.convolve_count_sketches(X_count_sketch, Z_count_sketch)
    
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------