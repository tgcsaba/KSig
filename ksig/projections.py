# from abc import ABCMeta, abstractmethod

# import warnings

# import numpy as np
# import cupy as cp

# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.utils.validation import check_is_fitted 

# from . import utils

# class Projection(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
#     """Base class for projections."""
    
#     def init(self, n_components=100, random_state=None):
#         self.n_components = utils.check_positive_int(n_components)
#         self.random_state = random_state
        
#     def _check_batch_dim(self, X, Z=None):
#         if Z is not None and X.shape[:-1] != Z.shape[:-1]:
#             raise ValueError(f'The arrays X and Z have different batch dimensions. ({X.shape[:-1]} != {Z.shape[:-1]})')
    
#     def _check_n_features(self, X, Z=None, reset=False):
#         n_features = X.shape[-1]
#         if Z is not None:
#             n_features *= Z.shape[-1]
#         if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
#             self.n_features_ = n_features
#         elif n_features != self.n_features_:
#             raise ValueError(f'Received a different number of features than at fit time. ({n_features} != {self.n_features_})')
        
#     def _check_n_components(self):
#         if not hasattr(self, 'n_features_') or self.n_features_ is None:
#             warnings.warn('Components could not be checked, because the number of features are as of yet unknown.')
#         elif n_components > self.n_features_:
#             warnings.warn(f'n_features < n_components ({self._n_features_} < {n_components}). The dimensionality of the problem will not be reduced.')
#             self.n_components_ = self.n_features_
#         else:
#             self.n_components_ = n_components
            
#     def _validate_data(self, X, Z=None, reset=False):
#         if Z is not None:
#             self._check_batch_dim(X, Z=Z)
#         self._check_n_features(X, Z=Z, reset=reset)
            
#     @abstractmethod
#     def fit(self, X, y=None, Z=None):
#         random_state = utils.check_random_state(self.random_state)
#         self._validate_data(X, Z=Z, reset=True)
#         self._check_n_components()
#         return self
    
#     @abstractmethod
#     def _project_feature(self, X):
#         pass
        
#     @abstractmethod
#     def _project_outer_prod(self, X, Z):
#         pass
            
#     def transform(self, X, Z=None):
#         check_is_fitted(self)
#         self._validate_data(X, Z=Z)
#         if Z is None:
#             return self._project_feature(X)
#         else:
#             return self._project_outer_prod(X, Z)
        
#     def __call__(self, X, Z=None):
#         return self.transform(X, Z=Z)
            
#     def fit_transform(self, X, y=None, Z=None, **fit_params):
#         return self.fit(X, Z=Z, **fit_params).transform(X, Z=Z)
        
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# class SVDProjection(Projection):
#     """Truncated SVD Projection."""
    
#     def fit(self, X, y=None, Z=None):
#         return super().fit(X, y=y, Z=Z)
        
#     def _project_feature(self, X):
#         U, S, _ = cp.linalg.svd(X, full_matrices=True)
#         return U[..., :, :self.n_components_] * S[..., :self.n_components_]
    
#     def _project_outer_prod(self, X, Z):
#         return self._project_feature(utils.outer_prod(X, Z))
        
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# class GaussianRandomProjection(Projection):
#     """Gaussian Random Projection."""
    
#     def fit(self, X, y=None, Z=None):
#         super().fit(X, y=y, Z=Z)
#         self.components_ = random_state.normal(loc=0.0, scale=1.0 / cp.sqrt(self.n_components_), size=(self.n_components_, self.n_features_))
#         return self
        
#     def _project_feature(self, X):
#         return utils.matrix_mult(X.reshape((-1, self.n_features_)), self.components_, transpose_Y=True).reshape(X.shape[:-1] + (-1,))
        
#     def _project_outer_prod(self, X, Z):
#         return self._project_feature(utils.outer_prod(X, Z))
    
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# class SubsamplingRandomProjection(Projection):
#     """Subsampling-based Random Projection"""
    
#     def fit(self, X, y=None, Z=None):
#         super().fit(X, y=y, Z=Z)
#         self.features_idx_ = random_state.choice(self.n_features_, size=self.n_components_, replace=False)
#         return self
        
#     def _project_feature(self, X):
#         return cp.take(X, self.features_idx_, axis=-1)
    
#     def _project_outer_prod(self, X, Z):
#         idx_X = cp.arange(X.shape[-1]).reshape([-1, 1, 1])
#         idx_Z = cp.arange(Z.shape[-1]).reshape([1, -1, 1])
#         idx_pairs = cp.reshape(cp.concatenate((idx_X + cp.zeros_like(idx_Z), idx_Z + cp.zeros_like(idx_X)), axis=-1), (-1, 2))
#         features_idx_pairs = cp.take(idx_pairs, self.features_idx_, axis=0)
#         X_proj = cp.take(X, features_idx_pairs[:, 0], axis=-1)
#         Z_proj = cp.take(Z, features_idx_pairs[:, 1], axis=-1)
#         return X_proj * Z_proj
        
# ### TODO: scaling (?)
    
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# class VerySparseRandomProjection(Projection):
#     """Very Sparse Random Projection."""
    
#     def _check_sparsity_mode(self, sparsity):
#         if sparsity not in ['sqrt', 'log']:
#             raise ValueError(f'Unknown sparsity mode ({sparsity}). Possible values are \'sqrt\' and \'log\'.')
#         self.sparsity = sparsity
    
#     def __init__(self, n_components=100, sparsity='sqrt', random_state=None):
#         self._check_sparsity_mode(sparsity)
#         super().__init__(self, n_components=n_components, random_state=random_state)
    
#     def fit(self, X, y=None, Z=None):
#         super().fit(X, y=y, Z=Z)
#         if self.sparsity == 'log':
#             prob_nonzero_ = cp.log(self.n_features_) / self.n_features_
#         elif self.sparsity == 'sqrt':
#             prob_nonzero_ = 1. / cp.sqrt(self.n_features_)
#         components = utils.draw_rademacher_matrix((self.n_components_, self.n_features_), random_state=self.random_state) \
#                      * utils.draw_bernoulli_matrix((self.n_components_, self.n_features_), prob_nonzero_, random_state=self.random_state)
#         self.features_idx_ = cp.where(cp.any(utils.robust_nonzero(components_full_), axis=0))
#         self.components_ = cp.sqrt(1. / (prob_nonzero * self.n_components_)) * cp.take(components, self.features_idx_, axis=1)
#         return self
        
#     def _project_feature(self, X):
#         return utils.matrix_mult(cp.take(X, self.features_idx_, axis=-1).reshape(X.shape[:-1] + (-1,)), self.components_, transpose_Y=True).reshape(X.shape[:-1] + (-1,))
    
#     def _project_outer_prod(self, X, Z):
#         idx_X = cp.arange(X.shape[-1]).reshape([-1, 1, 1])
#         idx_Z = cp.arange(Z.shape[-1]).reshape([1, -1, 1])
#         idx_pairs = cp.reshape(cp.concatenate((idx_X + cp.zeros_like(idx_Z), idx_Z + cp.zeros_like(idx_X)), axis=-1), (-1, 2))
#         features_idx_pairs = cp.take(idx_pairs, self.features_idx_, axis=0)
#         X_proj = cp.take(X, features_idx_pairs[:, 0], axis=-1)
#         Z_proj = cp.take(Z, features_idx_pairs[:, 1], axis=-1)
#         return utils.matrix_mult(cp.reshape(X_proj * Z_Proj, (-1, X_proj.shape[-1])), self.components_, transpose_Y=True).reshape(X_proj.shape[:-1] + (-1,))
    
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ### TODO: implement the following    


# # class TensorizedRandomProjection(Projection):
# #     """Subsampling based random projection"""
# #     def __init__(self, n_components=100, )

# # # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# # class CountSketchProjection(Projection):
# #     """Subsampling based random projection"""
# #     def __init__(self, n_components=100, )
    
# # # ----------------------------------------------------------------------------------------------------------------------------------------------------------------