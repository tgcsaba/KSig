# from abc import ABCMeta, abstractmethod

# import warnings

# import numpy as np
# import cupy as cp

# from sklearn.base import TransformerMixin
# from sklearn.utils.validation import check_is_fitted

# from .. import utils
# from .kernels import Kernel

# class LowRankFeatures(Kernel, TransformerMixin, metaclass=ABCMeta):
#     """Base class for Low-Rank Feature Maps."""
    
#     @abstractmethod
#     def __init__(self, n_components=100, random_state=None):
#         self.n_components = 100
#         self.random_state = random_state
    
#     @abstractmethod
#     def transform(self, X):
#         pass
        
#     def K(self, X, Y=None):
#         X_feat = self.transform(X)
#         if Y is None:
#             return utils.matrix_mult(X_feat, X_feat, transpose_Y=True)
#         else:
#             Y_feat = self.transform(Y)
#             return utils.matrix_mult(X_feat, Y_feat, transpose_Y=True)
    
#     def Kdiag(self, X):
#         X_feat = self.transform(X)
#         return utils.squared_norm(X_feat, axis=-1)

# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        
# class NystroemFeatures(LowRankFeatures):
#     """Nystroem Low-Rank Feature Map."""
    
#     def __init__(self, base_kernel, n_components=100, random_state=None):
#         super().__init__(n_components=n_components, random_state=random_state)
#         self.base_kernel = base_kernel
    
#     def fit(self, X, y=None):
#         # X = self._validate_data(X, ensure_2d=False)
#         # todo: write custom validator to work with cupy and sequence inputs
#         n_samples = X.shape[0]
#         if self.n_components >= n_samples:
#             self.n_components_ = n_samples
#             warnings.warn('n_components >= n_samples, n_components was set to n_samples, which results in inefficient evaluation of the full kernel.')
#         else:
#             self.n_components_ = self.n_components
        
#         random_state = utils.check_random_state(self.random_state)
#         basis_inds = random_state.choice(n_samples, size=self.n_components_, replace=False)
#         basis = X[basis_inds]
#         basis_K = self.base_kernel(basis)
#         S, U = cp.linalg.eigh(basis_K)
#         self.normalization_ = U / utils.robust_sqrt(S)
#         self.components_ = basis
#         self.component_idx_ = inds
#         return self
        
#     def transform(self, X):
#         """Computes an approximate Nystroem feature map using the kernel between the basis points and X."""
#         check_is_fitted(self)
#         # X = self._validate_data(X, ensure_2d=False, reset=False)
#         embedded = self.base_kernel(X, self.components_)
#         return utils.matrix_mult(embedded, self.normalization_, transpose_Y=True)

# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ### TODO: Check the following implementation

# # class RBFFourierFeatures(LowRankFeatures):
# #     """Class for RBF Fourier Features, which approximates the RBF kernel by sampling from its Fourier transform, i.e. the spectral measure."""
    
# #     def __init__(self, sigma=1., lengthscale=1., n_components=100, random_state=None):
# #         super().__init__(n_components=n_components, random_state=random_state)
# #         self.sigma = utils.check_positive_float(sigma)
# #         self.lengthscale = utils.check_positive_float(lengthscale)
        
# #     def fit(self, X):
# #         """Fits the RBFSampler feature map to the provided input array (code is based on the sklearn implementation)."""
# #         # X = self._validate_data(X)
# #         random_state = utils.check_random_state()
# #         n_features = X.shape[1]
        
# #         self.random_weights_ = (cp.sqrt(2 / self.lengthscale**2) * random_state.normal(size=(n_features, self.n_components)))
# #         self.random_offset = random_state.uniform(0, 2 * np.pi, size=self.n_components)
# #         return self
    
# #     def transform(self, X):
# #         """Apply the approximate feature map to X."""
# #         check_is_fitted(self)
# #         # X = self._validate_data(X, reset=False)
# #         projection = utils.matrix_mult(X, self.random_weights_)
# #         projection = self.random_offset_
# #         cp.cos(projection, projection)
# #         projection *= self.sigma * cp.sqrt(2.) / cp.sqrt(self.n_components)
# #         return projection
        
# # # ----------------------------------------------------------------------------------------------------------------------------------------------------------------