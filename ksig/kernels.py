"""Signature kernels for sequential data."""

import cupy as cp
import numpy as np
import warnings

from abc import ABCMeta
from typing import Optional

from . import utils
from . import static
from .algorithms import (signature_kern, signature_kern_low_rank,
                         signature_kern_pde, global_align_kern_log,
                         random_warping_series)
from .projections import RandomProjection
from .static.kernels import Kernel, StaticKernel
from .static.features import KernelFeatures, StaticFeatures
from .utils import _EPS, ArrayOnGPU, ArrayOnCPUOrGPU, RandomStateOrSeed


# ------------------------------------------------------------------------------
# Signature kernel base class.
# ------------------------------------------------------------------------------

class SignatureBase(Kernel, metaclass=ABCMeta):
  """Base class for signature kernels.

  Warning: This class should not be used directly, only derived classes.
  """

  def __init__(self, n_levels: int = 5, order: int = 1,
               difference: bool = True, normalize: bool = True,
               n_features: Optional[int] = None):
    """Initializer for `SignatureBase` base class.

    Args:
      n_levels: Number of signature levels.
      order: Signature embedding order.
      difference: Whether to take increments of lifted sequences in the RKHS.
      normalize: Whether to normalize kernel to unit norm in feature space.
      n_features: Optional, the number of features (state-space dim.).
        Provide this when feeding in flattened sequence arrays of ndim=2.

    Raises:
      ValueError: If `n_levels` is not positive.
      ValueError: If `n_features is not None` and it is not positive.
    """
    self.n_levels = utils.check_positive_value(n_levels, 'n_levels')
    self.order = (self.n_levels if order <= 0 or order >= self.n_levels
                  else order)
    self.normalize = normalize
    self.difference = difference
    self.n_features = (utils.check_positive_value(n_features, 'n_features')
                       if n_features is not None else None)

  def _validate_data(self, X: ArrayOnCPUOrGPU, reset: Optional[bool] = False
                     ) -> ArrayOnCPUOrGPU:
    """Validates the input data.

    Args:
      X: A data array on CPU or GPU.
      reset: Whether to reset already fitted parameters.

    Raises:
      ValueError: If the number of features in `X` != `n_features`.
    """

    n_features = (self.n_features_ if hasattr(self, 'n_features_')
                  and self.n_features_ is not None else self.n_features)
    if X.ndim == 2:
      if n_features is None or reset:
        warnings.warn(
          '`X` has` ndim==2. Assuming inputs are univariate time series.',
          'Recommend passing an `n_features` parameter during init when using',
          'flattened arrays of ndim==2.')
        n_features = 1
    elif X.ndim == 3:
      if n_features is None or reset:
        n_features = X.shape[-1]
      elif X.shape[-1] != n_features:
        raise ValueError(
          'Number of features in `X` does not match saved `n_features` param.')
    else:
      raise ValueError(
        'Only input sequence arrays with ndim==2 or ndim==3 are supported.')
    # Reshape data to ndim==3.
    X = X.reshape([X.shape[0], -1, n_features])
    if reset:
      self.n_features_ = n_features
    return X


# ------------------------------------------------------------------------------
# Full-Rank Signature Kernels.
# ------------------------------------------------------------------------------

class SignatureKernel(SignatureBase):
  """Class for full-rank signature kernel."""

  def __init__(self, n_levels: int = 5, order: int = 1,
               difference: bool = True, normalize: bool = True,
               n_features: Optional[int] = None,
               static_kernel: Optional[StaticKernel] = None):
    """Initializes the `SignatureKernel` class.

    Args:
      n_levels: Number of signature levels.
      order: Signature embedding order.
      difference: Whether to take increments of lifted sequences in the RKHS.
      normalize: Whether to normalize kernel to unit norm in feature space.
      n_features: Optional, the number of features (state-space dim.).
        Provide this when feeding in flattened sequence arrays of ndim=2.
      static_kernel: A static kernel from `ksig.static.kernels`.

    Raises:
      ValueError: If `n_levels` is not positive.
      ValueError: If `n_features is not None` and it is not positive.
    """

    super().__init__(n_levels=n_levels, order=order, difference=difference,
                     normalize=normalize, n_features=n_features)
    self.static_kernel = static_kernel or static.kernels.RBFKernel()

  def _compute_embedding(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None,
                      diag: bool = False) -> ArrayOnGPU:
    """Computes the embedding of pairwise kernel evaluations.

    Args:
      X: An array of sequences on GPU.
      Y: Another array of sequences on GPU.
      diag: Whether to compute only the diagonals of K(X, X). Ignores `Y`.

    Returns:
      Pairwise static kernel evaluations required to compute the kernel.
    """
    if diag:
      M = self.static_kernel(X[..., None, :], return_on_gpu=True)
    else:
      if Y is None:
        M = self.static_kernel(X[:, None, :, None, :], X[None, :, :, None, :],
                               return_on_gpu=True)
      else:
        M = self.static_kernel(X[:, None, :, None, :], Y[None, :, :, None, :],
                               return_on_gpu=True)
    return M

  def _compute_kernel(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None,
                      diag: bool = False) -> ArrayOnGPU:
    """Computes the signature kernel matrix.

    Args:
      X: An array of sequences on GPU.
      Y: Another array of sequences on GPU.
      diag: Whether to compute only the diagonals of K(X, X). Ignores `Y`.

    Returns:
      Signature kernel matrix K(X, Y) or the diagonals of K(X, X).
    """
    # M has shape `[n_X, l_X, l_Y]` if `diag` else `[n_X, n_Y, l_X, l_Y]`.
    M = self._compute_embedding(X, Y, diag=diag)
    K = signature_kern(M, self.n_levels, order=self.order,
                       difference=self.difference,
                       return_levels=self.normalize)
    return K

  def _Kdiag(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the diagonal kernel entries.

    Args:
      X: An array of sequences on GPU.

    Returns:
      Diagonal entries of the signature kernel matrix K(X, X).
    """
    if self.normalize:
      return cp.full((X.shape[0],), 1.)
    else:
      return self._compute_kernel(X, diag=True)

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the kernel matrix.

    Args:
      X: An array of sequences on GPU.
      Y: Another array of sequences on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    K = self._compute_kernel(X, Y)
    if self.normalize:
      if Y is None:
        K_Xd = utils.matrix_diag(K)
        if hasattr(self, 'is_log_space') and self.is_log_space:
          K -= 1./2 * (K_Xd[..., :, None] + K_Xd[..., None, :])
        else:
          K_Xd_sqrt = cp.maximum(utils.robust_sqrt(K_Xd), _EPS)
          K /= K_Xd_sqrt[..., :, None] * K_Xd_sqrt[..., None, :]
      else:
        K_Xd = self._compute_kernel(X, diag=True)
        K_Yd = self._compute_kernel(Y, diag=True)
        if hasattr(self, 'is_log_space') and self.is_log_space:
          K -= 1./2 * (K_Xd[..., :, None] + K_Yd[..., None, :])
        else:
          K_Xd_sqrt = cp.maximum(utils.robust_sqrt(K_Xd), _EPS)
          K_Yd_sqrt = cp.maximum(utils.robust_sqrt(K_Yd), _EPS)
          K /= K_Xd_sqrt[..., :, None] * K_Yd_sqrt[..., None, :]
    # If log-space, then exponentiate now.
    if hasattr(self, 'is_log_space') and self.is_log_space:
      K = cp.exp(K)
    # If there is an `n_levels+1` axis in the beginning, do averaging.
    if K.ndim == 3:
      K = cp.mean(K, axis=0)
    return K

# ------------------------------------------------------------------------------

class SignaturePDEKernel(SignatureKernel):
  """Class for full-rank signature-pde kernel."""

  def __init__(self, difference: bool = True, normalize: bool = True,
               n_features: Optional[int] = None,
               static_kernel: Optional[StaticKernel] = None):
    """Initializes the `SignaturePDEKernel` class.

    Args:
      difference: Whether to take increments of lifted sequences in the RKHS.
      normalize: Whether to normalize kernel to unit norm in feature space.
      n_features: Optional, the number of features (state-space dim.).
        Provide this when feeding in flattened sequence arrays of ndim=2.
      static_kernel: A static kernel from `ksig.static.kernels`.

    Raises:
      ValueError: If `n_features is not None` and it is not positive.
    """
    self.difference = difference
    self.normalize = normalize
    self.n_features = (utils.check_positive_value(n_features, 'n_features')
                       if n_features is not None else None)
    self.static_kernel = static_kernel or static.kernels.RBFKernel()

  def _compute_kernel(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None,
                      diag: bool = False) -> ArrayOnGPU:
    """Computes the signature kernel matrix.

    Args:
      X: An array of sequences on GPU.
      Y: Another array of sequences on GPU.
      diag: Whether to compute only the diagonals of K(X, X). Ignores `Y`.

    Returns:
      The SigPDE kernel matrix KSigPDE(X, Y) or the diagonals of KSigPDE(X, X).
    """
    M = self._compute_embedding(X, Y, diag=diag)
    K = signature_kern_pde(M, difference=self.difference)
    return K


# ------------------------------------------------------------------------------
# Signature Features.
# ------------------------------------------------------------------------------

class SignatureFeatures(SignatureBase, KernelFeatures):
  """Class for featurized signature kernels."""

  def __init__(self, n_levels: int = 5, order: int = 1,
               difference: bool = True, normalize: bool = True,
               indep_feat: bool = True, n_features : Optional[int] = None,
               static_features : Optional[StaticFeatures] = None,
               projection: Optional[RandomProjection] = None):
    """Initializes the `SignatureFeatures` class.

    Args:
      n_levels: Number of signature levels.
      order: Signature embedding order.
      difference: Whether to take increments of lifted sequences in the RKHS.
      normalize: Whether to normalize kernel to unit norm in feature space.
      indep_feat: Whether to fit independent features for each tensor product.
      n_features: Optional, the number of features (state-space dim.).
        Provide this when feeding in flattened sequence arrays of ndim=2.
      static_features: A static features object from `ksig.static.features`.
      projection: A projection object from `ksig.projections`.

    Raises:
      ValueError: If `n_levels` is not positive.
      ValueError: If `n_features is not None` and it is not positive.
    """
    super().__init__(n_levels=n_levels, order=order,
                     difference=difference, normalize=normalize,
                     n_features=n_features)
    self.indep_feat = indep_feat
    self.static_features = static_features
    self.projection = projection

  def _make_feature_components(self, X: ArrayOnCPUOrGPU):
    """Initializes internal variables, called by `fit`.

    Args:
      X: An array of sequences on CPU or GPU.
    """
    if self.static_features is not None:
      seq_shape = X.shape[:-1] + (-1,)
      # Merge batch and sequence axes.
      X = X.reshape([-1, X.shape[-1]])
      if self.indep_feat:
        # Create `n_levels` copies of `static_features`.`
        static_feat_hps = self.static_features.get_params()
        static_feat_cls = self.static_features.__class__
        self.static_features_ = [
          static_feat_cls(**static_feat_hps).fit(X)
          for i in range(self.n_levels)]
        U = self.static_features_[0].transform(X, return_on_gpu=True)
      else:
        self.static_features_ = self.static_features.fit(X)
        U = self.static_features_.transform(X, return_on_gpu=True)
      # Reshape back to sequence shape.
      U = U.reshape(seq_shape)
    else:
      U = X
    if self.projection is not None:
      proj_hps = self.projection.get_params()
      proj_cls = self.projection.__class__
      self.projections_ = [proj_cls(**proj_hps).fit(U)]
      V = self.projections_[0](U)
      self.projections_ += [
        proj_cls(**proj_hps).fit(V, Y=U) for i in range(1, self.n_levels)]
    else:
      self.projections_ = None

  def _compute_features(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the feature map, called by `transform`.

    Args:
      X: An array of sequences on GPU.

    Returns:
      Signature features on GPU.
    """
    if self.static_features is not None:
      # Merge batch and sequence axes.
      seq_shape = X.shape[:-1] + (-1,)
      X = X.reshape([-1, X.shape[-1]])
      if isinstance(self.static_features_, list):
        U = [self.static_features_[i].transform(X, return_on_gpu=True).reshape(
          seq_shape) for i in range(self.n_levels)]
      else:
        U = (self.static_features_.transform(X, return_on_gpu=True).reshape(
          seq_shape) if self.static_features is not None else X)
    else:
      U = X
    P = signature_kern_low_rank(
      U, self.n_levels, order=self.order, difference=self.difference,
      return_levels=self.normalize, projections=self.projections_)
    if self.normalize:
      P_norms = [cp.maximum(
        utils.robust_sqrt(utils.squared_norm(p, axis=-1)), _EPS) for p in P]
      P = [p / P_norms[i][..., None] for i, p in enumerate(P)]
      P = cp.concatenate(P, axis=-1) / cp.sqrt(self.n_levels + 1)
    return P


# ------------------------------------------------------------------------------
# Other sequence kernels.
# ------------------------------------------------------------------------------

class GlobalAlignmentKernel(SignatureKernel):
  """Class for full-rank global-alignment kernel."""

  def __init__(self, n_features: Optional[int] = None,
               static_kernel: Optional[StaticKernel] = None):
    """Initializes the `GlobalAlignmentKernel` class.

    This class computes the normalized GA kernel.

    Args:
      n_features: Optional, the number of features (state-space dim.).
        Provide this when feeding in flattened sequence arrays of ndim=2.
      static_kernel: A static kernel from `ksig.static.kernels`.

    Raises:
      ValueError: If `n_features is not None` and it is not positive.
    """
    self.n_features = (utils.check_positive_value(n_features, 'n_features')
                       if n_features is not None else None)
    self.static_kernel = static_kernel or static.kernels.RBFKernel()

  @property
  def normalize(self):
    return True

  @property
  def is_log_space(self):
    return True

  def _compute_kernel(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None,
                      diag: bool = False) -> ArrayOnGPU:
    """Computes the global alignment kernel matrix.

    Args:
      X: An array of sequences on GPU.
      Y: Another array of sequences on GPU.
      diag: Whether to compute only the diagonals of K(X, X). Ignores `Y`.

    Returns:
      The GA kernel matrix GA(X, Y) or the diagonals of GA(X, X).
    """
    M = self._compute_embedding(X, Y, diag=diag)
    K = global_align_kern_log(M)
    return K

# ------------------------------------------------------------------------------

class RandomWarpingSeries(KernelFeatures):
  """Class for normalized Random Warping Series features."""

  def __init__(self, n_components: int = 100, stdev: float = 1.,
               max_warp: int = 32, normalize: bool = True,
               n_features : Optional[int] = None,
               random_state: Optional[RandomStateOrSeed] = None):
    """Initializes the `RandomWarpingSeries` class.

    Args:
      n_components: Number of warping series to use.
      stdev: The standard deviation of the centered Gaussian warping series.
      max_warp: The maximum warping length.
      normalize: Whether to normalize features to unit norm.
      n_features: Optional, the number of features (state-space dim.).
        Provide this when feeding in flattened sequence arrays of ndim=2.
      random_state: A `cupy.random.RandomState`, an `int` seed or `None`.

    Raises:
      ValueError: If `stdev` is not positive.
      ValueError: If `max_warp_len is not positive.
      ValueError: If `n_features is not None` and it is not positive.
    """
    self.n_components = utils.check_positive_value(
      n_components, 'n_components')
    self.stdev = utils.check_positive_value(stdev, 'stdev')
    self.max_warp = utils.check_positive_value(max_warp, 'max_warp')
    self.normalize = normalize
    self.n_features = (utils.check_positive_value(n_features, 'n_features')
                       if n_features is not None else None)
    self.random_state = utils.check_random_state(random_state)

  def _make_feature_components(self, X: ArrayOnCPUOrGPU):
    """Initializes internal variables, called by `fit`.

    Args:
      X: An array of sequences on CPU or GPU.
    """
    X = cp.asarray(X).reshape([-1, X.shape[-1]])
    self.data_means_ = cp.mean(X, axis=0)
    self.data_stdevs_ = cp.std(X, axis=0)
    self.warp_lens_ = self.random_state.randint(
      1, self.max_warp+1, size=[self.n_components])
    self.warp_series_ = self.data_means_[None, :] + self.stdev * (
      self.data_stdevs_[None, :] * self.random_state.normal(
        size=[int(cp.sum(self.warp_lens_)), self.n_features_], dtype=X.dtype))


  def _compute_features(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the feature map, called by `transform`.

    Args:
      X: An array of sequences on GPU.

    Returns:
      Random Warping Series features on GPU.
    """
    # Compute the distance matrices.
    D = utils.squared_euclid_dist(X[:, :, :], self.warp_series_[None, :, :])
    P = random_warping_series(D, self.warp_lens_)
    if self.normalize:
    # Normalize the features.
      P_norm = utils.robust_sqrt(utils.squared_norm(P, axis=-1))
      P /= cp.maximum(P_norm, _EPS)[..., None]
    return P
  
# -----------------------------------------------------------------------------