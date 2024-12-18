"""Static (state-space) feature maps implementing a `transform` method."""

import cupy as cp
import numpy as np
import warnings

from abc import ABCMeta, abstractmethod
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Optional

from .. import utils
from ..utils import ArrayOnCPU, ArrayOnGPU, ArrayOnCPUOrGPU, RandomStateOrSeed
from .kernels import Kernel


# ------------------------------------------------------------------------------
# Kernels with explicitly computable feature maps.
# ------------------------------------------------------------------------------

class KernelFeatures(Kernel, TransformerMixin, metaclass=ABCMeta):
  """Base class for featurized kernels.

  Deriving classes should implement the following methods:
    _make_feature_components: Initializes internal variables, called by `fit`.
    _compute_features: Computes the feature map, called by `transform`.

  Warning: This class should not be used directly, only derived classes.
  """

  def __init__(self, n_components: int = 100,
               random_state: Optional[RandomStateOrSeed] = None):
    """Initializer for the `KernelFeatures` base class.

    Args:
      n_components: Number of feature components to use.
      random_state: A `cupy.random.RandomState`, an `int` seed or `None`.
    """
    self.n_components = utils.check_positive_value(
      n_components, 'n_components')
    self.random_state = utils.check_random_state(random_state)

  def _validate_format(self, X: ArrayOnCPUOrGPU, reset: bool = False):
    """Validates the `n_features` dimension in the data, i.e. the last axis.

    Args:
      X: A data array on CPU or GPU.
      reset: Whether to reset the `n_features` parameter.

    Raises:
      ValueError: If `not reset` and the `n_features` dimension doesn't match.
    """
    n_features = X.shape[-1]
    if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
      self.n_features_ = n_features
    elif n_features != self.n_features_:
      raise ValueError(
        'Data `X` has a different number of features than during fit.',
        f'({n_features} != {self.n_features_})')

  def _validate_data(self, X: ArrayOnCPUOrGPU, reset: bool = False
                     ) -> ArrayOnCPUOrGPU:
    """Validates the input data array `X`.

    This method returns `X` as derived classes might make changes to it.

    Args:
      X: A data array on CPU or GPU.
      reset: Whether to reset the `n_features` parameter.
    """
    self._validate_format(X, reset=reset)
    return X

  @abstractmethod
  def _make_feature_components(self, X: ArrayOnCPUOrGPU):
    """Initializes internal variables, called by `fit`.

    Args:
      X: A data array on CPU or GPU.
    """
    pass

  def fit(self, X: ArrayOnCPUOrGPU) -> 'KernelFeatures':
    """Validates the input data, and initializes internal variables.

    Args:
      X: A data array on CPU or GPU.

    Returns:
      A fitted `KernelFeatures` object.
    """
    X = self._validate_data(X, reset=True)
    self._make_feature_components(X)
    return self

  @abstractmethod
  def _compute_features(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the feature map, called by `transform`.

    Args:
      X: A data array on GPU.

    Returns:
      The feature map as A data array on GPU.
    """
    pass

  def transform(self, X: ArrayOnCPUOrGPU,
                return_on_gpu: bool = False) -> ArrayOnCPUOrGPU:
    """Validates the input data, and computes the feature map.

    This method checks whether the object is fitted, validates and ensures
    the data is in GPU memory, and computes the feature map. Depending on
    `return_on_gpu`, it may move the result back to CPU memory.

    Args:
      X: A data array on CPU or GPU.
      return_on_gpu: Whether to return result on GPU.

    Returns:
      A feature set on CPU or GPU.
    """
    check_is_fitted(self)
    # Validate data and move it to GPU.
    X = cp.asarray(self._validate_data(X))
    X_feat = self._compute_features(X)
    if not return_on_gpu:
      X_feat = cp.asnumpy(X_feat)
    return X_feat

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the kernel matrix via the explicit feature map.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    X_feat = self._compute_features(X)
    Y_feat = self._compute_features(Y) if Y is not None else None
    return utils.matrix_mult(X_feat, Y_feat, transpose_Y=True)

  def _Kdiag(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the diagonal kernel entries via the explicit feature map.

    Args:
      X: A data array on GPU.

    Returns:
      The diagonal entries of the kernel matrix of `X`.
    """
    X_feat = self._compute_features(X)
    return utils.squared_norm(X_feat, axis=-1)

  def __call__(self, X: ArrayOnCPUOrGPU, Y: Optional[ArrayOnCPUOrGPU] = None,
               diag: bool = False, return_on_gpu: bool = False
               ) -> ArrayOnCPUOrGPU:
    """Implementes the basic call method of a feature map.

    It takes as input one or two arrays, either on CPU (as `numpy`) or on
    GPU (as `cupy`), validates these arrays, and computes the corresponding
    kernel matrix via the Euclidean inner product of the feature maps.


    Args:
      X: A data on CPU or GPU.
      Y: An optional data array on CPU or GPU (if not given uses `X`).
      diag: Whether to compute only the diagonals. Ignores `Y` in this case.
      return_on_gpu: Whether to return the result on GPU.

    Returns:
      A kernel matrix or its diagonal entries on CPU or GPU.
    """
    check_is_fitted(self)
    return super().__call__(X, Y=Y, diag=diag, return_on_gpu=return_on_gpu)


class StaticFeatures(KernelFeatures):
  """Base class for static features.

  Deriving classes should implement the following methods:
    _make_feature_components: Initializes internal variables, called by `fit`.
    _compute_features: Computes the feature map, called by `transform`.

  Warning: This class should not be used directly, only derived classes.
  """
  def _validate_data(self, X: ArrayOnCPUOrGPU, reset: bool = False
                     ) -> ArrayOnCPUOrGPU:
    """Validates the input data.

    Args:
      X: A data array on CPU or GPU.
      reset: Whether to reset already fitted parameters.
    """
    # Flatten the time axis.
    if X.ndim > 2:
      X = X.reshape(X.shape[:-2] + (-1,))
    return super()._validate_data(X, reset=reset)


# ------------------------------------------------------------------------------

class NystroemFeatures(StaticFeatures):
  """Class for the Nystroem feature map.

  Reference:
    * Williams, C.K.I. and Seeger, M.
      "Using the Nystroem method to speed up kernel machines",
      Advances in neural information processing systems 2001
  """

  def __init__(self, base_kernel: Kernel, n_components: int = 100,
               random_state: Optional[RandomStateOrSeed] = None):
    """Initializer for the `NystroemFeatures` class.

    Args:
      base: The base kernel object to use for computing Nystroem features.
      n_components: The number of feature components to use in the embedding.
        At most the number of samples of the dataset its fitted on.
      random_state: A `cupy.random.RandomState`, an `int` seed or `None`.
    """
    super().__init__(n_components=n_components, random_state=random_state)
    self.base_kernel = base_kernel

  def _make_feature_components(self, X: ArrayOnCPUOrGPU):
    """Initializes internal variables, called by `fit`.

    This method subsamples `n_components` number of data points, builds and
    decomposes the corresponding symmetric kernel matrix, and saves the
    resulting normalized matrix factor for building the Nystroem feature map.

    Args:
      X: A data array on CPU or GPU.
    """
    X = X.reshape([-1, self.n_features_])
    n_samples = X.shape[0]
    # Number of components can't be larger than the number of input samples.
    if self.n_components >= n_samples:
      warnings.warn('`n_samples <= n_components`, hence `n_components` was ' +
                    'set to `n_samples`, which results in inefficient ' +
                    'evaluation of the full kernel.')
      self.n_components_ = n_samples
    else:
      self.n_components_ = self.n_components
    # Subsample the data.
    basis_idx = self.random_state.choice(n_samples, size=self.n_components_,
                                         replace=False)
    if isinstance(X, ArrayOnCPU):
      basis_idx = cp.asnumpy(basis_idx)  # Doing this seems most efficient.
    basis = cp.asarray(X[basis_idx])
    # Build the kernel matrix.
    basis_K = self.base_kernel(basis, return_on_gpu=True)
    # Now decompose it.
    S, U = cp.linalg.eigh(basis_K)
    # Mask zero eigenvalues.
    nonzero_eigs_mask = utils.robust_nonzero(S)
    # Compute and save K^(-1/2).
    self.normalization_ = (U[..., nonzero_eigs_mask] /
                           utils.robust_sqrt(S[..., None, nonzero_eigs_mask]))
    self.components_ = basis
    self.component_idx_ = basis_idx

  def _compute_features(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes Nystroem features, called by `transform`.

    Builds the cross kernel matrix between the supplied datapoints and the
    basis components, then projects it onto the projection matrix.

    Args:
      X: A data array on CPU or GPU.

    Returns:
      The Nystroem feature map as a data array on GPU.
    """
    # Flatten the batch.
    embedded = self.base_kernel(X.reshape([-1, self.n_features_]),
                                self.components_, return_on_gpu=True)
    features = utils.matrix_mult(embedded, self.normalization_)
    # Reshape to original batch shape.
    features = features.reshape(X.shape[:-1] + (-1,))
    return features


# ------------------------------------------------------------------------------

class RandomFourierFeatures(StaticFeatures):
  """Class for Random Fourier Features for the Gaussian (RBF) kernel.

  Reference:
    * Rahimi, A. and Recht, B.
      "Random Features for Large-Scale Kernel Machines",
      Advances in neural information processing systems 2008
  """

  def __init__(self, bandwidth: float = 1., n_components: int = 100,
               random_state: Optional[RandomStateOrSeed] = None):
    """Initializer for the `RandomFourierFeatures` class.

    Args:
      bandwidth: Bandwidth hyperparameter dividing the input data.
      n_components: The number of feature components to use in the embedding.
      random_state: A `cupy.random.RandomState`, an `int` seed or `None`.
    """
    super().__init__(n_components=n_components, random_state=random_state)
    self.bandwidth = utils.check_positive_value(bandwidth, 'bandwidth')

  def _make_feature_components(self, X: ArrayOnCPUOrGPU):
    """Initializes internal variables, called by `fit`.

    This method samples a Gaussian matrix with shape
    `[n_features, n_components]` and standard deviation `1. / bandwidth`.

    Args:
      X: A data array on CPU or GPU.
    """
    self.n_components_ = self.n_components
    self.random_weights_ = 1. / self.bandwidth * self.random_state.normal(
      size=[self.n_features_, self.n_components_])

  def _compute_features(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes random Fourier features, called by `transform`.

    This method projects the data onto the stored random weights, pushes it
    through cosine and a sine activations, and rescales it.

    Args:
      X: A data array on CPU or GPU.

    Returns:
      The random Fourier feature map as A data array on GPU.
    """
    projection = utils.matrix_mult(
      X.reshape([-1, self.n_features_]), self.random_weights_)
    features = cp.concatenate(
      (cp.sin(projection), cp.cos(projection)), axis=-1)
    features /= cp.sqrt(self.n_components)
    return features.reshape(X.shape[:-1] + (-1,))

# -----------------------------------------------------------------------------


class RandomFourierFeatures1D(StaticFeatures):
  """Class for Random Fourier Features for the Gaussian (RBF) kernel.

  Reference:
    Yarin Gal and Richard Turner. Improving the gaussian process sparse spectrum
    approximation by representing uncertainty in frequency inputs. In Interna-
    tional Conference on Machine Learning, pages 655–664. PMLR, 2015.
  """

  def __init__(self, bandwidth: float = 1., n_components: int = 100,
               random_state: Optional[RandomStateOrSeed] = None):
    """Initializer for the `RandomFourierFeatures` class.

    Args:
      bandwidth: Bandwidth hyperparameter dividing the input data.
      n_components: The number of feature components to use in the embedding.
      random_state: A `cupy.random.RandomState`, an `int` seed or `None`.
    """
    super().__init__(n_components=n_components, random_state=random_state)
    self.bandwidth = utils.check_positive_value(bandwidth, 'bandwidth')

  def _make_feature_components(self, X: ArrayOnCPUOrGPU):
    """Initializes internal variables, called by `fit`.

    This method samples a Gaussian matrix with shape
    `[n_features, n_components]` and standard deviation `1. / bandwidth`.

    Args:
      X: A data array on CPU or GPU.
    """
    self.n_components_ = self.n_components
    self.random_weights_ = 1. / self.bandwidth * self.random_state.normal(
      size=[self.n_features_, self.n_components_])
    self.random_shift_ = 2*cp.pi * self.random_state.uniform(
      size=[1, self.n_components_]
    )

  def _compute_features(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes random Fourier features, called by `transform`.

    This method projects the data onto the stored random weights, pushes it
    through cosine and a sine activations, and rescales it.

    Args:
      X: A data array on CPU or GPU.

    Returns:
      The random Fourier feature map as A data array on GPU.
    """
    projection = utils.matrix_mult(
      X.reshape([-1, self.n_features_]), self.random_weights_)
    features = cp.cos(projection + self.random_shift_)
    features *= cp.sqrt(2 / self.n_components)
    return features.reshape(X.shape[:-1] + (-1,))

# -----------------------------------------------------------------------------