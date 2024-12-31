"Low-dimensional (random) projection operations for tensor contraction."

import cupy as cp
import numpy as np
import warnings

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Optional, Tuple

from . import utils
from .utils import ArrayOnGPU, ArrayOnCPUOrGPU, RandomStateOrSeed


# ------------------------------------------------------------------------------
# Random projection base class.
# ------------------------------------------------------------------------------

class RandomProjection(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
  """Base class for random projections.

  Random projections are used for computing low-rank signature kernels, where
  a crucial step is computing the outer product of multiple feature arrays.

  Deriving classes should implement the following methods:
    _make_projection_components: Initializes internal vars, called by `fit`.
    _project: Projects an input array, called by `transform`.
    _project_outer_prod: Projects the outer product of a pair of input arrays,
      called by `transform`.

  Warning: This class should not be used directly, only derived classes.
  """

  def __init__(self, n_components : int = 100,
               random_state: Optional[RandomStateOrSeed] = None):
    """Initializer for `RandomProjection` base class.

    Args:
      n_components: Number of projection components to use.
      random_state: A `cupy.random.RandomState`, an `int` seed or `None`.
    """
    self.n_components = utils.check_positive_value(
      n_components, 'n_components')
    self.random_state = utils.check_random_state(random_state)

  def _validate_format(self, X: ArrayOnCPUOrGPU,
                       Y: Optional[ArrayOnCPUOrGPU] = None,
                       reset: bool = False):
    """Checks the `n_features` dimension in the data, i.e. the last axis.

    Note: If `Y` is given then `n_features` is equal to
      (number of features in `X`) * (number of features in `Y`).

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
      reset: Whether to reset the `n_features` parameter.

    Raises:
      ValueError: If `not reset` and the `n_features` dimension doesn't match.
    """
    n_features = X.shape[-1]
    if Y is not None:
      n_features *= Y.shape[-1]
    if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
      self.n_features_ = n_features
    elif n_features != self.n_features_:
      raise ValueError(
        'Data has a different number of features than during fit.',
        f'({n_features} != {self.n_features_})')

  def _validate_data(self, X: ArrayOnCPUOrGPU,
                     Y: Optional[ArrayOnCPUOrGPU] = None,
                     reset: bool = False
                     ) -> Tuple[ArrayOnCPUOrGPU, Optional[ArrayOnCPUOrGPU]]:
    """Validates the input data, i.e. dim. check for matching `n_features`.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
      reset: Whether to reset the `n_features` parameter.
    """
    self._validate_format(X, Y=Y, reset=reset)

  @abstractmethod
  def _make_projection_components(self, X: ArrayOnCPUOrGPU,
                                  Y: Optional[ArrayOnCPUOrGPU] = None):
    """Initializes internal variables, called by `fit`.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
    """
    pass

  def fit(self, X: ArrayOnCPUOrGPU, Y: Optional[ArrayOnCPUOrGPU] = None
          ) -> 'RandomProjection':
    """Validates the input data, and initializes internal variables.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.

    Returns:
      A fitted `RandomProjection` object.
    """
    self._validate_data(X, Y=Y, reset=True)
    self._make_projection_components(X, Y=Y)
    return self

  @abstractmethod
  def _project(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the input array, called by `transform`.

    Args:
      X: A data array on GPU.

    Returns:
      The projected array on GPU.
    """
    pass

  @abstractmethod
  def _project_outer_prod(self, X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the outer product of input arrays, called by `transform`.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    pass

  def transform(self, X: ArrayOnCPUOrGPU, Y: Optional[ArrayOnCPUOrGPU] = None,
                return_on_gpu: bool = False) -> ArrayOnCPUOrGPU:
    """Validates the input data, and computes its projection.

    If `Y` is given, then the projection is applied to the outer product
    of arrays `X` and `Y` along the last axis, maybe more efficiently than
    computing the outer product first, then the projection.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
      return_on_gpu: Whether to return result on GPU.

    Returns:
      Projected data on CPU or GPU.
    """
    check_is_fitted(self)
    self._validate_data(X, Y=Y)
    # Move data to GPU if not there already.
    X = cp.asarray(X)
    Y = cp.asarray(Y) if Y is not None else None
    if Y is not None:
      proj = self._project_outer_prod(X, Y)
    else:
      proj = self._project(X)
    if not return_on_gpu:
      proj = cp.asnumpy(proj)
    return proj

  def __call__(self, X: ArrayOnCPUOrGPU, Y: Optional[ArrayOnCPUOrGPU] = None,
               return_on_gpu : bool = False) -> ArrayOnCPUOrGPU:
    """Implementes the basic call method of a random projection object.

    If `Y` is given, then the projection is applied to the outer product
    of arrays `X` and `Y` along the last axis, maybe more efficiently than
    computing the outer product first, then the projection.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
      return_on_gpu: Whether to return result on GPU.

    Returns:
      Projected data on CPU or GPU.
    """
    return self.transform(X, Y=Y, return_on_gpu=return_on_gpu)


# ------------------------------------------------------------------------------
# Standalone random projections - can be used outside of signature algorithms.
# ------------------------------------------------------------------------------

class GaussianRandomProjection(RandomProjection):
  """Class for vanilla Gaussian random projections.

  Reference:
    * Bingham, E. and Mannila, H.
      "Random projection in dimensionality reduction:
        applications to image and text data",
      Proceedings of the seventh ACM SIGKDD international conference on
        Knowledge discovery and data mining 2001
    * Dasgupta, S.
      "Experiments with random projection."
      Proceedings of the Sixteenth conference on Uncertainty in artificial
        intelligence 2000
  """

  def _make_projection_components(self, X: ArrayOnCPUOrGPU,
                                  Y: Optional[ArrayOnCPUOrGPU] = None):
    """Initializes internal variables, called by `fit`.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
    """
    self.components_ = self.random_state.normal(
      size=[self.n_components, self.n_features_])
    self.scaling_ = 1. / cp.sqrt(self.n_components)

  def _project(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the input array, called by `transform`.

    Args:
      X: A data array on GPU.

    Returns:
      The projected array on GPU.
    """
    X_reshaped = X.reshape([-1, self.n_features_])
    proj = self.scaling_ * utils.matrix_mult(
      X_reshaped, self.components_, transpose_Y=True)
    return proj.reshape(X.shape[:-1] + (self.n_components,))

  def _project_outer_prod(self, X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the outer product of input arrays, called by `transform`.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    return self._project(utils.outer_prod(X, Y))


# ------------------------------------------------------------------------------

class SubsamplingProjection(RandomProjection):
  """Class for subsampling as a projection."""

  def _make_projection_components(self, X: ArrayOnCPUOrGPU,
                                  Y: Optional[ArrayOnCPUOrGPU] = None):
    """Initializes internal variables, called by `fit`.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
    """
    self.sampled_idx_ = self.random_state.choice(
      self.n_features_, size=self.n_components, replace=False)
    self.scaling_ = cp.sqrt(self.n_features_ / self.n_components)

  def _project(self, X : ArrayOnGPU) -> ArrayOnGPU:
    """Projects the input array, called by `transform`.

    Args:
      X: A data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    return self.scaling_ * cp.take(X, self.sampled_idx_, axis=-1)

  def _project_outer_prod(self, X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the outer product of input arrays, called by `transform`.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    XY_proj = utils.subsample_outer_prod(X, Y, self.sampled_idx_)
    return self.scaling_ * XY_proj


# ------------------------------------------------------------------------------

class VerySparseRandomProjection(RandomProjection):
  """Class for very sparse random projections.
  Reference:
    * Li, P., Hastie, T.J. and Church, K.W.
      "Very sparse random projections"
      Proceedings of the 12th ACM SIGKDD international conference on Knowledge
        discovery and data mining 2006
  """

  def _check_sparsity_mode(self, sparsity : str) -> None:
    """Checks whether the `sparsity` argument falls within accepted values.

    Args:
      sparsity (str): Rate of sparsity. Possible values are 'sqrt' and 'log'.

    Raises:
      ValueError: If `sparsity` is not among the accepted values.
    """
    if sparsity not in ['sqrt', 'log']:
      raise ValueError(f'Unknown sparsity mode ({sparsity}).',
                       'Possible values are [\'sqrt\', \'log\'].')

  def __init__(self, n_components: int = 100, sparsity: str = 'log',
               random_state : Optional[RandomStateOrSeed] = None):
    """Initializer for `VerySparseRandomProjection` class.

    Args:
      n_components: Number of projection components to use.
      sparsity: Rate of sparsity. Possible values are 'sqrt' and 'log'.
      random_state: A `cupy.random.RandomState`, an `int` seed or `None`.
    """
    self._check_sparsity_mode(sparsity)
    self.sparsity = sparsity
    super().__init__(n_components=n_components, random_state=random_state)

  def _make_projection_components(self, X: ArrayOnCPUOrGPU,
                                  Y: Optional[ArrayOnCPUOrGPU] = None):
    """Initializes internal variables, called by `fit`.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.
    """
    if self.sparsity == 'log':  # Very sparse.
      prob_nonzero = cp.log(self.n_features_) / self.n_features_
    elif self.sparsity == 'sqrt':  # Less sparse.
      prob_nonzero = 1. / cp.sqrt(self.n_features_)
    components_full = utils.draw_bernoulli_matrix(  # Draw sparse Bernoulli matrix.
        [self.n_components, self.n_features_], prob=prob_nonzero,
        random_state=self.random_state)
    components_full[0, 0] = 1  # Force at least one nonzero component.
    components_full = utils.draw_rademacher_matrix(  # Random flips.
      [self.n_components, self.n_features_], random_state=self.random_state)
    self.sampled_idx_ = cp.where(  # Subsample nonzero columns.
      cp.any(utils.robust_nonzero(components_full), axis=0))[0]
    self.n_sampled_ = self.sampled_idx_.shape[0]
    self.components_ = cp.take(components_full, self.sampled_idx_, axis=1)
    self.scaling_ = cp.sqrt(1. / (prob_nonzero * self.n_components))

  def _project(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the input array, called by `transform`.

    Args:
      X: A data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    X_sampled = cp.take(
      X, self.sampled_idx_, axis=-1).reshape([-1, self.n_sampled_])
    X_proj = self.scaling_ * utils.matrix_mult(
      X_sampled, self.components_, transpose_Y=True)
    return X_proj.reshape(X.shape[:-1] + (-1,))

  def _project_outer_prod(self, X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the outer product of input arrays, called by `transform`.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    XY_proj = utils.subsample_outer_prod(X, Y, self.sampled_idx_)
    return self.scaling_ * utils.matrix_mult(
      cp.reshape(XY_proj, [-1, self.n_sampled_]), self.components_,
      transpose_Y=True).reshape(XY_proj.shape[:-1] + (-1,))


# ------------------------------------------------------------------------------

class TensorSketch(RandomProjection):
  """Class for computing count sketch random projections.

  Reference:
    * Charikar, M, Chen, K. and Farach-Colton, M.
      "Finding frequent items in data streams"
      International Colloquium on Automata, Languages, and Programming 2002
    * Pham, M. and Pagh, R.
      "Fast and scalable polynomial kernels via explicit feature maps"
      Proceedings of the 19th ACM SIGKDD international conference on Knowledge
        discovery and data mining 2013
  """
  def _validate_format(self, X: ArrayOnCPUOrGPU,
                        Y: Optional[ArrayOnCPUOrGPU] = None,
                        reset : bool = False):
    """Checks the `n_features` dimension in the data, i.e. the last axis.

    Note: If `Y` is given then `n_features` is equal to the number of features
      in `Y`, otherwise to the number of features in `X`.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
      reset: Whether to reset the `n_features` parameter.

    Raises:
      ValueError: If `not reset` and the `n_features` dimension doesn't match.
    """
    n_features = Y.shape[-1] if Y is not None else X.shape[-1]
    if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
      self.n_features_ = n_features
    elif n_features != self.n_features_:
      raise ValueError(
        'Received data with a different number of features than at fit time.',
        f'({n_features} != {self.n_features_})')

  def _make_projection_components(self, X: ArrayOnCPUOrGPU,
                                  Y: Optional[ArrayOnCPUOrGPU] = None):
    """Initializes internal variables, called by `fit`.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.
    """
    self.hash_idx_ = self.random_state.randint(
      self.n_components, size=[self.n_features_])
    self.hash_bit_ = utils.draw_rademacher_matrix(
      [self.n_features_,], random_state=self.random_state)

  def _project(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the input array, called by `transform`.

    Args:
      X: A data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    return utils.compute_count_sketch(
      X, self.hash_idx_, self.hash_bit_, n_components=self.n_components)

  def _project_outer_prod(self, X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the outer product of input arrays, called by `transform`.

    Note: assumes that `X` is already sketched, only sketches `Y` and takes
    their elementwise product, i.e. it should be applied recursively.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    Y_sketch = utils.compute_count_sketch(
      Y, self.hash_idx_, self.hash_bit_, n_components=self.n_components)
    return utils.convolve_fft(X, Y_sketch)

CountSketch = TensorSketch
CountSketchRandomProjection = TensorSketch


# ------------------------------------------------------------------------------
# Non-standalone random projections - can only be used in signature algorithms.
# ------------------------------------------------------------------------------

class TensorizedRandomProjection(RandomProjection):
  """Class for tensorized random projections.

  Warning: This function is not meant to be used as a standalone RP, it is
    meant to be called recursively for each consecutive multiplication step in
    the construction of a tensor. For example, if a tensor is given as
      X = x_1 \otimes x_2 \otimes \dots \otimes x_m,
    then the projection of the tensor X can be recursively achieved by
      X^\prime = TRP_m(...(TRP_2(TRP_1(x_1), x_2), ...), x_m),
    where each TRP function is an instance of this class.

  Reference:
    * Sun, Y., Guo, Y., Tropp, J.A. and Udell, M.
      "Tensor random projection for low memory dimension reduction"
      arXiv preprint arXiv:2105.00105 (2021).
    * Rakhshan, B. and Rabusseau, G.
      "Tensorized random projections"
      International Conference on Artificial Intelligence and Statistics 2020
  """
  def __init__(self, n_components: int = 100, rank: int = 1,
               random_state : Optional[RandomStateOrSeed] = None):
    """Initializer for `TensorizedRandomProjection` class.

    Args:
      n_components: Number of projection components to use.
      rank: Rank of projection tensors.
      random_state: A `cupy.random.RandomState` or an integer seed or `None`.
    """
    super().__init__(n_components=n_components, random_state=random_state)
    self.rank = utils.check_positive_value(rank, 'rank')

  @property
  def _n_components(self):
    return self.n_components * self.rank

  def _validate_format(self, X: ArrayOnCPUOrGPU,
                        Y: Optional[ArrayOnCPUOrGPU] = None,
                        reset : bool = False):
    """Checks the `n_features` dimension in the data, i.e. the last axis.

    Note:
      If `Y` is given then:
        `n_features` is equal to the number of features in `Y`
        `n_components` is equal to the number of features in `X`
      Otherwise:
        `n_features` is equal to the number of features `X`.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
      reset: Whether to reset the `n_features_` parameter.

    Raises:
      ValueError: If `not reset` and the `n_features_` dimension doesn't match.
    """
    n_features = Y.shape[-1] if Y is not None else X.shape[-1]
    if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
      self.n_features_ = n_features
    if Y is not None:
      if Y.shape[-1] != self.n_features_:
        raise ValueError(
          'Data `Y` has a different number of features than during fit.',
          f'({Y.shape[-1]} != {self.n_features_})')
      if X.shape[-1] != self._n_components:
        raise ValueError(
          'Data `X` has incorrect number of features.',
          f'({X.shape[-1]} != {self._n_components})')
    else:
      if X.shape[-1] != self.n_features_:
        raise ValueError(
          'Data `X` has a different number of features than during fit.',
          f'({X.shape[-1]} != {self.n_features_})')

  def _make_projection_components(self, X: ArrayOnCPUOrGPU,
                                  Y: Optional[ArrayOnCPUOrGPU] = None):
    """Initializes internal variables, called by `fit`.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.
    """
    self.components_ = self.random_state.normal(
      size=[self.n_features_, self.n_components])
    self.scaling_ = 1. / cp.sqrt(self.n_components) if Y is None else 1.

  def _project(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the input array, called by `transform`.

    Note: Analogous to vanilla Gaussian RP with with `n_components * rank`
      number of components.

    Args:
      X: A data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    X_proj =  self.scaling_ * utils.matrix_mult(
      X.reshape([-1, self.n_features_]), self.components_)
    return X_proj.reshape(X.shape[:-1] + (-1,))

  def _project_outer_prod(self, X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the outer product of input arrays, called by `transform`.

    Note: Assumes that `X` is already projected, and hence, only projects `Y`
    and takes their elementwise product, i.e. projection applied recursively.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    XY_proj = self.scaling_ * X * utils.matrix_mult(
      Y.reshape([-1, self.n_features_]), self.components_).reshape(
        Y.shape[:-1] + (-1,))
    return XY_proj


# ------------------------------------------------------------------------------

class DiagonalProjection(RandomProjection):
  """Class for diagonal projection.

  Warning: This function is not meant to be used as a standalone RP, it is
    only meant to be used inside the low-rank signature kernel algorithm
    combined with random Fourier features as static features.
  """
  def __init__(self, internal_size=2):
    """Initializer for `DiagonalProjection` class."""
    self.internal_size = internal_size

  def _validate_data(self, X: ArrayOnCPUOrGPU,
                     Y: Optional[ArrayOnCPUOrGPU] = None,
                     reset: bool = False
                     ) -> Tuple[ArrayOnCPUOrGPU, Optional[ArrayOnCPUOrGPU]]:
    """Validates the input data, i.e. reshape `Y` and check dims.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
      reset: Whether to reset the `n_features` parameter.
    """
    self._validate_format(X, Y=Y, reset=reset)

  def _validate_format(self, X: ArrayOnCPUOrGPU,
                        Y: Optional[ArrayOnCPUOrGPU] = None,
                        reset: bool = False):
    """Validates the `n_features` dimension in the data, i.e. the last axis.

    Note: `n_features` is equal to the number of features in `X` and `Y`.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU.
      reset: Whether to reset the `n_features_` parameter.

    Raises:
      ValueError: If `not reset` and the `n_features_` dimension doesn't match.
    """
    n_features = X.shape[-1]
    q = self.internal_size
    if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
      self.n_features_ = n_features
    if X.shape[-1] != self.n_features_:
      raise ValueError(
        'Data `X` has a different number of features than during fit.',
        f'({X.shape[-1]} != {self.n_features_})')
    if Y is not None and Y.shape[-1] // q != self.n_features_:
      raise ValueError(
        'Data `Y` has a different number of features than during fit.',
        f'({Y.shape[-1]} != {self.n_features_})')

  def _make_projection_components(self, X: ArrayOnCPUOrGPU,
                                  Y: Optional[ArrayOnCPUOrGPU] = None):
    """Placeholder, this class does not have any internal variables.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.
    """
    pass

  def _project(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Projects the input array, called by `transform`.

    Args:
      X: A data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    q = self.internal_size
    return X.reshape(X.shape[:-1] + (q, -1))

  def _project_outer_prod(self, X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the Hadamard product and rescales, called by `transform`.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU.

    Returns:
      The projected outer-product array on GPU.
    """
    q = self.internal_size
    Y = Y.reshape(Y.shape[:-1] + (q, -1))
    return cp.sqrt(self.n_features_) * cp.reshape(
      X[..., None, :] * Y[..., None, :, :], X.shape[:-2] + (-1, X.shape[-1]))


# ------------------------------------------------------------------------------