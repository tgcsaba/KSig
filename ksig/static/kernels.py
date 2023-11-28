"""Static (state-space) kernels."""

import cupy as cp
import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from typing import Optional

from .. import utils
from ..utils import _EPS, ArrayOnGPU, ArrayOnCPUOrGPU


# ------------------------------------------------------------------------------
# Kernel base classes.
# ------------------------------------------------------------------------------

class Kernel(BaseEstimator, metaclass=ABCMeta):
  """Base class for kernels.

  Deriving classes should implement the following methods:
    _K: Computes the kernel matrix between two data arrays.
    _Kdiag: Computes the diagonal kernel entries of a given data array.
    _validate_data: Performs any data checking and reshaping on data arrays.

  Warning: This class should not be used directly, use derived classes.
  """

  def fit(self, X: ArrayOnCPUOrGPU, y: Optional[ArrayOnCPUOrGPU] = None
            ) -> 'Kernel':
    return self

  @abstractmethod
  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the kernel matrix.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """

  @abstractmethod
  def _Kdiag(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the diagonal kernel entries.

    Args:
      X: A data array on GPU.

    Returns:
      Diagonal entries of the kernel matrix of `X`.
    """

  @abstractmethod
  def _validate_data(X: ArrayOnCPUOrGPU, reset: bool = False
                     ) -> ArrayOnCPUOrGPU:
    """Validates the input data array `X`.

    This method returns `X` as derived classes might make changes to it.

    Args:
      X: A data array on CPU or GPU.
      reset: Whether to reset internal variables if any.

    Returns:
      Validated data on CPU or GPU.
    """

  def __call__(self, X: ArrayOnCPUOrGPU, Y: Optional[ArrayOnCPUOrGPU] = None,
               diag: bool = False, return_on_gpu: bool = False
               ) -> ArrayOnCPUOrGPU:
    """Implementes the basic call method of a kernel object.

    It takes as input one or two arrays, either on CPU (as `numpy`) or on
    GPU (as `cupy`), and computes the corresponding kernel matrix.

    Args:
      X: A data array on CPU or GPU.
      Y: An optional data array on CPU or GPU (if not given uses `X`).
      diag: Whether to compute only the diagonal. Ignores `Y` in this case.
      return_on_gpu: Whether to return the result on GPU.

    Returns:
      A kernel matrix or its diagonal entries on CPU or GPU.
    """
    # Validate data and move it to GPU.
    X = cp.asarray(self._validate_data(X))
    if diag:
      K = self._Kdiag(X)
    else:
      Y = cp.asarray(self._validate_data(Y)) if Y is not None else None
      K =  self._K(X, Y)
    if not return_on_gpu:
      K = cp.asnumpy(K)
    return K


class StaticKernel(Kernel, metaclass=ABCMeta):
  """Base class for static kernels.
  
  Note: Static kernels merge the last two axes for arrays with `ndim > 2`,
  so that it can be readily used on sequential data without pipeline changes.
  
  Deriving classes should implement the following methods:
    _K: Computes the kernel matrix between two data arrays.
    _Kdiag: Computes the diagonal kernel entries of a given data array.

  Warning: This class should not be used directly, use derived classes.
  """

  def _validate_data(self, X: ArrayOnCPUOrGPU, reset: bool = False
                     ) -> ArrayOnCPUOrGPU:
    """This method merges the last two axes for input arrays with `ndims > 2`.

    Args:
      X: A data array on CPU or GPU.
      reset: Provided for API consistency, unused.

    Returns:
      Reshaped data array on CPU or GPU.
    """
    # Merge the time axis with the feature axis.
    if X.ndim > 2:
      X = X.reshape(X.shape[:-2] + (-1,))
    return X


# ------------------------------------------------------------------------------
# Dot-product kernels.
# ------------------------------------------------------------------------------

class LinearKernel(StaticKernel):
  """Class for linear static kernel."""

  def __init__(self, scale: float = 1.):
    """Initializes the `LinearKernel` object.
    
    Args:
      scale: Scaling hyperparameter used to rescale kernel entries.
    """
    self.scale = utils.check_positive_value(scale, 'scale')

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the Gram matrix and rescales by `scale`.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The Gram matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    return self.scale * utils.matrix_mult(X, Y, transpose_Y=True)

  def _Kdiag(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the squared Euclid. norm of the samples in `X` and rescales it.

    Args:
      X: A data array on GPU.

    Returns:
      Rescaled squared Euclidean norm of the samples in `X`.
    """
    return self.scale * utils.squared_norm(X, axis=-1)


class PolynomialKernel(StaticKernel):
  """Class for polynomial kernel."""

  def __init__(self, degree: float = 3., gamma: float = 1., scale: float = 1.):
    """Initializes the `PolynomialKernel` object.

    Args:
      degree: Polynomial degree.
      gamma: Offset applied before raising to a power.
      scale: Scaling hyperparameter used to rescale kernel entries.
    """
    self.degree = utils.check_positive_value(degree, 'degree')
    self.gamma = gamma
    self.scale = utils.check_positive_value(scale, 'scale')

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the polynomial kernel matrix between data arrays.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      Poly. kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    return self.scale * cp.power(
      utils.matrix_mult(X, Y, transpose_Y=True) + self.gamma, self.degree)

  def _Kdiag(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the diagonal entries for the polynomial kernel.

    Args:
      X: A data array on GPU.

    Returns:
      Diagonal entries of the poly. kernel matrix of `X`.
    """
    return self.scale * cp.power(
      utils.squared_norm(X, axis=-1) + self.gamma, self.degree)


# ------------------------------------------------------------------------------
# Stationary kernels
# ------------------------------------------------------------------------------

class StationaryKernel(StaticKernel):
  """Base class for stationary kernels.
  
  Stationary kernels considered here have diagonals equal to 1.
  
  Deriving classes should implement the following methods:
    _K: Computes the kernel matrix between two data arrays.

  Warning: This class should not be used directly, use derived classes.
  """

  def __init__(self, bandwidth: float = 1.) -> None:
    """Initializes the `StationaryKernel` object.

    Args:
      bandwidth: Bandwidth hyperparameter that inversely scales the input data.
    """
    self.bandwidth = utils.check_positive_value(bandwidth, 'bandwidth')

  def _Kdiag(self, X: ArrayOnGPU) -> ArrayOnGPU:
    """Computes the diagonal entries for stationary kernel.

    Args:
      X: A data array on GPU.

    Returns:
      Diagonal entries of the kernel matrix.
    """
    return cp.full((X.shape[0],), 1)


class RBFKernel(StationaryKernel):
  """RBF kernel also called the Gaussian kernel ."""

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the RBF kernel matrix.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    D2_scaled = utils.squared_euclid_dist(X, Y) / cp.maximum(
      2*self.bandwidth**2, _EPS)
    return cp.exp(-D2_scaled)


class Matern12Kernel(StationaryKernel):
  """Matern12 kernel also called the exponential kernel."""

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the Matern12 kernel matrix.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    D_scaled = utils.euclid_dist(X, Y) / cp.maximum(self.bandwidth, _EPS)
    return cp.exp(-D_scaled)


class Matern32Kernel(StationaryKernel):
  """Matern32 (static) kernel ."""

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the Matern32 kernel matrix.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    sqrt3 = cp.sqrt(3.)
    D_scaled = sqrt3 * utils.euclid_dist(X, Y) / cp.maximum(
      self.bandwidth, _EPS)
    return (1. + D_scaled) * cp.exp(-D_scaled)


class Matern52Kernel(StationaryKernel):
  """Matern52 (static) kernel ."""

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the Matern52 kernel matrix.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    D2_scaled = 5 * utils.squared_euclid_dist(X, Y) / cp.maximum(
      self.bandwidth**2, _EPS)
    D_scaled = utils.robust_sqrt(D2_scaled)
    return (1. + D_scaled + D2_scaled / 3.) * cp.exp(-D_scaled)


class RationalQuadraticKernel(StationaryKernel):
  """Rational Quadratic (static) kernel ."""

  def __init__(self, bandwidth: float = 1., alpha: float = 1.):
    """Initializes the rational quadratic kernel.

    Args:
      bandwidth: Bandwidth hyperparameter that inversely scales the input data.
      alpha: Alpha hyperparameter for the rational quadratic kernel.
    """
    super().__init__(bandwidth=bandwidth)
    self.alpha = utils.check_positive_value(alpha, 'alpha')

  def _K(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
    """Computes the rational quadratic kernel matrix.

    Args:
      X: A data array on GPU.
      Y: An optional data array on GPU (if not given uses `X`).

    Returns:
      The kernel matrix between `X` and `Y`, or `X` and `X` if `Y is None`.
    """
    D2_scaled = utils.squared_euclid_dist(X, Y) / cp.maximum(
      2 * self.alpha * self.bandwidth**2, _EPS)
    return cp.power((1 + D2_scaled), -self.alpha)


# -----------------------------------------------------------------------------