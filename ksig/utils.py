"""Validation, linear algebra and probability utilities."""

import cupy as cp
import numpy as np

from cupy.random import RandomState
from numbers import Integral, Number
from typing import Optional, Sequence, Tuple, Union


ArrayOnCPU = np.ndarray
ArrayOnGPU = cp.ndarray
ArrayOnCPUOrGPU = Union[cp.ndarray, np.ndarray]
RandomStateOrSeed = Union[Integral, RandomState]

_EPS = 1e-12


# -----------------------------------------------------------------------------
# Type checking.
# -----------------------------------------------------------------------------

def check_positive_value(scalar: Number, name: str) -> Number:
  """Checks whether `scalar` is a positive number.

  Args:
    scalar: A variable to check.
    name: The name of the variable.

  Returns:
    The variable unchanged or raises an error if it is not positive.
  """
  if scalar <= 0:
    raise ValueError(f'The parameter \'{name}\' should have a positive value.')
  return scalar


def check_random_state(random_state: Optional[RandomStateOrSeed] = None
                       ) -> RandomState:
  """Check if random state or seed is valid and return a random state object.

  Args:
      random_state: A `cupy.random.RandomState`, an `int` seed or `None`.

  Raises:
      ValueError: If the value is invalid for creating a random state.

  Returns:
      A `cupy.random.RandomState` object.
  """
  if random_state is None or isinstance(random_state, Integral):
    return RandomState(random_state)
  elif isinstance(random_state, RandomState):
    return random_state
  raise ValueError(
    f'{random_state} cannot seed a `cupy.random.RandomState` instance')


# -----------------------------------------------------------------------------
# Linear Algebra.
# -----------------------------------------------------------------------------

def multi_cumsum(M: ArrayOnGPU, exclusive: bool = False, axis: int = -1
         ) -> ArrayOnGPU:
  """Computes the cumulative sum along a given set of axes.

  Args:
    M: A data array on GPU.
    axis: An axis or a set of axes.
  """

  ndim = M.ndim
  axis = [axis] if cp.isscalar(axis) else axis
  axis = [ndim+ax if ax < 0 else ax for ax in axis]

  if exclusive:
    # Slice off last element.
    slices = tuple(
      slice(-1) if ax in axis else slice(None) for ax in range(ndim))
    M = M[slices]

  for ax in axis:
    M = cp.cumsum(M, axis=ax)

  if exclusive:
    # Pre-pad with zeros.
    pads = tuple((1, 0) if ax in axis else (0, 0) for ax in range(ndim))
    M = cp.pad(M, pads)

  return M


def matrix_diag(A: ArrayOnGPU) -> ArrayOnGPU:
  """Extracts the diagonals from a batch of matrices.

  Args:
    A: A batch of matrices of shape `[..., d, d]`.

  Returns:
    The extracted diagonals of shape `[..., d]`.
  """
  return cp.einsum('...ii->...i', A)


def matrix_mult(X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None,
                transpose_X: bool = False, transpose_Y: bool = False
                ) -> ArrayOnGPU:
  """Performs batch matrix multiplication.

  Args:
    X: A batch of matrices.
    Y: Another batch of matrices (if not given uses `X`).
    transpose_X: Whether to transpose `X`.
    transpose_Y: Whether to transpose `Y`.

  Returns:
    The result of matrix multiplication, another batch of matrices.
  """
  subscript_X = '...ji' if transpose_X else '...ij'
  subscript_Y = '...kj' if transpose_Y else '...jk'
  return cp.einsum(
    f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)


def squared_norm(X: ArrayOnGPU, axis: int = -1) -> ArrayOnGPU:
  """Computes the squared norm by reducing over a given axis.

  Args:
    X: An n-dim. array to compute the norm of.
    axis: An axis to perform the reduction over.

  Returns:
    An (n-1)-dim. array containing the squared norms.
  """
  return cp.sum(cp.square(X), axis=axis)


def squared_euclid_dist(X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None
                        ) -> ArrayOnGPU:
  """Computes pairwise squared Euclidean distances.

  Args:
    X: An array of shape `[..., m, d]`.
    Y: Another array of shape `[..., n, d]`. Uses `X` if not given.

  Returns:
    An array of shape `[..., m, n]`.
  """
  X_n2 = squared_norm(X)
  if Y is None:
    D2 = (X_n2[..., :, None] + X_n2[..., None, :]
          - 2 * matrix_mult(X, X, transpose_Y=True))
  else:
    Y_n2 = squared_norm(Y, axis=-1)
    D2 = (X_n2[..., :, None] + Y_n2[..., None, :]
          - 2 * matrix_mult(X, Y, transpose_Y=True))
  return D2


def outer_prod(X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
  """Computes the outer product of two batch of vectors along the last axes.

  Args:
    X: A batch of vectors of shape `[..., d1]`.
    Y: A batch of vectors of shape `[..., d2]`.

  Returns:
    A batch of vectors of shape `[..., d1 * d2]`.
  """
  return cp.reshape(X[..., :, None] * Y[..., None, :], X.shape[:-1] + (-1,))


def robust_sqrt(X: ArrayOnGPU) -> ArrayOnGPU:
  """Robust elementwise square root.

  Args:
      X: An array to take the elementwise square root of.

  Returns:
      An array of the same shape.
  """
  return cp.sqrt(cp.maximum(X, _EPS))


def euclid_dist(self, X: ArrayOnGPU, Y: Optional[ArrayOnGPU] = None
                ) -> ArrayOnGPU:
  """Computes pairwise Euclidean distances.

  Args:
    X: An array of shape `[..., m, d]`.
    Y: Another array of shape `[..., n, d]`. Uses `X` if not given.

  Returns:
    An array of shape `[..., m, n]`.
  """
  return robust_sqrt(squared_euclid_dist(X, Y))


def robust_nonzero(X: ArrayOnGPU) -> ArrayOnGPU:
  """Robust elementwise nonzero check.

  Args:
      X: An array to check the elements of.

  Returns:
      A boolean array of the same shape.
  """
  return cp.abs(X) > _EPS


# -----------------------------------------------------------------------------
# Probability.
# -----------------------------------------------------------------------------

def draw_rademacher_matrix(shape: Sequence[int], prob: float = 0.5,
                           random_state: Optional[RandomStateOrSeed] = None
                           ) -> ArrayOnGPU:
  """Draw a random matrix with i.i.d. Rademacher entries.

  Args:
    shape: Shape of the matrix.
    prob: Probability of an entry being 1.
    random_state: A `cupy.random.RandomState` or an integer seed or `None`.

  Returns:
    A matrix of shape `shape`.
  """
  random_state = check_random_state(random_state)
  return cp.where(
    random_state.uniform(size=shape) < prob, cp.ones(shape), -cp.ones(shape))


def draw_bernoulli_matrix(shape: Sequence[int], prob: float = 0.5,
                          random_state: Optional[RandomStateOrSeed] = None
                          ) -> ArrayOnGPU:
  """Draw a random matrix with i.i.d. Bernoulli entries.

  Args:
    shape: Shape of the matrix.
    prob: Probability of an entry being 1.
    random_state: A `cupy.random.RandomState` or an integer seed or `None`.

  Returns:
    A matrix of shape `shape`.
  """
  random_state = check_random_state(random_state)
  return cp.where(
    random_state.uniform(size=shape) < prob, cp.ones(shape), cp.zeros(shape))


# -----------------------------------------------------------------------------
# Projection utils.
# -----------------------------------------------------------------------------

def subsample_outer_prod(X: ArrayOnGPU, Y: ArrayOnGPU,
                          sampled_idx: Union[ArrayOnGPU, Sequence[int]]
                          ) -> ArrayOnGPU:
  """Computes a subsampled outer product of two batch of features.

  Args:
    X: A data array on GPU.
    Y: An optional data array on GPU.
    sampled_idx: Indices to sample from the Cartesian product.

  Returns:
    The outer product of `X` and `Y` subsampled.
  """
  idx_X = cp.arange(X.shape[-1]).reshape([-1, 1, 1])
  idx_Y = cp.arange(Y.shape[-1]).reshape([1, -1, 1])
  idx_pairs = cp.reshape(cp.concatenate(
    (idx_X + cp.zeros_like(idx_Y), idx_Y + cp.zeros_like(idx_X)),
    axis=-1), (-1, 2))
  sampled_idx_pairs = cp.squeeze(cp.take(idx_pairs, sampled_idx, axis=0))
  X_proj = cp.take(X, sampled_idx_pairs[:, 0], axis=-1)
  Y_proj = cp.take(Y, sampled_idx_pairs[:, 1], axis=-1)
  return X_proj * Y_proj


def compute_count_sketch(X: ArrayOnGPU, hash_idx: ArrayOnGPU,
                         hash_bit: ArrayOnGPU,
                         n_components: Optional[int] = None) -> ArrayOnGPU:
  """Computes the count sketch of a feature array.

  Args:
    X: A data array on GPU.
    hash_idx: The hash indices.
    hash_bit: The hash bits.
    n_components: The number of sketch components.

  Returns:
    Sketched features on GPU.
  """
  # If `n_components is None`, get it from `hash_idx`.
  n_components = n_components or cp.max(hash_idx)
  hash_mask = cp.asarray(
    hash_idx[:, None] == cp.arange(n_components)[None, :], dtype=X.dtype)
  X_count_sketch = cp.einsum('...i,ij,i->...j', X, hash_mask, hash_bit)
  return X_count_sketch


def convolve_fft(X: ArrayOnGPU, Y: ArrayOnGPU) -> ArrayOnGPU:
  """Convolves two feature arrays via FFT.

  Args:
    X: A data array on GPU.
    Y: An optional data array on GPU.

  Returns:
    Convolved features on GPU."""
  X_fft = cp.fft.fft(X, axis=-1)
  Y_fft = cp.fft.fft(Y, axis=-1)
  XY = cp.real(cp.fft.ifft(X_fft * Y_fft, axis=-1))
  return XY


# -----------------------------------------------------------------------------