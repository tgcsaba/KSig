"""Signature computation dynamic programming algorithms."""

import cupy as cp
import math
import numpy as np
import warnings

from .projections import (DiagonalProjection, RandomProjection,
                          TensorizedRandomProjection)
from .utils import _EPS, ArrayOnGPU, multi_cumsum
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from typing import List, Optional, Union

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


# -----------------------------------------------------------------------------
# Signature Algorithms.
# -----------------------------------------------------------------------------

def signature_kern(M: ArrayOnGPU, n_levels: int, order: int = -1,
                   difference: bool = True, return_levels: bool = False
                   ) -> ArrayOnGPU:
  """Computes the full-rank signature kernel using the kernel trick.

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.
    n_levels: Number of signature levels.
    order: Signature embedding order.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the kernel for each level separately.

  Returns:
    The signature kernel matrix of shape `[n_X, n_Y]` or `[n]`, see `M` above.
  """
  order = n_levels if order <= 0 or order >= n_levels else order
  if order==1:
    return signature_kern_first_order(
      M, n_levels, difference=difference, return_levels=return_levels)
  else:
    return signature_kern_higher_order(
      M, n_levels, order=order, difference=difference,
      return_levels=return_levels)


def signature_kern_first_order(M: ArrayOnGPU, n_levels: int,
                               difference: bool = True,
                               return_levels: bool = False) -> ArrayOnGPU:
  """Computes the first-order full-rank signature kernel using a kernel trick.

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.
    n_levels: Number of signature levels.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the kernel for each level separately.

  Returns:
    The signature kernel matrix of shape `[..., n_X, n_Y]` or `[..., n]`,
      depending on `M` above, and `...` is `n_levels` when `return_levels`.
  """

  if difference:
    M = cp.diff(cp.diff(M, axis=-2), axis=-1)
  if M.ndim == 4:
    n_X, n_Y  = M.shape[:2]
    K = cp.ones((n_X, n_Y), dtype=M.dtype)
  else:
    n_X = M.shape[0]
    K = cp.ones((n_X,), dtype=M.dtype)

  if return_levels:
    K = [K, cp.sum(M, axis=(-2, -1))]
  else:
    K += cp.sum(M, axis=(-2, -1))

  R = cp.copy(M)
  for i in range(1, n_levels):
    R = M * multi_cumsum(R, exclusive=True, axis=(-2, -1))
    if return_levels:
      K.append(cp.sum(R, axis=(-2, -1)))
    else:
      K += cp.sum(R, axis=(-2, -1))

  return cp.stack(K, axis=0) if return_levels else K


def signature_kern_higher_order(M: ArrayOnGPU, n_levels: int, order: int,
                                difference: bool = True,
                                return_levels: bool = False) -> ArrayOnGPU:
  """Computes the higher-order full rank signature kernel using a kernel trick.

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.
    n_levels: Number of signature levels.
    order: Signature embedding order.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the kernel for each level separately.

  Returns:
    The signature kernel matrix of shape `[..., n_X, n_Y]` or `[..., n]`,
      depending on `M` above, and `...` is `n_levels` when `return_levels`.
  """

  if difference:
    M = cp.diff(cp.diff(M, axis=-2), axis=-1)

  if M.ndim == 4:
    n_X, n_Y = M.shape[0], M.shape[1]
    K = cp.ones((n_X, n_Y), dtype=M.dtype)
  else:
    n_X = M.shape[0]
    K = cp.ones((n_X,), dtype=M.dtype)

  if return_levels:
    K = [K, cp.sum(M, axis=(-2, -1))]
  else:
    K += cp.sum(M, axis=(-2, -1))

  R = cp.copy(M)[None, None, ...]
  for i in range(1, n_levels):
    d = min(i+1, order)
    R_next = cp.empty((d, d) + M.shape, dtype=M.dtype)
    # Both time axes are non-repeating.
    R_next[0, 0] = M * multi_cumsum(
      cp.sum(R, axis=(0, 1)), exclusive=True, axis=(-2, -1))
    for r in range(1, d):
      R_next[0, r] = 1./(r+1) * M * multi_cumsum(
        cp.sum(R[:, r-1], axis=0), exclusive=True, axis=-2)
      R_next[r, 0] = 1./(r+1) * M * multi_cumsum(
        cp.sum(R[r-1, :], axis=0), exclusive=True, axis=-1)
      for s in range(1, d):
        R_next[r, s] = 1./((r+1)*(s+1)) * M * R[r-1, s-1]
    R = R_next
    if return_levels:
      K.append(cp.sum(R, axis=(0, 1, -2, -1)))
    else:
      K += cp.sum(R, axis=(0, 1, -2, -1))

  return cp.stack(K, axis=0) if return_levels else K


# -----------------------------------------------------------------------------
# Low-Rank Signature Algorithms.
# -----------------------------------------------------------------------------

def signature_kern_low_rank(
  U: ArrayOnGPU, n_levels: int, order: int = -1, difference: bool = True,
  return_levels: bool = False,
  projections : Optional[List[RandomProjection]] = None
  ) -> Union[List[ArrayOnGPU], ArrayOnGPU]:
  """Computes the low-rank signature kernel in feature space.

  Args:
    U: Transformed sequences of shape `[n_X, l_X, n_d]`.
    n_levels: Number of signature levels.
    order: Signature embedding order.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the features for each level separately.
    projections: Random projections for the outer product approximation.

  Returns:
    The signature features Sig(X).
  """
  order = n_levels if order <= 0 or order >= n_levels else order
  if order==1:
    return signature_kern_first_order_low_rank(
      U, n_levels, difference=difference, return_levels=return_levels,
      projections=projections)
  else:
    return signature_kern_higher_order_low_rank(
      U, n_levels, order=order, difference=difference,
      return_levels=return_levels, projections=projections)


def signature_kern_first_order_low_rank(
  U: ArrayOnGPU, n_levels: int, difference: bool = True,
  return_levels: bool = False,
  projections: Optional[List[RandomProjection]] = None
  ) -> Union[List[ArrayOnGPU], ArrayOnGPU]:
  """Computes the first-order low-rank signature kernel in feature space.

  Args:
    U: Transformed sequences of shape `[n_X, l_X, n_d]`.
    n_levels: Number of signature levels.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the features for each level separately.
    projections: Random projections for outer product approximation.

  Returns:
    The first-order signature features Sig(X).
  """

  if isinstance(U, list):
    if difference:
      U = [cp.diff(U[i], axis=1) for i in range(n_levels)]

    n_X, l_X, n_d = U[0].shape
    P = cp.ones((n_X, 1), dtype=U[0].dtype)
    R = (projections[0](U[0], return_on_gpu=True) if projections is not None
         else cp.copy(U[0]))
  else:
    if difference:
      U = cp.diff(U, axis=1)
    n_X, l_X, n_d = U.shape
    P = cp.ones((n_X, 1), dtype=U.dtype)
    R = (projections[0](U, return_on_gpu=True) if projections is not None else
         cp.copy(U))

  if (projections is not None and
      isinstance(projections[0], TensorizedRandomProjection)):
    R_reshaped = R.reshape(
      [n_X, l_X, projections[0].n_components, projections[0].rank])
    R_sum = cp.sum(R_reshaped, axis=(1, -1))
  else:
    R_sum = cp.sum(R, axis=1)

  if return_levels:
    P = [P, R_sum.reshape([n_X, -1])]
  else:
    P = cp.concatenate((P, R_sum.reshape([n_X, -1])), axis=-1)

  for i in range(1, n_levels):
    R = multi_cumsum(R, axis=1, exclusive=True)
    if projections is None:
      if isinstance(U, list):
        R = cp.reshape(R[..., :, None] * U[i][..., None, :], (n_X, l_X, -1))
      else:
        R = cp.reshape(R[..., :, None] * U[..., None, :], (n_X, l_X, -1))
      R_sum = cp.sum(R, axis=1)
    else:
      if isinstance(U, list):
        R = projections[i](R, U[i], return_on_gpu=True)
      else:
        R = projections[i](R, U, return_on_gpu=True)
      if isinstance(projections[i], TensorizedRandomProjection):
        R_reshaped = R.reshape(
          [n_X, l_X, projections[i].n_components, projections[i].rank])
        R_sum = cp.sum(R_reshaped, axis=(1, -1))
      else:
        R_sum = cp.sum(R, axis=1)
    R_sum = R_sum.reshape([n_X, -1])
    if return_levels:
      P.append(R_sum)
    else:
      P = cp.concatenate((P, R_sum), axis=-1)
  return P

def signature_kern_higher_order_low_rank(
  U: ArrayOnGPU, n_levels: int, order: int = -1, difference: bool = True,
  return_levels: bool = False,
  projections: Optional[List[RandomProjection]] = None
  ) -> Union[List[ArrayOnGPU], ArrayOnGPU]:
  """Computes the higher-order low-rank signature kernel in feature space.

  Args:
    U: Transformed sequences of shape `[n_X, l_X, n_d]`.
    n_levels: Number of signature levels.
    order: Signature embedding order.
    difference: Whether to take increments of lifted sequences in the RKHS.
    return_levels: Whether to return the features for each level separately.
    projections: Random projections for outer product approximation.

  Returns:
    The higher-order signature features Sig(X).
  """
  if isinstance(U, list):
    if difference:
      U = [cp.diff(U[i], axis=1) for i in range(n_levels)]
    n_X, l_X = U[0].shape[:2]
    n_d = U[0].shape[-1]
    P = cp.ones((n_X, 1), dtype=U[0].dtype)
    R = (projections[0](U[0], return_on_gpu=True) if projections is not None
         else cp.copy(U[0]))
  else:
    if difference:
      U = cp.diff(U, axis=1)
    n_X, l_X = U.shape[:2]
    n_d = U.shape[-1]
    P = cp.ones((n_X, 1), dtype=U.dtype)
    R = (projections[0](U, return_on_gpu=True) if projections is not None else
         cp.copy(U))

  if (projections is not None and
      isinstance(projections[0], TensorizedRandomProjection)):
    R_reshaped = R.reshape(
      [n_X, l_X, projections[0].n_components, projections[0].rank])
    R_sum = cp.sum(R_reshaped, axis=(1, -1))
  else:
    R_sum = cp.sum(R, axis=1)

  R_sum = R_sum.reshape([n_X, -1])
  if return_levels:
    P = [P, R_sum]
  else:
    P = cp.concatenate((P, R_sum), axis=-1)

  R = R[None]
  for i in range(1, n_levels):
    d = min(i+1, order)
    n_components = R.shape[-1] if projections is not None else n_d**(i+1)
    if (projections is not None and
        isinstance(projections[i], DiagonalProjection)):
      internal_size = projections[i].internal_size
      R_next = cp.empty((d, n_X, l_X, internal_size**(i+1), n_components))
    else:
      R_next = cp.empty((d, n_X, l_X, n_components))
    U_next = U[i] if isinstance(U, list) else U
    Q = multi_cumsum(cp.sum(R, axis=0), axis=1, exclusive=True)
    if projections is None:
      R_next[0] = cp.reshape(
        Q[..., :, None] * U_next[..., None, :], (n_X, l_X, -1))
    else:
      R_next[0] = projections[i](Q, U_next, return_on_gpu=True)
    if projections is None:
      for r in range(1, d):
        R_next[r] = 1./(r+1) * cp.reshape(
          R[r-1, ..., :, None] * U_next[..., None, :],
          (n_X, l_X, n_components))
    else:
      for r in range(1, d):
        R_next[r] = 1./(r+1) * projections[i](
          R[r-1], U_next, return_on_gpu=True)
    R = R_next
    if (projections is not None and
        isinstance(projections[i], TensorizedRandomProjection)):
      R_reshaped = R.reshape(
        [d, n_X, l_X, projections[i].n_components, projections[i].rank])
      R_sum = cp.sum(R_reshaped, axis=(0, 2, -1))
    else:
      R_sum = cp.sum(R, axis=(0, 2))
    R_sum = R_sum.reshape([n_X, -1])
    if return_levels:
      P.append(R_sum)
    else:
      P = cp.concatenate((P, R_sum), axis=-1)
  return P


# -----------------------------------------------------------------------------
# Signature-PDE Kernel.
# -----------------------------------------------------------------------------

@cuda.jit
def _signature_kern_pde(M: ArrayOnGPU, K: ArrayOnGPU,
                        K_sol: ArrayOnGPU):
  """CUDA kernel for computing the signature-PDE kernel.

  Args:
    M: A data array on GPU of shape `[n_X, n_Y, l_X, l_Y]`.
    K: Output array of shape `[n_X, n_Y]`.
    K_sol: Temp array of shape `[n_X, n_Y, 3 * min(l_X, l_Y)]` or of shape
      `[0]`. In the latter case, shared memory should be allocated at the launch
      of the kernel of `size = 3 * min(l_X, l_Y)`.
  """
  _, _, l_X, l_Y = M.shape
  l = min(l_X, l_Y)
  num_iter = l_X + l_Y - 1
  idx_X, idx_Y = cuda.blockIdx.x, cuda.blockIdx.y
  # Temp array that contains the current and the previous 2 antidiagonals.
  if K_sol.ndim == 1:
    temp = cuda.shared.array(0, M.dtype)
  else:
    temp = K_sol[idx_X, idx_Y]
  # Iterate over the antidiagonals one by one.
  for it in range(num_iter):
    num_anti_diag = min(it+1, l_X, l_Y, num_iter-it)
    # Break the given antidiagonal into chunks of size `blockDim.x`.
    num_chunks = 1 + (num_anti_diag-1) // cuda.blockDim.x
    # Starting index on the given antidiagonal.
    I, J = max(0, it - l_Y + 1), min(it, l_Y - 1)
    for p in range(num_chunks):
      # Location on the given antidiagonal.
      k = p * cuda.blockDim.x + cuda.threadIdx.x
      # Index to process on the given antidiagonal.
      i, j = I + k, J - k
      # if i < l_X and 0 <= j:  This is equivalent to the following.
      if k < num_anti_diag:
        Mij = M[idx_X, idx_Y, i, j]
        K01 = 1. if i==0 else temp[3*k-2] if I==0 else temp[3*k+1]
        K10 = 1. if j==0 else temp[3*k+1] if I==0 else temp[3*k+4]
        K00 = (1. if i==0 or j==0 else temp[3*k-3] if I==0 else temp[3*k]
               if I==1 else temp[3*k+3])
        temp[3*k+2] = (
          (K01 + K10) * (1. + 1./2*Mij+1./12*Mij**2) - K00*(1. - 1./12*Mij**2))
    # Wait until all the antidiagonal entries have been processed.
    cuda.syncthreads()
    # Roll the solution array by 1.
    for p in range(num_chunks):
      k = p * cuda.blockDim.x + cuda.threadIdx.x
      if k < num_anti_diag:
        temp[3*k] = temp[3*k+1]
        temp[3*k+1] = temp[3*k+2]
    # Wait for writes to be visible for all threads.
    cuda.threadfence_block()
    cuda.syncthreads()
  # Save the result.
  if cuda.threadIdx.x == 0:
    K[idx_X, idx_Y] = temp[2]


def signature_kern_pde(M: ArrayOnGPU, difference: bool = True) -> ArrayOnGPU:
  """Computes the signature-PDE kernel using a kernel trick.

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.
    difference: Whether to take increments of lifted sequences in the RKHS.
    vanilla_scheme: Whether to use the vanilla first-order PDE solver.

  Returns:
    The SigPDE kernel matrix of shape `[n_X, n_Y]` or `[n]`, see `M` above.
  """
  # Handle diagonal case.
  is_diag = M.ndim == 3
  if is_diag:
    M = M[:, None, :, :]
  if M.ndim != 4:
    raise ValueError('The `M` matrix must have `.ndim==3` or `.ndim==4`.')
  # Take increments.
  if difference:
    M = cp.diff(cp.diff(M, axis=-2), axis=-1)
  # Set the parameters.
  n_X, n_Y, l_X, l_Y = M.shape
  num_blocks = (n_X, n_Y)
  num_threads = min(l_X, l_Y, 1024)
  stream = cuda.default_stream()
  num_shared = 3 * min(l_X, l_Y)
  shared_mem = num_shared * M.itemsize
  max_shared_mem = cp.cuda.device.Device().attributes[
    'MaxSharedMemoryPerBlock']
  # Array of output kernel values.
  K = cp.empty((n_X, n_Y), dtype=M.dtype)
  # If temporary values do not fit in shared memory, use global.
  if shared_mem > max_shared_mem:
    # Can't use shared memory, allocate global array now.
    K_sol = cp.empty((n_X, n_Y, num_shared), dtype=M.dtype)
    shared_mem = 0
  else:
    # Fits in shared memory, allocate dynamically.
    K_sol = cp.empty((0), dtype=M.dtype)
  # Launch kernel maybe with shared memory for propagating the PDE solution.
  _signature_kern_pde[num_blocks, num_threads, stream, shared_mem](
    M, K, K_sol)
  if is_diag:
    K = cp.squeeze(K, axis=1)
  return K


# -----------------------------------------------------------------------------
# Global Alignment Kernel.
# -----------------------------------------------------------------------------

@cuda.jit
def _global_align_kern_log(logM: ArrayOnGPU, logK: ArrayOnGPU,
                           logK_sol: ArrayOnGPU):
  """CUDA kernel for computing the logarithm of the GA kernel.

  Args:
    logM: A data array on GPU of shape `[n_X, n_Y, l_X, l_Y]`.
    logK: Output array of shape `[n_X, n_Y]`.
    logK_sol: Temp array of shape `[n_X, n_Y, 3 * min(l_X, l_Y)]` or of shape
      `[0]`. In the latter case, shared memory should be allocated at the
      launch of the kernel of `size = 3 * min(l_X, l_Y)`.
  """
  _, _, l_X, l_Y = logM.shape
  l = min(l_X, l_Y)
  num_iter = l_X + l_Y - 1
  idx_X, idx_Y = cuda.blockIdx.x, cuda.blockIdx.y
  # Temp array that contains the current and the previous 2 antidiagonals.
  if logK_sol.ndim == 1:
    temp = cuda.shared.array(0, logM.dtype)
  else:
    temp = logK_sol[idx_X, idx_Y]
  # Iterate over the antidiagonals one by one.
  for it in range(num_iter):
    num_anti_diag = min(it+1, l_X, l_Y, num_iter-it)
    # Break the given antidiagonal into chunks of size `blockDim.x`.
    num_chunks = 1 + (num_anti_diag-1) // cuda.blockDim.x
    # Starting index on the given antidiagonal.
    I, J = max(0, it-l_Y+1), min(it, l_Y-1)
    for p in range(num_chunks):
      # Location on the given antidiagonal.
      k = p * cuda.blockDim.x + cuda.threadIdx.x
      # Index to process on the given antidiagonal.
      i, j = I + k, J - k
      # if i < l_X and 0 <= j:  This is equivalent to the following.
      if k < num_anti_diag:
        logMij = logM[idx_X, idx_Y, i, j]
        logK01 = -np.inf if i==0 else temp[3*k-2] if I==0 else temp[3*k+1]
        logK10 = -np.inf if j==0 else temp[3*k+1] if I==0 else temp[3*k+4]
        logK00 = (0. if i==0 and j==0 else -np.inf if i==0 or j==0 else
                  temp[3*k-3] if I==0 else temp[3*k] if I==1 else temp[3*k+3])
        # Use log-sum-exp trick.
        max_logK = max(logK01, logK10, logK00)
        temp[3*k+2] = logMij + max_logK + math.log(math.exp(logK01-max_logK) +
                                                   math.exp(logK10-max_logK) +
                                                   math.exp(logK00-max_logK))
    # Wait until all the antidiagonal entries have been processed.
    cuda.syncthreads()
    # Roll the solution array by 1.
    for p in range(num_chunks):
      k = p * cuda.blockDim.x + cuda.threadIdx.x
      if k < num_anti_diag:
        temp[3*k] = temp[3*k+1]
        temp[3*k+1] = temp[3*k+2]
    # Wait for writes to be visible for all threads.
    cuda.threadfence_block()
    cuda.syncthreads()
  # Save the result.
  if cuda.threadIdx.x == 0:
    logK[idx_X, idx_Y] = temp[2]


def global_align_kern_log(M: ArrayOnGPU) -> ArrayOnGPU:
  """Computes the Global Alignment Kernel.

  Args:
    M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]`
        for computing K with shape `[n_X, n_Y], or of shape `[n, l_X, l_Y]`
        in which case the output matrix has shape `[n]`.

  Returns:
    The GA kernel matrix of shape `[n_X, n_Y]` or `[n]`, see `M` above.
  """
  # Handle diagonal case.
  is_diag = M.ndim == 3
  if is_diag:
    M = M[:, None, :, :]
  if M.ndim != 4:
    raise ValueError('The `M` matrix must have `.ndim==3` or `.ndim==4`.')
  # Transform `M` to make it "infinitely divisible".
  M = M / (2. - M)
  # Do the computations in log-space.
  logM = cp.log(cp.maximum(M, _EPS))
  # Set the parameters.
  n_X, n_Y, l_X, l_Y = M.shape
  num_blocks = (n_X, n_Y)
  num_threads = min(l_X, l_Y, 1024)
  stream = cuda.default_stream()
  num_shared = 3 * min(l_X, l_Y)
  shared_mem = num_shared * M.itemsize
  max_shared_mem = cp.cuda.device.Device().attributes[
    'MaxSharedMemoryPerBlock']
  # Array of output kernel values.
  logK = cp.empty((n_X, n_Y), dtype=M.dtype)
  # If temporary values do not fit in shared memory, use global.
  if shared_mem > max_shared_mem:
    # Can't use shared memory, allocate global array now.
    logK_sol = cp.empty((n_X, n_Y, num_shared), dtype=M.dtype)
    shared_mem = 0
  else:
    # Fits in shared memory, allocate dynamically.
    logK_sol = cp.empty((0), dtype=M.dtype)
  # Launch kernel with shared memory for propagating the DP solution.
  _global_align_kern_log[num_blocks, num_threads, stream, shared_mem](
    logM, logK, logK_sol)
  if is_diag:
    logK = cp.squeeze(logK, axis=1)
  return logK


# -----------------------------------------------------------------------------
# Random Warping Series.
# -----------------------------------------------------------------------------

@cuda.jit
def _random_warping_series_dtw(D: ArrayOnGPU, P: ArrayOnGPU, P_sol: ArrayOnGPU,
                               warp_segments: ArrayOnGPU):
  """CUDA kernel for computing Random Warping Series features.

  Args:
    D: Squared distances from (variable length) warping series given as a GPU
      array of shape `[n_X, l_X, sum of l_Y]`.
    P: Output array for the log of features of shape `[n_X, n_Y]`.
    P_sol: An array of shape `[n_X, n_Y, 3 * min(l_X, l_Y)]` or of shape
      `[0]`. In the latter case, shared memory should be allocated at the
      launch of the kernel of `size = 3 * min(l_X, l_Y)`.
    warp_segments: Array of shape `[n_Y + 1]`, the endpoints of the warps.
  """
  idx_X, idx_Y = cuda.blockIdx.x, cuda.blockIdx.y
  l_X = D.shape[1]
  l_Y = warp_segments[idx_Y + 1] - warp_segments[idx_Y]
  num_iter = l_X + l_Y - 1
  # Temp array that contains the current and the previous 2 antidiagonals.
  if P_sol.ndim == 1:
    temp = cuda.shared.array(0, D.dtype)
  else:
    temp = P_sol[idx_X, idx_Y]
  # Iterate over the antidiagonals one by one.
  for it in range(num_iter):
    num_anti_diag = min(it+1, l_X, l_Y, num_iter-it)
    # Break the given antidiagonal into chunks of size `blockDim.x`.
    num_chunks = 1 + (num_anti_diag-1) // cuda.blockDim.x
    # Starting index on the given antidiagonal.
    I, J = max(0, it-l_Y+1), min(it, l_Y-1)
    for p in range(num_chunks):
      # Location on the given antidiagonal.
      k = p * cuda.blockDim.x + cuda.threadIdx.x
      # Index to process on the given antidiagonal.
      i, j = I + k, J - k
      # if i < l_X and 0 <= j:  This is equivalent to the following.
      if k < num_anti_diag:
        Dij = D[idx_X, i, warp_segments[idx_Y] + j]
        P01 = np.inf if i==0 else temp[3*k-2] if I==0 else temp[3*k+1]
        P10 = np.inf if j==0 else temp[3*k+1] if I==0 else temp[3*k+4]
        P00 = (0 if i==0 and j==0 else np.inf if i==0 or j==0 else
                  temp[3*k-3] if I==0 else temp[3*k] if I==1 else temp[3*k+3])
        temp[3*k+2] = Dij + min(P01, P10, P00)
    # Wait until all the antidiagonal entries have been processed.
    cuda.syncthreads()
    # Roll the solution array by 1.
    for p in range(num_chunks):
      k = p * cuda.blockDim.x + cuda.threadIdx.x
      if k < num_anti_diag:
        temp[3*k] = temp[3*k+1]
        temp[3*k+1] = temp[3*k+2]
    # Wait for writes to be visible for all threads.
    cuda.threadfence_block()
    cuda.syncthreads()
  # Save the result.
  if cuda.threadIdx.x == 0:
    P[idx_X, idx_Y] = temp[2]


def random_warping_series(D: ArrayOnGPU, warp_lens: ArrayOnGPU) -> ArrayOnGPU:
  """Computes the logarithm of Random Warping Series features.

  Args:
    M: Squared distances from (variable length) random time series given as a
      GPU array of shape `[n_X, l_X, sum of l_Y]`.
    warp_lens: Lengths of each warping series, array of shape `[n_Y]'.

  Returns:
    Random Warping Series features of shape `[n_X, n_Y]`.

  Raises:
    ValueError: If `D` does not have `.ndim==3`.
  """
  if D.ndim != 3:
    raise ValueError('`D` distances array must have `.ndim==3`.')
  # Set the parameters.
  n_X, l_X = D.shape[:2]
  n_Y = warp_lens.shape[0]
  l_Y = int(cp.max(warp_lens))
  num_blocks = (n_X, n_Y)
  num_threads = min(l_X, l_Y, 1024)
  stream = cuda.default_stream()
  num_shared = 3 * min(l_X, l_Y)
  shared_mem = num_shared * D.itemsize
  max_shared_mem = cp.cuda.device.Device().attributes[
    'MaxSharedMemoryPerBlock']
  # Array of output for the log of feature values.
  P = cp.empty((n_X, n_Y), dtype=D.dtype)
  # If temporary values do not fit in shared memory, use global.
  if shared_mem > max_shared_mem:
    # Can't use shared memory, allocate global array now.
    P_sol = cp.empty((n_X, n_Y, num_shared), dtype=D.dtype)
    shared_mem = 0
  else:
    # Fits in shared memory, allocate dynamically.
    P_sol = cp.empty((0), dtype=D.dtype)
  # Launch kernel maybe with shared memory for propagating the solution.
  warp_segments = cp.concatenate(
    (cp.zeros([1], dtype=int),
     cp.cumsum(warp_lens, axis=0)), axis=0)  # Endpoints of warps.
  _random_warping_series_dtw[num_blocks, num_threads, stream, shared_mem](
    D, P, P_sol, warp_segments)
  return P


# -----------------------------------------------------------------------------