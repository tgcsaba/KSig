import numpy as np
import cupy as cp

from typing import Optional, Union, List

from .projections import RandomProjection, TensorizedRandomProjection

from .utils import ArrayOnCPU, ArrayOnGPU, ArrayOnCPUOrGPU, RandomStateOrSeed

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def multi_cumsum(M : ArrayOnGPU, exclusive : bool = False, axis : int = -1) -> ArrayOnGPU:
    """Computes the exclusive cumulative sum along a given set of axes.

    Args:
        K (cp.ndarray): A matrix over which to compute the cumulative sum
        axis (int or iterable, optional): An axis or a collection of them. Defaults to -1 (the last axis).
    """
    
    ndim = M.ndim
    axis = [axis] if cp.isscalar(axis) else axis
    axis = [ndim+ax if ax < 0 else ax for ax in axis]
    
    # create slice for exclusive cumsum (slice off last element along given axis then pre-pad with zeros)
    if exclusive:
        slices = tuple(slice(-1) if ax in axis else slice(None) for ax in range(ndim))
        M = M[slices]
    
    # compute actual cumsums
    for ax in axis:
        M = cp.cumsum(M, axis=ax)
        
    # pre-pad with zeros along the given axis if exclusive cumsum
    if exclusive:
        pads = tuple((1, 0) if ax in axis else (0, 0) for ax in range(ndim))
        M = cp.pad(M, pads)
        
    return M 
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Signature algs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern(M : ArrayOnGPU, n_levels : int, order : int = -1, difference : bool = True, return_levels : bool = False) -> ArrayOnGPU:
    """Wrapper for signature kernel algorithms. If order==1 then it uses a simplified, more efficient implementation."""
    order = n_levels if order <= 0 or order >= n_levels else order
    if order==1:
        return signature_kern_first_order(M, n_levels, difference=difference, return_levels=return_levels)
    else:
        return signature_kern_higher_order(M, n_levels, order=order, difference=difference, return_levels=return_levels)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_first_order(M : ArrayOnGPU, n_levels : int, difference : bool = True, return_levels : bool = False) -> ArrayOnGPU:
    """
    Computes the signature kernel matrix with first-order embedding into the tensor algebra. 
    """
    
    if difference:
        M = cp.diff(cp.diff(M, axis=1), axis=-1)
    
    if M.ndim == 4:
        n_X, n_Y = M.shape[0], M.shape[2]
        K = cp.ones((n_X, n_Y), dtype=M.dtype)
    else:
        n_X = M.shape[0]
        K = cp.ones((n_X,), dtype=M.dtype)
        
    if return_levels:
        K = [K, cp.sum(M, axis=(1, -1))]    
    else:
        K += cp.sum(M, axis=(1, -1))
    
    R = cp.copy(M)
    for i in range(1, n_levels):
        R = M * multi_cumsum(R, exclusive=True, axis=(1, -1))
        if return_levels:
            K.append(cp.sum(R, axis=(1, -1)))
        else:
            K += cp.sum(R, axis=(1, -1))
        
    return cp.stack(K, axis=0) if return_levels else K
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_higher_order(M : ArrayOnGPU, n_levels : int, order : int, difference : bool = True, return_levels : bool = False) -> ArrayOnGPU:
    """
    Computes the signature kernel matrix with higher-order embedding into the tensor algebra. 
    """
    
    if difference:
        M = cp.diff(cp.diff(M, axis=1), axis=-1)
    
    if M.ndim == 4:
        n_X, n_Y = M.shape[0], M.shape[2]
        K = cp.ones((n_X, n_Y), dtype=M.dtype)
    else:
        n_X = M.shape[0]
        K = cp.ones((n_X,), dtype=M.dtype)
    
    if return_levels:
        K = [K, cp.sum(M, axis=(1, -1))]
    else:
        K += cp.sum(M, axis=(1, -1))
    
    R = cp.copy(M)[None, None, ...]
    for i in range(1, n_levels):
        d = min(i+1, order)
        R_next = cp.empty((d, d) + M.shape, dtype=M.dtype)
        R_next[0, 0] = M * multi_cumsum(cp.sum(R, axis=(0, 1)), exclusive=True, axis=(1, -1))
        for r in range(1, d):
            R_next[0, r] = 1./(r+1) * M * multi_cumsum(cp.sum(R[:, r-1], axis=0), exclusive=True, axis=1)
            R_next[r, 0] = 1./(r+1) * M * multi_cumsum(cp.sum(R[r-1, :], axis=0), exclusive=True, axis=-1)
            for s in range(1, d):
                R_next[r, s] = 1./((r+1)*(s+1)) * M * R[r-1, s-1]
        R = R_next
        if return_levels:
            K.append(cp.sum(R, axis=(0, 1, 3, -1)))
        else:
            K += cp.sum(R, axis=(0, 1, 3, -1))
        
    return cp.stack(K, axis=0) if return_levels else K
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Low-Rank algs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_low_rank(U : ArrayOnGPU, n_levels : int, order : int = -1, difference : bool = True, return_levels : bool = False,
                            projections : Optional[RandomProjection] = None) -> Union[List[ArrayOnGPU], ArrayOnGPU]:
    """Wrapper for low-rank signature kernel algs. If order==1 then it uses a simplified, more efficient implementation."""
    order = n_levels if order <= 0 or order >= n_levels else order
    if order==1:
        return signature_kern_first_order_low_rank(U, n_levels, difference=difference, return_levels=return_levels, projections=projections)
    else:
        return signature_kern_higher_order_low_rank(U, n_levels, order=order, difference=difference, return_levels=return_levels, projections=projections)
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_first_order_low_rank(U : ArrayOnGPU, n_levels : int, difference : bool = True, return_levels : bool = False,
                                        projections : Optional[RandomProjection] = None) -> Union[List[ArrayOnGPU], ArrayOnGPU]:
    """
    Computes a low-rank feature approximation corresponding to the signature kernel with first-order embedding into the tensor algebra.
    """
    
    if difference:
        U = cp.diff(U, axis=1)
        
    n_X, l_X, n_d = U.shape
    P = cp.ones((n_X, 1), dtype=U.dtype)
    
    R = projections[0](U, return_on_gpu=True) if projections is not None else cp.copy(U)
    R_sum = cp.sum(R.reshape([n_X, l_X, projections[0].n_components, projections[0].rank]), axis=(1, -1)) if projections is not None \
            and isinstance(projections[0], TensorizedRandomProjection) else cp.sum(R, axis=1)
    if return_levels:
        P = [P, R_sum]
    else:
        P = cp.concatenate((P, R_sum), axis=-1)
        
    for i in range(1, n_levels):
        R = multi_cumsum(R, axis=1, exclusive=True)
        if projections is None:
            R = cp.reshape(R[..., :, None] * U[..., None, :], (n_X, l_X, -1))
        else:
            R = projections[i](R, U, return_on_gpu=True)
        R_sum = cp.sum(R.reshape([n_X, l_X, projections[i].n_components, projections[i].rank]), axis=(1, -1)) if projections is not None \
                and isinstance(projections[i], TensorizedRandomProjection) else cp.sum(R, axis=1)
        if return_levels:
            P.append(R_sum)
        else:
            P = cp.concatenate((P, cp.sum(R, axis=1)), axis=-1)
    return P

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_higher_order_low_rank(U : ArrayOnGPU, n_levels : int, order : int = -1, difference : bool = True, return_levels : bool = False,
                                         projections : Optional[RandomProjection] = None) -> Union[List[ArrayOnGPU], ArrayOnGPU]:
    """
    Computes a low-rank feature approximation corresponding to the signature kernel with higher-order embedding into the tensor algebra.
    """
    
    if difference:
        U = cp.diff(U, axis=1)
        
    n_X, l_X, n_d = U.shape
    P = cp.ones((n_X, 1), dtype=U.dtype)
    
    R = projections[0](U, return_on_gpu=True) if projections is not None else cp.copy(U)
    R_sum = cp.sum(R.reshape([n_X, l_X, projections[0].n_components, projections[0].rank]), axis=(1, -1)) if projections is not None \
            and isinstance(projections[0], TensorizedRandomProjection) else cp.sum(R, axis=1)
    if return_levels:
        P = [P, R_sum]
    else:
        P = cp.concatenate((P, R_sum), axis=-1)
    
    R = R[None]
    for i in range(1, n_levels):
        d = min(i+1, order)
        n_components = projections[i].n_components_ if projections is not None else n_d**(i+1)
        R_next = cp.empty((d, n_X, l_X, n_components))
        
        Q = multi_cumsum(cp.sum(R, axis=0), axis=1, exclusive=True)
        if projections is None:
            R_next[0] = cp.reshape(Q[..., :, None] * U[..., None, :], (n_X, l_X, -1))
        else:
            R_next[0] = projections[i](Q, U, return_on_gpu=True)
        for r in range(1, d):
            if projections is None:
                R_next[r] = 1. / (r+1) * cp.reshape(R[r-1, ..., :, None] * U[..., None, :], (n_X, l_X, n_components))
            else:
                R_next[r] = 1. / (r+1) * projections[i](R[r-1], U, return_on_gpu=True)
        R = R_next
        R_sum = cp.sum(R.reshape([d, n_X, l_X, projections[i].n_components, projections[i].rank]), axis=(0, 2, -1)) if projections is not None \
                and isinstance(projections[i], TensorizedRandomProjection) else cp.sum(R, axis=(0, 2))
        if return_levels:
            P.append(R_sum)
        else:
            P = cp.concatenate((P, R_sum), axis=-1)
    return P

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------