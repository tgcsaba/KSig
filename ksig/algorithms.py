import numpy as np
import cupy as cp

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def multi_cumsum(M, exclusive=False, axis=-1):
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

def signature_kern(M, n_levels, order=-1, difference=True, return_levels=False):
    """Wrapper for signature kernel algorithms. If order==1 then it uses a simplified, more efficient implementation."""
    order = n_levels if order==-1 else order
    if order==1:
        return signature_kern_first_order(M, n_levels, difference=difference, return_levels=return_levels)
    else:
        return signature_kern_higher_order(M, n_levels, order=order, difference=difference, return_levels=return_levels)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_first_order(M, n_levels, difference=True, return_levels=False):
    """
    Computes the (truncated) signature kernel matrix with first-order embedding into the tensor algebra. 
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

def signature_kern_higher_order(M, n_levels, order=-1, difference=True, return_levels=False):
    """
    Computes the (truncated) signature kernel matrix with higher-order embedding into the tensor algebra. 
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

def signature_kern_first_order_low_rank(U, n_levels, difference=True, return_levels=False, projections=None):
    """
    Computes a low-rank feature approximation corresponding to the (truncated) signature kernel with first-order embedding into the tensor algebra.
    """
    
    if difference:
        U = cp.diff(U, axis=1)
        
    n_X, l_X, d = U.shape
    P = cp.ones((n_X), dtype=U.dtype)
    
    if projections is not None:
        U = projections[0](U)
    
    if return_levels:
        P = [P, cp.sum(U, axis=1)]
    else:
        P = cp.concatenate((P, cp.sum(U, axis=1)), axis=-1)
    
    R = cp.copy(U)
    for i in range(1, n_levels):
        R = multi_cumsum(R, axis=1, exclusive=True)
        if projections is None:
            R = cp.reshape(U[..., :, None] * R[..., None, :] (n_X, l_X, -1))
        else:
            R = projections[i](U, R)
        if return_levels:
            P.append(cp.sum(R, axis=1))
        else:
            P = cp.concatenate((P, cp.sum(R, axis=1)), axis=-1)
    return P

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_higher_order_low_rank(U, n_levels, order=-1, difference=True, return_levels=False, projections=None):
    """
    Computes a low-rank feature approximation corresponding to the (truncated) signature kernel with higher-order embedding into the tensor algebra.
    """
    
    if difference:
        U = cp.diff(U, axis=1)
        
    n_X, l_X, d = U.shape
    P = cp.ones((n_X), dtype=U.dtype)
    
    if projections is not None:
        U = projections[0](U)
    
    if return_levels:
        P = [P, cp.sum(U, axis=1)]
    else:
        P = cp.concatenate((P, cp.sum(U, axis=1)), axis=-1)
    
    R = cp.copy(U)
    for i in range(1, n_levels):
        p = min(i+1, order)
        n_components = projections[i].n_components_ if projections is not None else d**(i+1)
        R_next = np.empty((p, n_X, l_X, n_components))
        
        Q = multi_cumsum(cp.sum(R, axis=0), axis=1, exclusive=True)
        if projections is None:
            R_next[0] = cp.reshape(U[..., :, None] * Q[..., None, :] (n_X, l_X, -1))
        else:
            R_next[0] = projections[i](U, Q)
        for r in range(1, d):
            if projections is None:
                R_next[r] = 1. / (r+1) * cp.reshape(U[..., :, None] * R[r-1, ..., None, :], (n_X, l_X, n_components))
            else:
                R_next[r] = 1. / (r+1) * projections[i](U, R_next[r-1])
        R = R_next
        if return_levels:
            P.append(cp.sum(R, axis=1))
        else:
            P = cp.concatenate((P, cp.sum(R, axis=1)), axis=-1)
    return P

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------