import cupy as cp
import numbers

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (Type) Checkers
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_positive_float(scalar, name):
    if not isinstance(scalar ,float) or scalar <= 0.:
        raise ValueError(f'The parameter \'{name}\' should be a positive float.')
    return scalar

def check_positive_int(scalar, name):
    if not isinstance(scalar, int) or scalar <= 0:
        raise ValueError(f'The parameter \'{name}\' should be a positive int.')
    return scalar

def check_bool(scalar, name):
    if isinstance(scalar, int) and (scalar==0 or scalar==1):
        scalar = bool(scalar)
    if not isinstance(scalar, bool):
        raise ValueError(f'The parameter \'{name}\' should be a boolean.')
    return scalar
    
def check_random_state(seed):
    if seed is None or isinstance(seed, numbers.Integral):
        return cp.random.RandomState(seed)
    elif isinstance(seed, cp.random.RandomState):
        return seed
    raise ValueError(f'{seed} cannot be used to seed a cupy.random.RandomState instance')
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (Batched) computations
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def matrix_diag(A):
    """Takes as input an array of shape (..., d, d) and returns the diagonals along the last two axes with output shape (..., d)."""
    return cp.einsum('...ii->...i', A)

def matrix_mult(X, Y=None, transpose_X=False, transpose_Y=False):
    subscript_X = '...ji' if transpose_X else '...ij'
    subscript_Y = '...kj' if transpose_Y else '...jk'
    return cp.einsum(f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)
    
def squared_norm(X, axis=-1):
    return cp.sum(cp.square(X), axis=axis)

def squared_euclid_dist(X, Y=None):
    X_n2 = squared_norm(X)
    if Y is None:
        D2 = (X_n2[..., :, None] + X_n2[..., None, :]) - 2 * matrix_mult(X, X, transpose_Y=True)
    else:
        Y_n2 = squared_norm(Y, axis=-1)
        D2 = (X_n2[..., :, None] + Y_n2[..., None, :]) - 2 * matrix_mult(X, Y, transpose_Y=True)
    return D2
    
def outer_prod(X, Y):
    return cp.reshape(X[..., :, None] * Y[..., None, :], X.shape[:-1] + (-1,))
    
def robust_sqrt(X):
    return cp.sqrt(cp.maximum(X, 1e-36))
    
def euclid_dist(self, X, Y=None):
    return robust_sqrt(squared_euclid_dist(X, Y))
    
def robust_nonzero(X):
    return cp.abs(X) > 1e-18
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Probability stuff
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def draw_rademacher_matrix(shape, random_state=None):
    random_state = check_random_state(random_state)
    return cp.where(random_state.uniform(size=shape) < 0.5, cp.ones(shape), -cp.ones(shape))

def draw_bernoulli_matrix(shape, prob, random_state=None):
    random_state = check_random_state(random_state)
    return cp.where(random_state.uniform(size=shape) < prob, cp.ones(shape), cp.zeros(shape))
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------