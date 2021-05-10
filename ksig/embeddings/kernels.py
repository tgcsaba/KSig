from abc import ABCMeta, abstractmethod

import numpy as np
import cupy as cp

from sklearn.base import BaseEstimator, TransformerMixin

from .. import utils

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class Kernel(BaseEstimator, metaclass=ABCMeta):
    """Base class for kernels."""
        
    def fit(X, y=None):
        pass
        
    @abstractmethod
    def K(self, X, Y=None):
        pass
    
    @abstractmethod
    def Kdiag(self, X):
        pass

    def __call__(self, X, Y=None, diag=False):
        if diag:
            return self.Kdiag(X)
        else:
            return self.K(X, Y)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class LinearKernel(Kernel):
    """Class for linear (static) kernel."""
    
    def __init__(self, sigma=1.0):
        self.sigma = utils.check_positive_float(sigma, 'sigma')
        
    def K(self, X, Y=None):
        return self.sigma**2 * utils.matrix_mult(X, Y, transpose_Y=True)
        
    def Kdiag(self, X):
        return self.sigma**2 * utils.squared_norm(X, axis=-1)

class PolynomialKernel(Kernel):
    """Class for polynomial (static) kernel."""
    def __init__(self, sigma=1.0, degree=3.0, gamma=1.0):
        self.sigma = utils.check_positive_float(sigma, 'sigma')
        self.degree = utils.check_positive_float(degree, 'degree')
        self.gamma = float(gamma)
    
    def K(self, X, Y=None):
        return self.sigma**2 * cp.power(utils.matrix_mult(X, Y, transpose_Y=True) + self.gamma, self.degree)
        
    def Kdiag(self, X):
        return self.sigma**2 * cp.power(utils.squared_norm(X, axis=-1) + self.gamma, self.degree)
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class StationaryKernel(Kernel):
    """Base class for stationary (static) kernels."""
    
    def __init__(self, sigma=1.0, lengthscale=1.0):
        self.sigma = utils.check_positive_float(sigma, 'sigma')
        self.lengthscale = utils.check_positive_float(lengthscale, 'lengthscale')
        
    def Kdiag(self, X):
        return cp.full((X.shape[0],), self.sigma**2)
        
class RBFKernel(StationaryKernel):
    """Radial Basis Function aka Gauss (static) kernel ."""
    
    def K(self, X, Y=None):
        D2_scaled = utils.squared_euclid_dist(X, Y) / self.lengthscale**2
        return self.sigma**2 * cp.exp(-D2_scaled)
                          
class Matern12Kernel(StationaryKernel):
    """Matern12 (static) kernel ."""
        
    def K(self, X, Y=None):
        D_scaled = utils.euclid_dist(X, Y) / self.lengthscale
        return self.sigma**2 * cp.exp(-D_scaled)
        
class Matern32Kernel(StationaryKernel):
    """Matern32 (static) kernel ."""
        
    def K(self, X, Y=None):
        sqrt3 = cp.sqrt(3.)
        D_scaled = sqrt3 * utils.euclid_dist(X, Y) / self.lengthscale 
        return self.sigma**2 * (1. + D_scaled) * cp.exp(-D_scaled)
        
class Matern52Kernel(StationaryKernel):
    """Matern52 (static) kernel ."""
        
    def K(self, X, Y=None):
        D2_scaled = 5 * utils.squared_euclid_dist(X, Y) / self.lengthscale**2
        D_scaled = utils.robust_sqrt(D2_scaled) 
        return self.sigma**2 * (1. + D_scaled + D2_scaled / 3.) * cp.exp(-D_scaled)

class RationalQuadraticKernel(StationaryKernel):
    """Rational Quadratic (static) kernel ."""
    
    def __init__(self, sigma=1.0, lengthscale=1.0, alpha=1.0):
        self.sigma = utils.check_positive_float(sigma, 'sigma')
        self.lengthscale = utils.check_positive_float(lengthscale, 'lengthscale')
        self.alpha = utils.check_positive_float(alpha, 'alpha')
        
    def K(self, X, Y=None):
        D2_scaled = utils.squared_euclid_dist(X, Y) / (2 * self.alpha * self.lengthscale**2)
        return self.sigma**2 * cp.power((1 + D2_scaled), -self.alpha)
          
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------