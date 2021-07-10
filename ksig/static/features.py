from __future__ import annotations

from abc import ABCMeta, abstractmethod

import warnings

import numpy as np
import cupy as cp

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .kernels import Kernel

from .. import utils
from ..utils import ArrayOnCPU, ArrayOnGPU, ArrayOnCPUOrGPU, RandomStateOrSeed

from typing import Optional

class LowRankFeatures(Kernel, TransformerMixin, metaclass=ABCMeta):
    """Abstract base class for low-rank feature maps.

    Any classes deriving from this at the very least should implement the following methods:
        _make_feature_components(X : ArrayOnGPU): this (void) method should initialize dependent variables (except n_features_)
        _compute_features(X : ArrayOnGPU): this method should compute and return the finite dimensional embedding for input data X
    
    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    
    def __init__(self, n_components : int = 100, random_state : Optional[RandomStateOrSeed] = None) -> None:
        """Initializer for LowRankFeatures base class. 
        
        Derived classes with overridden initializers should provide a call to this. 

        Args:
            n_components (int, optional): The number of feature components to use in the embedding. Defaults to 100.
            random_state (Optional[RandomStateOrSeed], optional): Random state or seed value for reproducibility. Defaults to None.
        """
        self.n_components = utils.check_positive_value(n_components, 'n_components')
        self.random_state = random_state
        
    def _check_n_features(self, X : ArrayOnCPUOrGPU, reset : bool = False) -> None:
        """Base method to check or reset the n_features dimension in the data, which is the dimension along the last axis. 
        
        This function is meant to be called from within the _validate_data method and it may be overridden by some subclasses. 

        Args:
            X (ArrayOnCPUOrGPU): Input data as a NumPy or CuPy array.
            reset (bool, optional): Whether to reset an already saved n_features_ parameter. Defaults to False.

        Raises:
            ValueError: If not reset and n_features_ was set before and the current number of features does not match it, then it raises an error.
        """
        n_features = X.shape[-1]
        if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
            self.n_features_ = n_features
        elif n_features != self.n_features_:
            raise ValueError(f'Received data with a different number of features than at fit time. ({n_features} != {self.n_features_})')
        
    def _validate_data(self, X : ArrayOnCPUOrGPU, reset : bool = False) -> ArrayOnCPUOrGPU:
        """Base method for validating the input data. 
        
        Args:
            X (ArrayOnCPUOrGPU): Input data as a NumPy or CuPy array.
            reset (bool, optional): Whether to reset already saved parameters. Defaults to False.

        Returns:
            ArrayOnCPUOrGPU: returns the input data as a NumPy or CuPy array without any changes.
        """
        self._check_n_features(X, reset=reset)
        return X
    
    @abstractmethod
    def _make_feature_components(self, X : ArrayOnCPUOrGPU) -> None:
        """Abstract method to initialize data dependent variables for computing the embedding.

        This (void) method is meant to be called from within the fit method and it should initialize all data dependent variables
        except for the n_features_ parameter which is initialized within _validate_data.

        Warning: This method should not be called by the user, use instead the fit method.

        Args:
            X (ArrayOnCPUOrGPU): Input data as a NumPy or CuPy array.
        """
        pass
        
    def fit(self, X: ArrayOnCPUOrGPU, y : Optional[ArrayOnCPUOrGPU] = None) -> LowRankFeatures:
        """Validates the input data and initializes all dependent variables.

        Args:
            X (ArrayOnCPUOrGPU): Input data as a NumPy or CuPy array.
            y (Optional[ArrayOnCPUOrGPU], optional): Only provided for consistency with the sklearn API. Defaults to None.

        Returns:
            LowRankFeatures: A fitted LowRankFeatures object.
        """
        X = self._validate_data(X, reset=True)
        self._make_feature_components(X)
        return self
    
    @abstractmethod
    def _compute_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Abstract method for computing and returning the embedding for input data X.

        Warning: This method should not be called by the user, use instead the transform method.

        Args:
            X (ArrayOnGPU): Input data as a CuPy array.

        Returns:
            ArrayOnGPU: A feature set as a CuPy array.
        """
        pass
        
    def transform(self, X : ArrayOnCPUOrGPU, return_on_gpu : bool = False) -> ArrayOnCPUOrGPU:
        """Validates the input data and embeds it into the feature space of a low-rank kernel.

        This method checks if the class has been fitted, validates the data and moves it into  GPU memory (if it wasn't there to begin with),
        then computes the feature space embedding, and finally, depending on return_on_gpu it may move it back to base memory.  

        Args:
            X (ArrayOnCPUOrGPU): Input data as a NumPy or CuPy array.
            return_on_gpu (bool, optional): Whether to return the output on GPU (as CuPy) or on CPU (as NumPy).

        Returns:
            X_feat: (ArrayOnCPUOrGPU) A feature set as a NumPy or CuPy array.
        """
        check_is_fitted(self)
        X = self._validate_data(X)
        X = cp.asarray(X)
        X_feat = self._compute_features(X)
        if not return_on_gpu:
            X_feat = cp.asnumpy(X_feat)
        return X_feat
        
    def _K(self, X : ArrayOnGPU, Y : Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
        """Wrapper to compute the low-rank kernel matrix by taking inner products between feature vectors.

        The output is a kernel matrix between X and Y or X and X if Y is None.

        Warning: This method should not be called by the user, use instead the __call__ method via LowRankFeatures(X, Y).
         
        Args:
            X (ArrayOnGPU): First input data CuPy array.
            Y (Optional[ArrayOnGPU], optional): Second input data CuPy array. Defaults to None.

        Returns:
            ArrayOnGPU: The kernel matrix between X and Y or X and X as a CuPy array.
        """
        X_feat = self._compute_features(X)
        Y_feat = self._compute_features(Y) if Y is not None else None
        return utils.matrix_mult(X_feat, Y_feat, transpose_Y=True)
    
    def _Kdiag(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Wrapper to compute diagonal entries of the low-rank kernel matrix by taking the squared norm of feature vectors.

        Warning: This method should not be called by the user, use instead the __call__ method via LowRankFeatures(X, diag=True).

        Args:
            X (ArrayOnGPU): Input data CuPy array.

        Returns:
            ArrayOnGPU: The diagonal entries of the kernel matrix between X and X as a CuPy array.
        """
        X_feat = self._compute_features(X)
        return utils.squared_norm(X_feat, axis=-1)
        
    def __call__(self, X : ArrayOnCPUOrGPU, Y : Optional[ArrayOnCPUOrGPU] = None, diag : bool = False, return_on_gpu : bool = False) -> ArrayOnCPUOrGPU:
        """Wrapper function to make the class work as a callable for kernel computations by taking inner products between feature vectors.

        The 3 possible kinds of behaviour are:
            if diag then compute the diagonal entries of the kernel matrix between X and X
            if not diag and Y is not None then compute the kernel matrix between X and Y
            if not diag and Y is None then compute the kernel matrix between X and X 

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Y (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
            diag (bool, optional): Whether to compute only the diagonal entries (only if Y is None). Defaults to False.
            return_on_gpu (bool, optional): Whether to return the output on GPU (as CuPy) or on CPU (as NumPy).

        Returns:
            ArrayOnCPUOrGPU: A kernel matrix or the diagonal entries of a kernel matrix as a NumPy or CuPy array.
        """
        check_is_fitted(self)
        X = self._validate_data(X)
        if Y is not None:
            Y = self._validate_data(Y)
        return super().__call__(X, Y=Y, diag=diag, return_on_gpu=return_on_gpu)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        
class NystroemFeatures(LowRankFeatures):
    """Class for computing the Nystroem embedding for a static kernel.

    Reference:
        * Williams, C.K.I. and Seeger, M.
          "Using the Nystroem method to speed up kernel machines",
          Advances in neural information processing systems 2001
    """
    
    def __init__(self, static_kernel : Kernel, n_components : int = 100, random_state : Optional[RandomStateOrSeed] = None) -> None:
        """Initializer for the NystroemFeatures class.

        Args:
            static_kernel (Kernel): A static kernel object to use for computing the Nystroem features
            n_components (int, optional): The number of feature components to use in the embedding. Defaults to 100.
            random_state (Optional[RandomStateOrSeed], optional): Random state or seed value for reproducibility. Defaults to None.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.static_kernel = static_kernel
    
    def _make_feature_components(self, X : ArrayOnCPUOrGPU) -> None:
        """Initializes data dependent variables for the Nystroem embedding.

        Args:
            X (ArrayOnGPUOrCPU): Input data as a NumPy or CuPy array.
        """
        X = X.reshape([-1, self.n_features_])
        n_samples = X.shape[0]
        # number of components can't be larger than the number of input samples
        if self.n_components > n_samples:
            warnings.warn('n_samples <= n_components, hence n_components was set to n_samples, which results in inefficient evaluation of the full kernel.')
            self.n_components_ = n_samples
        else:
            self.n_components_ = self.n_components
        # subsample data
        random_state = utils.check_random_state(self.random_state)
        basis_inds = random_state.choice(n_samples, size=self.n_components_, replace=False)
        if isinstance(X, ArrayOnCPU):
            basis_inds = cp.asnumpy(basis_inds) # doing this seems to be the most efficient
        basis = cp.asarray(X[basis_inds])
        # compute and decompose the symmetric kernel matrix K
        basis_K = self.static_kernel(basis, return_on_gpu=True) # static_kernel moves only basis to GPU memory rather then the full X 
        S, U = cp.linalg.eigh(basis_K)
        nonzero_eigs_mask = utils.robust_nonzero(S)
        # compute and save K^(-1/2)
        self.normalization_ = U[..., :, nonzero_eigs_mask] / utils.robust_sqrt(S[..., nonzero_eigs_mask][..., None, :])
        self.components_ = basis
        self.component_idx_ = basis_inds
        
    def _compute_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Computes features for input data X using the Nystroem embedding.

        Args:
            X (ArrayOnGPU): Input data as a CuPy array.

        Returns:
            ArrayOnGPU: Nystroem features as a CuPy array.
        """
        embedded = self.static_kernel(X.reshape([-1, self.n_features_]), self.components_, return_on_gpu=True)
        return utils.matrix_mult(embedded, self.normalization_).reshape(X.shape[:-1] + (-1,))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class RBFFourierFeatures(LowRankFeatures):
    """Class for computing the Random Fourier Feature embedding for an RBF static kernel.
    
    Reference:
        * Rahimi, A. and Recht, B.
          "Random Features for Large-Scale Kernel Machines",
          Advances in neural information processing systems 2008      
    """
    
    def __init__(self, sigma : float = 1., lengthscale : float = 1., n_components : int = 100, random_state : Optional[RandomStateOrSeed] = None) -> None:
        """Initializer for the RBFFourierFeatures class.

        Args:
            sigma (float, optional): A multiplicative scaling factor applied to the features. Defaults to 1..
            lengthscale (float, optional): The lengthscale of the inputs (divides the input, x' = x / l for input x). Defaults to 1..
            n_components (int, optional): The number of feature components to use in the embedding. Defaults to 100.
            random_state (Optional[RandomStateOrSeed], optional): Random state or seed value for reproducibility. Defaults to None.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.lengthscale = utils.check_positive_value(lengthscale, 'lengthscale')
        
    def _make_feature_components(self, X : ArrayOnGPUOrCPU) -> None:
        """Initializes data dependent variables for the RFF embedding.

        Args:
            X (ArrayOnGPUOrCPU): Input data as a NumPy or CuPy array.
        """
        self.n_components_ = self.n_components
        random_state = utils.check_random_state(self.random_state)
        self.random_weights_ = (cp.sqrt(2 / self.lengthscale**2) * random_state.normal(size=(self.n_features_, self.n_components_)))
        self.random_offset_ = random_state.uniform(0, 2 * cp.pi, size=self.n_components_)
    
    def _compute_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Computes features for input data X using the Random Fourier Features embedding for an RBF static kernel.

        Args:
            X (ArrayOnGPU): Input data as a CuPy array.

        Returns:
            ArrayOnGPU: Nystroem features as a CuPy array.
        """
        projection = utils.matrix_mult(X.reshape([-1, self.n_features_]), self.random_weights_)
        projection += self.random_offset_[None, :]
        cp.cos(projection, projection)
        projection *= self.sigma * cp.sqrt(2.) / cp.sqrt(self.n_components)
        return projection.reshape(X.shape[:-1] + (-1,))
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------