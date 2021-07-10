from __future__ import annotations

from abc import ABCMeta, abstractmethod

import warnings

import numpy as np
import cupy as cp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted 

from . import utils
from .utils import ArrayOnCPU, ArrayOnGPU, ArrayOnCPUOrGPU, RandomStateOrSeed

from typing import Optional, Tuple


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class RandomProjection(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Abstract base class for random projections.
    
    The idea is that random projections will be used for computing low-rank signature kernels, where one of the crucial steps is computing the outer product of
    two feature arrays, say X and Z. Then, the naive thing is to first compute the outer product and then sketch the resulting feature array via a random projection,
    proj(outer(X, Z)). This is inefficient as the feature dimension of outer(X, Z) can get very high. It can lead to substantial computational savings to use
    projections which can sketch the outer product without ever directly computing it. For this purpose, several of the methods can take either X as a singular
    argument or both X and Z, which are interpreted respectively as either sketching the array X or directly sketching the array outer(X, Z) in some smart way.
        
    Any classes deriving from this at the very least should implement the following methods:
        _make_projection_components(X : ArrayOnGPU, Z : Optional[ArrayOnGPU]): this (void) method should initialize dependent variables (except n_features_)
        _project_features(X : ArrayOnGPU): this method should compute and return the projection applied to input data X
        _project_outer_prod(X : ArrayOnGPU, Z : ArrayOnGPU): this method should compute and return the projection applied to the array outer(X, Z)
    
    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    
    def __init__(self, n_components : int = 100, random_state : Optional[RandomStateOrSeed] = None) -> None:
        """Initializer for RandomProjetion base class.
        
        Derived classes with overridden initializers should provide a call to this. 

        Args:
            n_components (int, optional): The number of projection components to use. Defaults to 100.
            random_state (Optional[RandomStateOrSeed], optional): Random state or seed value for reproducibility. Defaults to None.
        """
        self.n_components = utils.check_positive_value(n_components, 'n_components')
        self.random_state = random_state
        
    def _check_batch_dim(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        """Base method to check the batch dimensions (all dimensions except the last) of the arrays X and Z match.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.

        Raises:
            ValueError: If the batch dimensions do not match then raises an error.
        """
        if Z is not None and X.shape[:-1] != Z.shape[:-1]:
            raise ValueError(f'The arrays X and Z have different batch dimensions. ({X.shape[:-1]} != {Z.shape[:-1]})')
    
    def _check_n_features(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, reset : bool = False) -> None:
        """Base method to check or reset the n_features dimension in the data, which is the dimension along the last axis. 
        
        Note that if Z is specified then the number of features is equal to (number of features in X) x (the number of features in Z). 

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array.
            reset (bool, optional): Whether to reset an already saved n_features_ parameter. Defaults to False.. Defaults to False.

        Raises:
            ValueError: If not reset and n_features_ was set before and the current number of features does not match it, then it raises an error.
        """
        n_features = X.shape[-1]
        if Z is not None:
            n_features *= Z.shape[-1]
        if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
            self.n_features_ = n_features
        elif n_features != self.n_features_:
            raise ValueError(f'Received data with a different number of features than at fit time. ({n_features} != {self.n_features_})')
        
    def _validate_data(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, reset : bool = False) -> Tuple[ArrayOnCPUOrGPU, Optional[ArrayOnCPUOrGPU]]:
        """Base method for validating the input data. 

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
            reset (bool, optional): Whether to reset already saved parameters. Defaults to False.

        Returns:
            Tuple[ArrayOnCPUOrGPU, Optional[ArrayOnCPUOrGPU]]: returns the input data(s) as a NumPy or CuPy array without any changes.
        """
        self._check_batch_dim(X, Z=Z)
        self._check_n_features(X, Z=Z, reset=reset)
        return X, Z
    
    def _check_n_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        """Check if the number of projection components is less than the number of features in the data.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
        """
        if not hasattr(self, 'n_features_') or self.n_features_ is None:
            warnings.warn('Components could not be checked, because the number of features are as of yet unknown.')
        elif self.n_components > self.n_features_:
            warnings.warn(f'n_features < n_components ({self.n_features_} < {self.n_components}). The dimensionality of the problem will not be reduced.')
        self.n_components_ = self.n_components
    
    @abstractmethod
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        """Abstract method to initialize data dependent variables for computing the projection.
        
        This (void) method is meant to be called from within the fit method and it should initialize all data dependent variables
        except for the n_features_ parameter which is initialized within _validate_data.
        
        Warning: This method should not be called by the user, use instead the fit method.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
        """
        pass
            
    def fit(self, X : ArrayOnCPUOrGPU, y : Optional[ArrayOnCPUOrGPU] = None, Z : Optional[ArrayOnCPUOrGPU] = None) -> RandomProjection:
        """Validates the input data and initializes all dependent variables.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            y (Optional[ArrayOnCPUOrGPU], optional): Only provided for consistency with the sklearn API. Defaults to None.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.

        Returns:
            RandomProjection: A fitted RandomProjection object.
        """
        X, Z = self._validate_data(X, Z=Z, reset=True)
        self._check_n_components(X, Z=Z)
        self._make_projection_components(X, Z=Z)
        return self
    
    @abstractmethod
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Abstract method for computing and returning the projection for input data X.

        Warning: This method should not be called by the user, use instead the transform method.

        Args:
            X (ArrayOnGPU): Input data as a CuPy array.

        Returns:
            ArrayOnGPU: Sketch of the input data, X, as a CuPy array.
        """
        pass
        
    @abstractmethod
    def _project_outer_prod(self, X : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        """Abstract method for computing and returning the projection for input data outer(X, Z).

        Args:
            X (ArrayOnGPU): First input data as a CuPy array.
            Z (ArrayOnGPU): Second input data as a CuPy array.
        Returns:
            ArrayOnGPU: Sketch of the input data, outer(X, Z), as a CuPy array.
        """
        pass
        
    def _project(self, X : ArrayOnGPU, Z : Optional[ArrayOnGPU] = None) -> ArrayOnGPU:
        """Wrapper method for computing and returning the projection for input data X or outer(X, Z).

        Args:
            X (ArrayOnGPU): First input data as a CuPy array.
            Z (Optional[ArrayOnGPU], optional): Second input data as a CuPy array.
        Returns:
            ArrayOnGPU: Sketch of the input data, X or outer(X, Z), as a CuPy array.
        """
        if Z is None:
            return self._project_features(X)
        else:
            return self._project_outer_prod(X, Z)
        
    def transform(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, return_on_gpu : bool = False) -> ArrayOnCPUOrGPU:
        """Validates the input data and sketches it using randomized projections.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
            return_on_gpu (bool, optional): Whether to return the output on GPU (as CuPy) or on CPU (as NumPy).

        Returns:
            ArrayOnCPUOrGPU: Sketch of the input data, X or outer(X, Z), as a NumPy or CuPy array.
        """
        check_is_fitted(self)
        X, Z = self._validate_data(X, Z=Z)
        X = cp.asarray(X)
        Z = cp.asarray(Z) if Z is not None else None
        proj = self._project(X, Z=Z)
        if not return_on_gpu:
            proj = cp.asnumpy(proj)
        return proj
        
    def __call__(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, return_on_gpu : bool = False) -> ArrayOnCPUOrGPU:
        """Wrapper method to make the class work as a callable object.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
            return_on_gpu (bool, optional): Whether to return the output on GPU (as CuPy) or on CPU (as NumPy).

        Returns:
            ArrayOnCPUOrGPU: Sketch of the input data, X or outer(X, Z), as a NumPy or CuPy array.
        """
        return self.transform(X, Z=Z, return_on_gpu=return_on_gpu)
            
    def fit_transform(self, X : ArrayOnCPUOrGPU, y : Optional[ArrayOnCPUOrGPU] = None, Z : Optional[ArrayOnCPUOrGPU] = None,
                      return_on_gpu : bool = False, **fit_params) -> ArrayOnCPUOrGPU:
        return self.fit(X, y=y, Z=Z, **fit_params).transform(X, Z=Z, return_on_gpu=return_on_gpu)
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class GaussianRandomProjection(RandomProjection):
    """Class for computing vanilla Gaussian Random Projections.

    Reference:
        * Bingham, E. and Mannila, H.
          "Random projection in dimensionality reduction: applications to image and text data",
          Proceedings of the seventh ACM SIGKDD international conference on Knowledge discovery and data mining 2001
        * Dasgupta, S.
          "Experiments with random projection."
          Proceedings of the Sixteenth conference on Uncertainty in artificial intelligence 2000
    """
    
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        """Initializes data dependent variables for a Gaussian RP.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
        """
        random_state = utils.check_random_state(self.random_state)
        self.components_ = random_state.normal(size=(self.n_components_, self.n_features_))
        self.scaling_ = 1. / cp.sqrt(self.n_components_)
        
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns a Gaussian RP for input data X.

        Args:
            X (ArrayOnGPU): Input data as a CuPy array.

        Returns:
            ArrayOnGPU: Sketch of the input data, X, as a CuPy array.
        """
        return self.scaling_ * utils.matrix_mult(X.reshape((-1, self.n_features_)), self.components_, transpose_Y=True).reshape(X.shape[:-1] + (self.n_components_,))
        
    def _project_outer_prod(self, X : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns a Gaussian RP for input data outer(X, Z).

        Args:
            X (ArrayOnGPU): First input data as a CuPy array.
            Z (ArrayOnGPU): Second input data as a CuPy array.
        Returns:
            ArrayOnGPU: Sketch of the input data, outer(X, Z), as a CuPy array.
        """
        return self._project_features(utils.outer_prod(X, Z))
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SubsamplingRandomProjection(RandomProjection):
    """Class for computing Subsampling-based Random Projections."""
    
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        """Initializes data dependent variables for a Subsampling RP.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
        """
        random_state = utils.check_random_state(self.random_state)
        self.sampled_idx_ = random_state.choice(self.n_features_, size=self.n_components_, replace=False)
        self.scaling_ = cp.sqrt(self.n_features_ / self.n_components_)
        
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns a Subsampling RP for input data X.

        Args:
            X (ArrayOnGPU): Input data as a CuPy array.

        Returns:
            ArrayOnGPU: Sketch of the input data, X, as a CuPy array.
        """
        return self.scaling_ * cp.take(X, self.sampled_idx_, axis=-1)
    
    def _project_outer_prod(self, X : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns a Subsampling RP for input data outer(X, Z).

        Args:
            X (ArrayOnGPU): First input data as a CuPy array.
            Z (ArrayOnGPU): Second input data as a CuPy array.
        Returns:
            ArrayOnGPU: Sketch of the input data, outer(X, Z), as a CuPy array.
        """
        X_proj, Z_proj = utils.subsample_outer_prod_comps(X, Z, self.sampled_idx_)
        return self.scaling_ * X_proj * Z_proj
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class VerySparseRandomProjection(RandomProjection):
    """Class for computing Very Sparse Random Projections.
    
    Reference:
        * Li, P., Hastie, T.J. and Church, K.W.
          "Very sparse random projections"
          Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining 2006
    """
    
    def _check_sparsity_mode(self, sparsity : str) -> None:
        """Checks whether the sparsity argument falls within the accepted falues.

        Args:
            sparsity (str): Possible values are \'sqrt\' or \'log\'

        Raises:
            ValueError: if sparsity is not among the accepted values then raises an error
        """
        if sparsity not in ['sqrt', 'log']:
            raise ValueError(f'Unknown sparsity mode ({sparsity}). Possible values are [\'sqrt\', \'log\'].')
        self.sparsity = sparsity
    
    def __init__(self, n_components : int = 100, sparsity : str = 'log', random_state : Optional[RandomStateOrSeed] = None) -> None:
        """Initializer for VerySparseRandomProjection class.
        
        Args:
            n_components (int, optional): The number of projection components to use. Defaults to 100.
            sparsity (str, optional): The magnitude of sparsity. Possible values are \'sqrt\' or \'log\'. Defaults to 'log'.
            random_state (Optional[RandomStateOrSeed], optional): Random state or seed value for reproducibility. Defaults to None.
        """
        self._check_sparsity_mode(sparsity)
        super().__init__(n_components=n_components, random_state=random_state)
    
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        """Initializes data dependent variables for a Very Sparse RP.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
        """
        random_state = utils.check_random_state(self.random_state)
        if self.sparsity == 'log':
            prob_nonzero = cp.log(self.n_features_) / self.n_features_
        elif self.sparsity == 'sqrt':
            prob_nonzero = 1. / cp.sqrt(self.n_features_)
        components_full = utils.draw_rademacher_matrix((self.n_components_, self.n_features_), random_state=random_state) \
                          * utils.draw_bernoulli_matrix((self.n_components_, self.n_features_), prob_nonzero, random_state=random_state)
        self.sampled_idx_ = cp.where(cp.any(utils.robust_nonzero(components_full), axis=0))[0]
        self.n_sampled_ = self.sampled_idx_.shape[0]
        self.components_ = cp.squeeze(cp.take(components_full, self.sampled_idx_, axis=1))
        self.scaling_ = cp.sqrt(1. / (prob_nonzero * self.n_components_))
        
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns a Very Sparse RP for input data X.

        Args:
            X (ArrayOnGPU): Input data as a CuPy array.

        Returns:
            ArrayOnGPU: Sketch of the input data, X, as a CuPy array.
        """
        return self.scaling_ * utils.matrix_mult(
            cp.take(X, self.sampled_idx_, axis=-1).reshape((-1, self.n_sampled_)), self.components_, transpose_Y=True).reshape(X.shape[:-1] + (-1,))
    
    def _project_outer_prod(self, X : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns a Very Sparse RP for input data outer(X, Z).

        Args:
            X (ArrayOnGPU): First input data as a CuPy array.
            Z (ArrayOnGPU): Second input data as a CuPy array.
        Returns:
            ArrayOnGPU: Sketch of the input data, outer(X, Z), as a CuPy array.
        """
        X_proj, Z_proj = utils.subsample_outer_prod_comps(X, Z, self.sampled_idx_)
        return self.scaling_ * utils.matrix_mult(
            cp.reshape(X_proj * Z_proj, (-1, self.n_sampled_)), self.components_, transpose_Y=True).reshape(X_proj.shape[:-1] + (-1,))
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class CountSketchRandomProjection(RandomProjection):
    """Class for computing Count Sketch Random Projections.
    
    Reference:
        * Charikar, M, Chen, K. and Farach-Colton, M.
          "Finding frequent items in data streams"
          International Colloquium on Automata, Languages, and Programming 2002
        * Pham, M. and Pagh, R.
          "Fast and scalable polynomial kernels via explicit feature maps"
          Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining 2013
    """ 
    def _check_n_features(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, reset : bool = False) -> None:
        """Custom method to check or reset the n_features dimension in the data, which is the dimension along the last axis. 
        
        Note that if Z is specified then the number of features is equal to number of features in Z, otherwise to the number of features in X. 

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array.
            reset (bool, optional): Whether to reset an already saved n_features_ parameter. Defaults to False.. Defaults to False.

        Raises:
            ValueError: If not reset and n_features_ was set before and the current number of features does not match it, then it raises an error.
        """
        n_features = Z.shape[-1] if Z is not None else X.shape[-1]
        if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
            self.n_features_ = n_features
        elif n_features != self.n_features_:
            raise ValueError(f'Received data with a different number of features than at fit time. ({n_features} != {self.n_features_})')
    
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        """Initializes data dependent variables for a Count Sketch RP.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
        """
        random_state = utils.check_random_state(self.random_state)
        self.hash_index_ = random_state.randint(self.n_components_, size=(self.n_features_))
        self.hash_bit_ = utils.draw_rademacher_matrix((self.n_features_,), random_state=random_state) 
        
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns a Count Sketch RP for input data X.

        Args:
            X (ArrayOnGPU): Input data as a CuPy array.

        Returns:
            ArrayOnGPU: Sketch of the input data, X, as a CuPy array.
        """
        return utils.compute_count_sketch(X, self.hash_index_, self.hash_bit_, n_components=self.n_components_)
    
    def _project_outer_prod(self, X_count_sketch : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns a Count Sketch RP for input data outer(X, Z) via FFT for polynomial multiplication.

        Args:
            X (ArrayOnGPU): First input data as a CuPy array.
            Z (ArrayOnGPU): Second input data as a CuPy array.
        Returns:
            ArrayOnGPU: Sketch of the input data, outer(X, Z), as a CuPy array.
        """
        Z_count_sketch = utils.compute_count_sketch(Z, self.hash_index_, self.hash_bit_, n_components=self.n_components_)
        return utils.convolve_count_sketches(X_count_sketch, Z_count_sketch)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class TensorizedRandomProjection(RandomProjection):
    """Tensorized Random Projection (in the CP format) exclusively for use within the Low-Rank Signature Algorithm.
    
    This function is meant to be called recursively for each consecutive multiplicative term in the CP decomposition of a tensor.
    For example, if a tensor is given as 
        X = \sum_i x_i1 \otimes x_i2 \otimes \dots \otimes x_im,
    then the desired low-rank projection of the tensor X can be achieved by
        X_proj = \sum_i TRP_m(...(TRP2(TRP1(x_i1), x_i2), ...), x_im)
    The reason for this is that this object is meant to be used within the low-rank signature algorithms, which have such recursive structure.
    
    Reference:
        * Sun, Y., Guo, Y., Tropp, J.A. and Udell, M.
          "Tensor random projection for low memory dimension reduction"
          arXiv preprint arXiv:2105.00105 (2021).
        * Rakhshan, B. and Rabusseau, G.
          "Tensorized random projections"
          International Conference on Artificial Intelligence and Statistics 2020
    
    Warning: this is not intended to be used as a standalone random projection.
    Specifically, when used as a standalone RP on a feature matrix, it works analogously to vanilla Gaussian RP.
    
    """
    def __init__(self, n_components : int = 100, rank : int = 10, random_state : Optional[RandomStateOrSeed] = None) -> None:
        """Initializer for the RBFFourierFeatures class.
        Args:
            n_components (int, optional): The number of projection components to use. Defaults to 100.
            rank (int, optional): The rank of projection tensors. Defaults to 10.
            random_state (Optional[RandomStateOrSeed], optional): Random state or seed value for reproducibility. Defaults to None.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.rank = utils.check_positive_value(rank, 'rank')

    def _check_n_features(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None, reset : bool = False) -> None:
        """Custom method to check or reset the n_features dimension in the data, which is the dimension along the last axis. 
        
        Note that if Z is specified then the number of features is equal to number of features in Z, otherwise to the number of features in X. 
        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array.
            reset (bool, optional): Whether to reset an already saved n_features_ parameter. Defaults to False.. Defaults to False.
        Raises:
            ValueError: If not reset and n_features_ was set before and the current number of features does not match it, then it raises an error.
        """
        n_features = Z.shape[-1] if Z is not None else X.shape[-1]
        if reset or not hasattr(self, 'n_features_') or self.n_features_ is None:
            self.n_features_ = n_features
        
    def _check_n_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        """Custom method to set the "effective" number of components for the Tensorized RP..

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
        """
        self.n_components_ = self.n_components * self.rank
    
    def _make_projection_components(self, X : ArrayOnCPUOrGPU, Z : Optional[ArrayOnCPUOrGPU] = None) -> None:
        """Initializes data dependent variables for a Tensorized RP.

        Args:
            X (ArrayOnCPUOrGPU): First input data as a NumPy or CuPy array.
            Z (Optional[ArrayOnCPUOrGPU], optional): Second input data as a NumPy or CuPy array. Defaults to None.
        """
        random_state = utils.check_random_state(self.random_state)
        self.components_ = random_state.normal(size=(self.n_features_, self.n_components_))
        self.scaling_ = 1. / cp.sqrt(self.n_components_) if Z is None else 1.
        
    def _project_features(self, X : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns the projection of the first multiplicate term in the CP decomposition.
        
        Warning: For a single feature matrix X, the Tensorized RP is simply analogous to a vanilla Gaussian RP
        with simply an (n_components * rank) number of components.

        Args:
            X (ArrayOnGPU): Input data as a CuPy array.

        Returns:
            ArrayOnGPU: Sketch of the input data, X, as a CuPy array.
        """
        return self.scaling_ * utils.matrix_mult(X.reshape([-1, self.n_features_]), self.components_).reshape(X.shape[:-1] + (-1,))
        
    def _project_outer_prod(self, X : ArrayOnGPU, Z : ArrayOnGPU) -> ArrayOnGPU:
        """Computes and returns the projection for the next multiplicative term in the CP decomposition. 
        
        This function assumes that X is already projected, and hence, only projects Z and multiplies the two together.

        Args:
            X (ArrayOnGPU): First input data as a CuPy array.
            Z (ArrayOnGPU): Second input data as a CuPy array.
        Returns:
            ArrayOnGPU: Sketch of the input data, outer(X, Z), as a CuPy array.
        """
        return self.scaling_ * X * utils.matrix_mult(Z.reshape([-1, self.n_features_]), self.components_).reshape(Z.shape[:-1] + (-1,))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------