"`SVC` with precomputed kernel."
import cupy as cp
import numpy as np

from ..utils import ArrayOnCPUOrGPU, ArrayOnCPU, ArrayOnGPU
from abc import ABCMeta, abstractmethod
from math import ceil
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from typing import Callable, Dict, Optional


_MAT_MUL_BSIZE = 1000


class PrecomputedSVCBase(BaseEstimator, ClassifierMixin, metaclass=ABCMeta):
  """Base class for precomputed SVC models.
  
  
  Deriving classes should implement the following methods:
    _get_svc_model: Constructs and returns the SVC model to use.
    _precompute_model_inputs: Precomputes the inputs fed to the SVC model.

  Warning: This class should not be used directly, only derived classes.
  """
  def __init__(self,
               kernel: Callable,
               svc_hparams: Dict = {},
               svc_grid: Optional[Dict] = None,
               cv: int = 3,
               n_jobs: int = -1,
               need_kernel_fit: bool = False,
               has_transform: bool = False,
               batch_size: Optional[int] = None,
               fit_samples: Optional[int] = None):
    """Initializer for `PrecomputedSVCBase`.

    Args:
      kernel: A callable for computing the kernel or feature matrix.
      svc_grid: Hyperparameter grid to cross-validate over, set to `None`
        for no cross-validation.
      svc_hparams: Additional hyperparameters for `SVC`.
      cv: Number of CV splits.
      n_jobs: Number of jobs for `GridSearchCV`.
      need_kernel_fit: Whether the kernel needs to be fitted to the data.
      has_transform: Whether the kernel has a feature transform.
      batch_size: If given, compute the kernel matrix in chunks of shape
        `[batch_size, batch_size]`, or in chunks of shape [batch_size, ...]
        if `has_transform` is set to true.
      fit_samples: If given, number of samples used for fitting the kernel.
    """
    # Save variables.
    self.kernel = kernel
    self.svc_hparams = svc_hparams
    self.svc_grid = svc_grid
    self.cv = cv
    self.n_jobs = n_jobs
    self.need_kernel_fit = need_kernel_fit
    self.has_transform = has_transform
    self.batch_size = batch_size
    self.fit_samples = fit_samples

  @abstractmethod
  def _get_svc_model(self) -> object:
    """Abstract method for constructing the SVC model to use.
    
    Returns:
      An instance of the SVC model, which is to be fitted to the data.
    """
    pass

  @abstractmethod
  def _precompute_model_inputs(self, X: Optional[ArrayOnCPUOrGPU] = None
                               ) -> ArrayOnCPUOrGPU:
    """Abstract method for precomputing the inputs to the SVC model.
    
    If `X` is not provided, training is assumed and the model inputs are
    computed using the stored training data `self.X`, otherwise model inputs
    are computed for testing an already trained model on the provided data `X`.

    Args:
      X: Optional array of inputs of shape `[num_examples, ...]`.

    Returns:
      Model inputs, i.e. when using dual SVM the kernel matrix, or when using
        primal SVM the required feature matrix.
    """
    pass

  def _precompute_kernel_mat(self, X: ArrayOnCPUOrGPU,
                             X2: Optional[ArrayOnCPUOrGPU] = None
                             ) -> ArrayOnCPU:
    """Precompute the kernel matrix.

    Args:
      X: Array of inputs of shape `[num_examples, ...]`.
      X2: Optional second array of inputs of shape `[num_examples2, ...]`.

    Returns:
      The kernel matrix on CPU of shape `[num_examples, num_examples]` if
        `X2 is None` otherwise of shape `[num_examples, num_examples2]`.
    """
    if self.batch_size is None:
      # No batching is used.
      kernel_mat = self.kernel(X) if X2 is None else self.kernel(X, X2)
    elif X2 is None:
      # Batching with symmetrizing the matrix.
      num_examples = X.shape[0]
      num_iters = ceil(num_examples / self.batch_size)
      # Iterate over chunks of shape `[self.batch_size, self.batch_size]`.
      kernel_mat = np.zeros([num_examples, num_examples])
      for i in range(num_iters):
        i_lower, i_upper = (
          i * self.batch_size, min((i+1) * self.batch_size, num_examples))
        for j in range(i, num_iters):
          j_lower, j_upper = (
            j * self.batch_size, min((j+1) * self.batch_size, num_examples))
          # Check if the current part is on the block-diagonal.
          if i_lower == j_lower and i_upper == j_upper:
            kernel_mat[i_lower:i_upper, j_lower:j_upper] = self.kernel(
              X[i_lower:i_upper])
          else:
            kernel_mat[i_lower:i_upper, j_lower:j_upper] = self.kernel(
              X[i_lower:i_upper], X[j_lower:j_upper])
            kernel_mat[j_lower:j_upper, i_lower:i_upper] = (
              kernel_mat[i_lower:i_upper, j_lower:j_upper].T[:, :])
    else:
      # Batching without symmetrizing.
      num_examples = X.shape[0]
      num_examples2 = X2.shape[0]
      num_iters = ceil(num_examples / self.batch_size)
      num_iters2 = ceil(num_examples2 / self.batch_size)
      # Iterate over chunks of shape `[self.batch_size, self.batch_size]`.
      kernel_mat = np.zeros([num_examples, num_examples2])
      for i in range(num_iters):
        i_lower, i_upper = (
          i * self.batch_size, min((i+1) * self.batch_size, num_examples))
        for j in range(num_iters2):
          j_lower, j_upper = (
            j * self.batch_size, min((j+1) * self.batch_size, num_examples2))
          kernel_mat[i_lower:i_upper, j_lower:j_upper] = self.kernel(
            X[i_lower:i_upper], X2[j_lower:j_upper])
    # Return the precomputed kernel matrix.
    return kernel_mat

  def _precompute_feature_mat(self, X: ArrayOnCPUOrGPU) -> ArrayOnGPU:
    """Precompute the feature matrix.

    Args:
      X: Array of inputs of shape `[num_examples, ...]`.

    Returns:
      The feature matrix on GPU of shape `[num_examples, num_features]` where
        `num_features` is the dimension of the feature map.
    """
    if self.batch_size is None:
      # No batching is used.
      feature_mat = self.kernel.transform(X, return_on_gpu=True)
    else:
      # Batching into blocks.
      num_examples = X.shape[0]
      num_iters = ceil(num_examples / self.batch_size)
      # Iterate over chunks of shape `[self.batch_size, ...]`.
      feature_lst = []
      for i in range(num_iters):
        i_lower, i_upper = (
          i * self.batch_size, min((i+1) * self.batch_size, num_examples))
        feature_lst.append(self.kernel.transform(
          X[i_lower:i_upper], return_on_gpu=True))
      # Concatenate feature blocks.
      feature_mat = cp.concatenate(feature_lst, axis=0)
      # Return the feature matrix.
    return feature_mat

  def _precompute_matrix_mult(self, X: ArrayOnCPUOrGPU,
                             X2: Optional[ArrayOnCPUOrGPU] = None
                             ) -> ArrayOnCPU:
    """Precompute matrix multiplication.

    Args:
      X: Array of features of shape `[num_examples, nun_features]`.
      X2: Optional second array of inputs of shape
        `[num_examples2, num_features]`.

    Returns:
      Product matrix of shape `[num_examples, num_examples]` if
        `X2 is None` otherwise of shape `[num_examples, num_examples2]`.
    """
    if X2 is None:
      # Batching with symmetrizing the matrix.
      num_examples = X.shape[0]
      num_iters = ceil(num_examples / _MAT_MUL_BSIZE)
      # Iterate over chunks of shape `[_MAT_MUL_BSIZE, _MAT_MUL_BSIZE]`.
      prod_mat = np.empty([num_examples, num_examples], dtype=float)
      for i in range(num_iters):
        i_lower, i_upper = (
          i * _MAT_MUL_BSIZE, min((i+1) * _MAT_MUL_BSIZE, num_examples))
        for j in range(i, num_iters):
          j_lower, j_upper = (
            j * _MAT_MUL_BSIZE, min((j+1) * _MAT_MUL_BSIZE, num_examples))
          _prod_mat = X[i_lower:i_upper] @ X[j_lower:j_upper].T
          if isinstance(_prod_mat, ArrayOnGPU):
            _prod_mat = cp.asnumpy(_prod_mat)
          prod_mat[i_lower:i_upper, j_lower:j_upper] = _prod_mat
          prod_mat[j_lower:j_upper, i_lower:i_upper] = _prod_mat.T[:, :]
    else:
      # Batching without symmetrizing.
      num_examples = X.shape[0]
      num_examples2 = X2.shape[0]
      num_iters = ceil(num_examples / _MAT_MUL_BSIZE)
      num_iters2 = ceil(num_examples2 / _MAT_MUL_BSIZE)
      # Iterate over chunks of shape `[_MAT_MUL_BSIZE, _MAT_MUL_BSIZE]`.
      prod_mat = np.zeros([num_examples, num_examples2])
      for i in range(num_iters):
        i_lower, i_upper = (
          i * _MAT_MUL_BSIZE, min((i+1) * _MAT_MUL_BSIZE, num_examples))
        for j in range(num_iters2):
          j_lower, j_upper = (
            j * _MAT_MUL_BSIZE, min((j+1) * _MAT_MUL_BSIZE, num_examples2))
          _prod_mat = X[i_lower:i_upper] @ X2[j_lower:j_upper].T
          if isinstance(_prod_mat, ArrayOnGPU):
            _prod_mat = cp.asnumpy(_prod_mat)
          prod_mat[i_lower:i_upper, j_lower:j_upper] = _prod_mat
    # Return the precomputed matrix product.
    return prod_mat

  def fit(self, X: ArrayOnCPUOrGPU, y: ArrayOnCPU) -> 'PrecomputedSVCBase':
    """Fits the SVC model to the data optionally by cross-validation.

    Args:
      X: Array of inputs.
      y: Array of outputs.

    Returns:
      A fitted model object.
    """
    # Save training data.
    self.X, self.y = X, y
    # Initialize model.
    self.model = self._get_svc_model() 
    if self.svc_grid is not None:
      # Stratified k-fold cross-validation.
      self.kfold = StratifiedKFold(
        n_splits=min(self.cv, np.min(np.bincount(y))))
      # Create `GridSearchCV` object.
      self.model = GridSearchCV(estimator=self.model, param_grid=self.svc_grid,
                                cv=self.kfold, n_jobs=self.n_jobs)
    # Fit the kernel if required.
    if self.need_kernel_fit:
      if self.fit_samples is not None:
        # Select fitting data.
        idx_fit = np.random.choice(
          range(X.shape[0]), min(self.fit_samples, X.shape[0]), replace=False)
        X_fit = X[idx_fit]
      else:
        X_fit = X
      self.kernel.fit(X_fit)
    # Precompute model inputs.
    model_inp = self._precompute_model_inputs()
    # Fit the model.
    self.model.fit(model_inp, y)
    # Return the fitted object.
    return self

  def predict(self, X: ArrayOnCPUOrGPU) -> ArrayOnCPU:
    """Predict classes for the data.

    Args:
      X: Array of inputs.

    Returns:
      Predicted classes for examples in `X`.
    """
    # Check if the model is fitted.
    check_is_fitted(self.model)
    # Precompute model inputs.
    model_inp = self._precompute_model_inputs(X)
    # Predict values.
    y_pred = self.model.predict(model_inp)
    # Return predicted classes.
    if isinstance(y_pred, ArrayOnGPU):
      y_pred = cp.asnumpy(y_pred)
    return y_pred
  
# ------------------------------------------------------------------------------