"`SVC` with precomputed kernel."
import cupy as cp
import os

from ..static.kernels import Kernel
from ..utils import ArrayOnCPUOrGPU, ArrayOnCPU
from .pre_base import PrecomputedSVCBase
from sklearn.svm import SVC
from typing import Dict, Optional
from warnings import simplefilter, filterwarnings

# Include this for ignoring `CorvergenceWarning` in SVC.
simplefilter('ignore')
filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = 'ignore'


# Default `SVC` hyperparameter values.
_DEFAULT_SVC_HPARAMS = {
  'decision_function_shape': 'ovo',
  'max_iter': 10000
}


class PrecomputedKernelSVC(PrecomputedSVCBase):
  """`SVC` with precomputed kernel with optional `GridSearchCV`."""
  def __init__(self,
               kernel: Kernel,
               svc_hparams: Dict = {},
               svc_grid: Optional[Dict] = None,
               cv: int = 3,
               n_jobs: int = -1,
               need_kernel_fit: bool = False,
               batch_size: Optional[int] = None,
               fit_samples: Optional[int] = None,
               has_transform: bool = False):
    """Initializer for `PrecomputedKernelSVC`.

    Args:
      kernel: A callable for computing the kernel matrix.
      svc_grid: Hyperparameter grid to cross-validate over, set to `None`
        for no cross-validation.
      svc_hparams: Additional hyperparameters for `SVC`.
      cv: Number of CV splits.
      n_jobs: Number of jobs for `GridSearchCV`.
      need_kernel_fit: Whether the kernel needs to be fitted to the data.
      batch_size: If given, compute the kernel matrix in chunks of shape
        `[batch_size, batch_size]`, or in chunks of shape [batch_size, ...]
        if `has_transform` is set to true.
      fit_samples: Number of samples used for fitting the kernel.
      has_transform: Whether the kernel has a feature transform.
    """
    super().__init__(kernel, svc_hparams=svc_hparams, svc_grid=svc_grid, cv=cv,
                     n_jobs=n_jobs, need_kernel_fit=need_kernel_fit,
                     batch_size=batch_size, fit_samples=fit_samples)
    self.has_transform = has_transform
    # Set default SVC hparams.
    for key, val in _DEFAULT_SVC_HPARAMS.items():
      self.svc_hparams.setdefault(key, val)
    # Force precomputed kernel.
    self.svc_hparams['kernel'] = 'precomputed'

  def _get_svc_model(self) -> object:
    """Returns a new instance of a dual SVC model.

    Returns:
      An instance of a dual SVC model, which is to be fitted to the data.
    """
    return SVC(**self.svc_hparams)

  def _precompute_model_inputs(self, X: Optional[ArrayOnCPUOrGPU] = None
                              ) -> ArrayOnCPU:
    """Precomputes the kernel matrix, which is used as iinput for the SVC model.

    If `X` is not provided, training is assumed and the kernel matrix is
    computed using the stored training data `self.X`, otherwise it is computed
    using the provided `X` matrix and the stored data in `self.X`.

    Args:
      X: Optional array of inputs of shape `[num_test, ...]`.

    Returns:
      Kernel matrix of shape `[num_train, num_train]` when `X is None` else
        it is of shape `[num_test, num_train]`.
    """
    if X is None:  # Training.
      if self.has_transform:
        # Compute feature matrix.
        feature_mat = self._precompute_feature_mat(self.X)
        # Save features for inference.
        self.X_feat = feature_mat
        # Compute kernel matrix and move to CPU memory.
        kernel_mat = cp.asnumpy(self._precompute_matrix_mult(feature_mat))
      else:
        # Compute kernel matrix.
        kernel_mat = self._precompute_kernel_mat(self.X)
    else:  # Testing.
      if self.has_transform:
        # Compute feature matrix.
        feature_mat = self._precompute_feature_mat(X)
        # Compute kernel matrix and move to CPU memory.
        kernel_mat = cp.asnumpy(self._precompute_matrix_mult(
          feature_mat, self.X_feat))
      else:
        # Compute kernel matrix.
        kernel_mat = self._precompute_kernel_mat(X, self.X)
    return kernel_mat

# ------------------------------------------------------------------------------