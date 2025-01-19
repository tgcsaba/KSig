"`LinearSVC` with precomputed features."
import cupy as cp

from ..static.features import KernelFeatures
from ..utils import ArrayOnCPUOrGPU, ArrayOnCPU, ArrayOnGPU
from .pre_base import PrecomputedSVCBase
from cuml.svm import LinearSVC as LinearSVCOnGPU
from sklearn.svm import LinearSVC
from typing import Dict, Optional


# Default `LinearSVC` hyperparameter values.
_DEFAULT_LIN_SVC_HPARAMS = {
  'dual': False,
  'fit_intercept': False,
  'tol': 1e-3,
}
_DEFAULT_LIN_SVC_GPU_HPARAMS = {
  'fit_intercept': False,
  'tol': 1e-3,
}


class PrecomputedFeatureLinSVC(PrecomputedSVCBase):
  """`LinearSVC` with precomputed features with optional cross-validation."""

  def __init__(self,
               kernel: KernelFeatures,
               svc_hparams: Dict = {},
               svc_grid: Optional[Dict] = None,
               cv: int = 5,
               n_jobs: int = -1,
               need_kernel_fit: bool = False,
               batch_size: Optional[int] = None,
               on_gpu: bool = False):
    """Initializer for `PrecomputedFeatureLinSVC`.

    Args:
      kernel: A callable for computing features.
      svc_grid: Hyperparameter grid to cross-validate over, set to `None`
        for no cross-validation.
      svc_hparams: Additional hyperparameters for `LinearSVC`.
      cv: Number of CV splits.
      n_jobs: Number of jobs for `GridSearchCV`.
      need_kernel_fit: Whether the features need to be fitted to the data.
      batch_size: If given, compute the feature matrix in chunks of shape
        `[batch_size, ...]` in order to save memory.
      on_gpu: Whether to use the GPU implementation or not.
    """
    n_jobs = 1 if on_gpu else n_jobs  # GPU implementation does not support it.
    super().__init__(kernel, svc_hparams=svc_hparams, svc_grid=svc_grid, cv=cv,
                     n_jobs=n_jobs, need_kernel_fit=need_kernel_fit,
                     batch_size=batch_size)
    self.on_gpu = on_gpu
    # Set default `LinearSVC` hparams.
    if on_gpu:
      for key, val in _DEFAULT_LIN_SVC_GPU_HPARAMS.items():
        self.svc_hparams.setdefault(key, val)
    else:
      for key, val in _DEFAULT_LIN_SVC_HPARAMS.items():
        self.svc_hparams.setdefault(key, val)

  def _get_svc_model(self) -> object:
    """Returns a new instance of a linear SVC model.

    Returns:
      An instance of a linear SVC model, which is to be fitted to the data.
    """
    return (LinearSVCOnGPU(**self.svc_hparams) if self.on_gpu else
            LinearSVC(**self.svc_hparams))

  def _precompute_model_inputs(self, X: Optional[ArrayOnCPUOrGPU] = None
                              ) -> ArrayOnCPU:
    """Precomputes the feature matrix, which is used as input for the LinearSVC.

    If `X` is not provided, training is assumed and the kernel matrix is
    computed using the stored training data `self.X`, otherwise it is computed
    using the provided `X` data matrix.

    Args:
      X: Optional array of inputs of shape `[num_test, ...]`.

    Returns:
      Feature matrix of shape `[num_train, ...]` when `X is None` else
        it is of shape `[num_test, ...]`.
    """
    if X is None:  # Training.
      feature_mat = self._precompute_feature_mat(self.X)
    else:  # Testing.
      feature_mat = self._precompute_feature_mat(X)
    # Move feature matrix to the required device.
    if self.on_gpu and not isinstance(feature_mat, ArrayOnGPU):
      feature_mat = cp.asarray(feature_mat)
    elif not self.on_gpu and not isinstance(feature_mat, ArrayOnCPU):
      feature_mat = cp.asnumpy(feature_mat)
    return feature_mat

# ------------------------------------------------------------------------------