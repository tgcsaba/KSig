"""Sequential data preprocessing utilities."""

import cupy as cp
import numpy as np

from enum import Enum
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union

from .utils import ArrayOnCPUOrGPU, ArrayOnGPU, check_positive_value


# -----------------------------------------------------------------------------

class SequenceTabulator(BaseEstimator, TransformerMixin):
  """Transformer that tabulates sequences to even length."""

  def __init__(self, max_len: Optional[int] = None):
    """Initializes the `SequenceTabulator` object.

    Args:
      max_len: Maximum length of sequences.
    """
    self.max_len = (check_positive_value(max_len, 'max_len')
                    if max_len is not None else None)

  def fit(self, X_seq: Union[ArrayOnCPUOrGPU, List[ArrayOnCPUOrGPU]]
          ) -> 'SequenceTabulator':
    """Fits the `SequenceTabulator` to the data and returns the fitted object.

    Args:
      X_seq: An array or list of sequences on CPU or GPU.
    """
    max_seq_len = np.max([x.shape[0] for x in X_seq])
    self.max_len_ = (min(self.max_len, max_seq_len)
                     if self.max_len is not None else max_seq_len)
    return self

  def transform(self, X_seq: Union[ArrayOnCPUOrGPU, List[ArrayOnCPUOrGPU]]
                ) -> ArrayOnCPUOrGPU:
    """Tabulates sequences contained in `X_seq` to uniform length.

    Args:
      X_seq: An array or list of sequences on CPU or GPU.

    Returns:
      A tabulated array of sequences on CPU or GPU.
    """
    # Correct linear algebra package to use.
    xp = cp if isinstance(X_seq[0], ArrayOnGPU) else np
    needs_interp = xp.any(xp.asarray([
      x.shape[0] != X_seq[0].shape[0] or xp.any(xp.isnan(x))
      or x.shape[0] > self.max_len_ for x in X_seq]))
    if needs_interp.item():
      # Filter NANs.
      X_seq = [x[xp.all(~xp.isnan(x), axis=-1)] for x in X_seq]
      # Channel-wise interpolation.
      X_seq = [
        xp.stack([xp.interp(
          xp.linspace(0, 1, self.max_len_),
          xp.linspace(0, 1, x.shape[0]), x[:, i_c])
          for i_c in range(x.shape[1])], axis=1)
        for i, x in enumerate(X_seq)]
    if isinstance(X_seq, list):
      X_seq = xp.stack(X_seq, axis=0)
    return X_seq


# -----------------------------------------------------------------------------

class SequenceAugmentor(BaseEstimator, TransformerMixin):
  """Transformer that tabulates sequences to even length."""

  def __init__(self, add_time: bool = True, lead_lag: bool = True,
               basepoint: bool = True, normalize: bool = True,
               max_time: float = 1., max_len: Optional[int] = None):
    """Initializes the `SequenceAugmentor` object.

    Args:
      add_time: Whether to augment with time coordinate.
      lead_lag: Whether to augment with lead-lag.
      basepoint: Whether to augment with basepoint.
      normalize: Whether to normalize time series.
      max_time: Maximum time if `add_time is True`.
      max_len: Maximum length of sequences.
    """
    self.add_time = add_time
    self.lead_lag = lead_lag
    self.basepoint = basepoint
    self.normalize = normalize
    self.max_time = max_time
    self.max_len = max_len

  def fit(self, X_seq: ArrayOnCPUOrGPU) -> 'SequenceAugmentor':
    """Fits the `SequenceAugmentor` to the data and returns the fitted object.

    Args:
      X_seq: An array sequences on CPU or GPU.
    """
    xp = cp if isinstance(X_seq, ArrayOnGPU) else np
    if self.normalize:
      self.scale_ = xp.max(X_seq)
    return self

  def transform(self, X_seq: ArrayOnCPUOrGPU) -> ArrayOnCPUOrGPU:
    """Augments sequences in `X_seq` by adding time and lead-lag.

    Args:
        X_seq: An array of sequences on CPU or GPU.

    Returns:
        An augmented array of sequences on CPU or GPU.
    """
    xp = cp if isinstance(X_seq, ArrayOnGPU) else np
    # Normalization.
    if self.normalize:
      X_seq /= self.scale_
    # Lead-lag augmentation.
    if self.lead_lag:
      X_seq = xp.repeat(X_seq, 2, axis=1)
      X_seq = xp.concatenate((X_seq[:, 1:], X_seq[:, :-1]), axis=-1)
    # Time augmentation.
    if self.add_time and self.max_time > 1e-6:
      time = xp.linspace(0., self.max_time, X_seq.shape[1])
      X_seq = xp.concatenate((
        xp.tile(time[None, :, None], [X_seq.shape[0], 1, 1]), X_seq), axis=-1)
    # Basepoint augmentation.
    if self.basepoint:
      X_seq = xp.concatenate((xp.zeros_like(X_seq[:, :1]), X_seq), axis=1)
    # If after augmentation exceeded max length, interpolate back.
    if self.max_len is not None and X_seq.shape[1] > self.max_len:
      current = xp.linspace(0, 1, X_seq.shape[1])
      target = xp.linspace(0, 1, self.max_len)
      interp_fn = lambda x: xp.interp(target, current, x)
      X_seq = xp.apply_along_axis(interp_fn, 1, X_seq)
    return X_seq


# -----------------------------------------------------------------------------