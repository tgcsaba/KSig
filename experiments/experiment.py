"""Base pipeline for the experiments."""

import sys
sys.path.append('..')

import cupy as cp
import ksig
import numpy as np
import os
import random
import yaml

from datasets import load_dataset, preprocess_dataset
from ksig.static.kernels import Kernel
from ksig.models import PrecomputedFeatureLinSVC, PrecomputedKernelSVC
from ksig.utils import _EPS, ArrayOnCPUOrGPU, ArrayOnGPU
from math import ceil
from sklearn.base import TransformerMixin as Transformer
from sklearn.model_selection import ParameterGrid
from time import time
from tqdm.auto import tqdm
from typing import Optional
from utils import flat_to_nested_dict


def suggest_bandwidth(X: ArrayOnCPUOrGPU, is_static: bool = False,
                      num_samples: int = 1000) -> float:
  """Suggests bandwidth according to the median heuristic.

  Args:
    X: Training dataset on CPU or GPU.
    is_static: Whether the kernel is a static kernel.
    num_samples: Number of samples used to approximate the distance matrix.

  Returns:
    Suggested bandwidth value.
  """
  xp = cp if isinstance(X, ArrayOnGPU) else np
  # Sample indices.
  sample_size = X.shape[0] if is_static else X.shape[0] * X.shape[1]
  num_samples = min(num_samples, sample_size)
  idx = xp.random.choice(range(sample_size), size=num_samples, replace=False)
  # Reshape data.
  if is_static:
    samples = X.reshape([X.shape[0], -1])[idx]
  else:
    samples = X.reshape([-1, X.shape[-1]])[idx]
  norms = xp.sum(xp.square(samples), axis=-1) / 2.
  sq_dists = norms[:, None] + norms[None, :] - samples @ samples.T
  # Mask diagonal entries and take sqrt.
  dists = xp.sqrt(xp.maximum(sq_dists[~xp.eye(num_samples, dtype=bool)], _EPS))
  # Compute median.
  med_dist = xp.median(dists) 
  return float(med_dist)


def get_kernel(X: ArrayOnCPUOrGPU, name: str, hparams: dict,
               seed: Optional[int] = None) -> Kernel:
  """Get an instance of a kernel with given hyperparameters.

  Args:
    X: Training dataset on CPU or GPU.
    name: Name of the kernel to get. Options: 'ksig_r', 'rfsf_dp', 'rfsf_trp'.
    hparams: Hyperparameters for the kernel.
    seed: Seed for random state.

  Returns:
    An instance of the requested kernel initialized with the given parameters. 
  """
  _name = name.replace('_', '').lower()
  # Suggest bandwidth via median heuristic.
  is_static = _name == 'rff' or _name == 'rbf'
  bandwidth = suggest_bandwidth(X, is_static=is_static)
  if 'static' in hparams and 'bandwidth' in hparams['static']:
    hparams['static']['bandwidth'] *= bandwidth
  
  # Kernel selection.  
  random_state = cp.random.RandomState(seed)

  if _name == 'ksigr':  # Truncated signature kernel.
    static_kern = ksig.static.kernels.RBFKernel(**hparams['static'])
    kern = ksig.kernels.SignatureKernel(static_kernel=static_kern,
                                        **hparams['signature'])
    
  elif _name == 'ksigpde':  # Signature-PDE kernel.
    static_kern = ksig.static.kernels.RBFKernel(**hparams['static'])
    kern = ksig.kernels.SignaturePDEKernel(static_kernel=static_kern,
                                           **hparams['signature'])
    
  elif _name == 'gak':  # Global Alignment Kernel.
    static_kern = ksig.static.kernels.RBFKernel(**hparams['static'])
    kern = ksig.kernels.GlobalAlignmentKernel(static_kernel=static_kern)

  elif _name == 'rfsfdp':  # RFSF-DP kernel.
    # Set number of components so that the effective feature size is as given.
    hparams['static']['n_components'] = ceil(
      hparams['static']['n_components'] /
      (2 * (2**hparams['signature']['n_levels'] - 1)))
    static_feat = ksig.static.features.RandomFourierFeatures(
      **hparams['static'], random_state=random_state)
    proj = ksig.projections.DiagonalProjection()
    kern = ksig.kernels.SignatureFeatures(
      static_features=static_feat, **hparams['signature'], projection=proj)
    
  elif _name == 'rfsftrp':  # RFSF-TRP kernel.
    # Set number of components so that the effective feature size is as given.
    hparams['static']['n_components'] = ceil(
      hparams['static']['n_components'] / hparams['signature']['n_levels'])
    static_feat = ksig.static.features.RandomFourierFeatures(
      **hparams['static'], random_state=random_state)
    proj = ksig.projections.TensorizedRandomProjection(
      n_components=hparams['static']['n_components'], random_state=random_state)
    kern = ksig.kernels.SignatureFeatures(
      static_features=static_feat, **hparams['signature'], projection=proj)
    
  elif _name == 'rbf':  # RBF kernel.
    kern = ksig.static.kernels.RBFKernel(**hparams['static'])

  elif _name == 'rff':  # Random Fourier Features kernel.
    kern = ksig.static.features.RandomFourierFeatures(
      **hparams['static'], random_state=random_state)
    
  elif _name == 'rws':  # Random Warping Series kernel.
    random_state = cp.random.RandomState(seed)
    kern = ksig.kernels.RandomWarpingSeries(**hparams,
                                            random_state=random_state)
  else:
    raise ValueError(f'Unknown kernel name: {name}')
  
  return kern


def get_model(X: ArrayOnCPUOrGPU, config: dict, seed: Optional[int] = None
              ) -> object:
  """Get an instance of a model with given hyperparameters.

  Args:
    X: Training dataset on CPU or GPU.
    config: Configuration dictionary for the model.
    seed: Seed for random state.

  Returns:
    An instance of the requested SVC model.
  """
  _name = config['svc_name'].replace('_', '').lower()
  kernel = get_kernel(
    X, config['kernel_name'], config['hparams']['kernel'], seed=seed)
  if _name == 'presvc':
    model = PrecomputedKernelSVC(kernel, **config['hparams']['svc'])
  elif _name == 'prelinsvc':
    model = PrecomputedFeatureLinSVC(kernel, **config['hparams']['svc'])
  else:
    raise ValueError(f'Unknown model name: {config["svc_name"]}')
  return model


def run_experiment(flat_config: dict):
  """
  Runs the experiment with given settings.

  Args:
    flat_config: Flat config file for the experiment.
  """

  # Name the experiment.
  kernel_name = flat_config['model__kernel_name']
  experiment_name = (
    flat_config['dataset__load__name'] + '__' +
    kernel_name + '__' +
    str(flat_config['dataset__load__max_len']) + '__' +
    str(flat_config['experiment__id']))
  
  # Create experiment directory.
  experiment_dir  = flat_config['experiment__dir']
  if not os.path.isdir(experiment_dir):
    os.makedirs(experiment_dir)

  # Create placeholder experiment result file or skip if exists.
  experiment_fp = os.path.join(experiment_dir, experiment_name + '.yml')
  if os.path.exists(experiment_fp):
    print(f'Experiment {experiment_name} already exists. Continuing...')
    return
  print(f'Experiment {experiment_name} is running...')
  with open(experiment_fp, 'w') as f:
    pass

  # Parse static hyperparameters.
  flat_config_other = {
    k: v for k, v in flat_config.items() if 'HPGrid' not in k}
  config_other = flat_to_nested_dict(flat_config_other)
  # Parse outer hyperparameter grid.
  flat_config_hps_grid = list(ParameterGrid({
    k: v for k, v in flat_config.items() if 'HPGrid' in k}))

  # Load dataset.
  print('Loading dataset...')
  X_tr, y_tr, X_te, y_te = load_dataset(config_other['dataset'],
                                        config_other['experiment']['id'])
  # Convert to float32.
  X_tr = np.float32(X_tr)
  X_te = np.float32(X_te)
  y_tr = np.int32(y_tr)
  y_te = np.int32(y_te)

  if X_tr.shape[0] < 1000:
    # Move input data into GPU memory for the GPU-based models.
    X_tr, X_te = cp.asarray(X_tr), cp.asarray(X_te)

  # Search over the grid.
  print(f'Iterating over hp grid ({len(flat_config_hps_grid)} items)...')
  best_score_cv = -np.inf
  best_config = {}
  # Time the full experiment.
  time_start = time()
  for flat_config_hps in tqdm(flat_config_hps_grid):
    # Fill nested config with grid hparams.
    flat_config_hps = {
      k[8:]: v for k, v in flat_config_hps.items()}
    config = flat_to_nested_dict(flat_config_hps, base_dict=config_other)
    # Preprocess sequences.
    X_tr_pre, X_te_pre = X_tr[:], X_te[:]
    X_tr_pre, X_te_pre  = preprocess_dataset(X_tr_pre, X_te_pre,
                                             config['dataset'])
    # Set seed.
    random.seed(config['experiment']['id'])
    np.random.seed(config['experiment']['id'])
    cp.random.seed(config['experiment']['id'])
    # Initialize the model.
    model = get_model(X_tr_pre, config['model'],
                      seed=config['experiment']['id'])
    # Fit the model.
    model.fit(X_tr_pre, y_tr)
    # Get CV score.
    score_cv = float(model.model.best_score_)
    # Test the model.
    acc_train = float(model.score(X_tr_pre, y_tr))
    acc_test = float(model.score(X_te_pre, y_te))
    # Check if best result.
    if score_cv >= best_score_cv:
      # Document results.
      del config['model']['hparams']['svc']['svc_grid']
      config['model']['hparams']['svc']['svc_hparams'] = (
        model.model.best_params_)
      config['results'] = {}
      config['results']['score_cv'] = score_cv
      config['results']['acc_train'] = acc_train
      config['results']['acc_test'] = acc_test
      # Set new best score and config.
      best_score_cv = score_cv
      best_config = config
  # Timing over.
  elapsed = time() - time_start
  best_config['results']['elapsed'] = elapsed
  # Export results.
  with open(experiment_fp, 'w') as f:
    yaml.dump(best_config, f)

# ------------------------------------------------------------------------------