"Dataset loader for the experiments."
import glob
import numpy as np
import os

from ksig.preprocessing import SequenceTabulator, SequenceAugmentor
from ksig.utils import ArrayOnCPU, ArrayOnCPUOrGPU
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

os.environ['PYTHONUTF8'] = '1'


_DEFAULT_DATA_LOAD_CONFIG = {
  'max_len': 200,
}

_DEFAULT_DATA_PREPROCESS_CONFIG = {
  'add_time': False,
  'lead_lag': False,
  'basepoint': False,
  'normalize': True
}


def load_dataset(config: dict, experiment_id: int) -> Tuple[ArrayOnCPU]:
  """Loads and tabulates the dataset.

  Args:
      config: Data load config to use.
      experiment_id: Seed for splitting if `config['load']['split']` is given.

  Returns:
      The loaded dataset: `X_tr`, `y_tr`, `X_te`, `y_te`.
  """
  # Set default load config params.
  for key, val in _DEFAULT_DATA_LOAD_CONFIG.items():
    config['load'].setdefault(key, val)
  # Extract config params.
  ds_name = config['load']['name']
  ds_dir = config['load']['dir']
  split = config['load']['split'] if 'split' in config['load'] else None

  # Load the dataset.
  if ds_name == 'fNIRS2MW':
    X_tr, X_te, y_tr, y_te = load_fnirs2mw_dataset(ds_dir, split,
                                                   experiment_id)
  elif ds_name == 'SITS1M':
    X_tr, X_te, y_tr, y_te = load_sits1m_dataset(ds_dir, split, experiment_id)
  else:
    X_tr, X_te, y_tr, y_te = load_uea_dataset(ds_name, ds_dir, split,
                                              experiment_id)

  # Tabulate input sequences.
  tabulator = SequenceTabulator(max_len=config['load']['max_len'])
  X_tr = tabulator.fit_transform(X_tr)
  X_te = tabulator.transform(X_te)

  # Encode output labels.
  encoder = LabelEncoder()
  y_tr = encoder.fit_transform(y_tr)
  y_te = encoder.transform(y_te)

  return X_tr, y_tr, X_te, y_te


def load_uea_dataset(ds_name: str, ds_dir: str,
                     split: Optional[float] = None,
                     experiment_id: Optional[int] = None
                     ) -> Tuple[ArrayOnCPU]:
  """Loads a UEA dataset.

  Args:
    ds_name: Name of the dataset
    ds_dir: Directory where the archive is extracted
    split: Ratio of the test set for manual splitting, if not given uses the
      standard train-test split provided.
    experiment_id: Seed used for splitting when `split is not None`.

  Returns:
    `X_tr, X_te, y_tr, y_te`, the inputs and outputs for training and testing.
  """
  ds_dir = os.path.join(ds_dir, ds_name)
  ds_tr_fp = os.path.join(ds_dir, f'{ds_name}_TRAIN.arff')
  ds_te_fp = os.path.join(ds_dir, f'{ds_name}_TEST.arff')
  def _load_and_parse_dataset(_fp):
    with open(_fp, 'r', encoding="utf-8") as _f:
      _data = loadarff(_f)
    _X = [
      np.asarray([[float(s) for s in c] for c in x[0]]).T for x in _data[0]]
    _y = [x[1] for x in _data[0]]
    return _X, _y
  # Load train and test data.
  X_tr, y_tr = _load_and_parse_dataset(ds_tr_fp)
  X_te, y_te = _load_and_parse_dataset(ds_te_fp)
  if split is not None:
    X, y = X_tr + X_te, y_tr + y_te
    X_tr, X_te, y_tr, y_te = train_test_split(
      X, y, test_size=split, random_state=experiment_id, stratify=y)
  return X_tr, X_te, y_tr, y_te


def load_fnirs2mw_dataset(ds_dir: str, split: float,
                          experiment_id: Optional[int] = None,
                          binary: bool = True
                          ) -> Tuple[ArrayOnCPU]:
  """Loads the fNIRS2MW dataset.

  Args:
    ds_dir: Directory where the dataset is extracted
    split: Ratio of the test set for splitting.
    experiment_id: Seed for splitting.
    binary: Whether to convert the problem into binary classification task.

  Returns:
    `X_tr, X_te, y_tr, y_te`, the inputs and outputs for training and testing.
  """
  files = glob.glob(os.path.join(ds_dir, '*.csv'))
  X, Y = [], []
  for file in files:
    with open(file, 'r') as f:
      lines = f.readlines()
    idx = -1
    for line in lines[1:]:
        line = line.strip().split(',')
        if idx == -1:
          idx = int(line[-2])
          x = []
        if idx != int(line[-2]):
          X.append(np.asarray(x))
          Y.append(y)
          idx = int(line[-2])
          x = []
        x.append(np.float32(line[:-2]))
        y = int(line[-1])
    X.append(np.asarray(x))
    Y.append(y)
  X = np.stack(X, axis=0)
  Y = np.asarray(Y)
  X_tr, X_te, y_tr, y_te = train_test_split(
    X, Y, test_size=split, random_state=experiment_id, stratify=Y)
  # Threshold values between low and high mental activity. (0, 1 - 2, 3).
  if binary:
    y_tr = np.int32(y_tr >= 2)
    y_te = np.int32(y_te >= 2)
  return X_tr, X_te, y_tr, y_te


def load_sits1m_dataset(ds_dir: str, split: Optional[float] = None,
                        experiment_id: int = 0) -> Tuple[ArrayOnCPU]:
  """Loads the SITS1M dataset.

  Args:
    ds_dir: Directory where the dataset is extracted
    split: Ratio of the test set for random splitting.
    experiment_id: The index of the fold to load, or seed for random split.

  Returns:
    `X_tr, X_te, y_tr, y_te`, the inputs and outputs for training and testing.
  """
  # Set file paths for train and test data.
  if split is None:
    file_tr = os.path.join(ds_dir, f'SITS1M_fold{experiment_id+1}_TRAIN.csv')
    file_te = os.path.join(ds_dir, f'SITS1M_fold{experiment_id+1}_TEST.csv')
  else:
    file_tr = os.path.join(ds_dir, f'SITS1M_fold1_TRAIN.csv')
    file_te = os.path.join(ds_dir, f'SITS1M_fold1_TEST.csv')

  # Load training data.
  with open(file_tr, 'r') as f:
    lines_tr = f.readlines()
  X_tr = np.asarray([np.float32(line.split(',')[1:]) for line in lines_tr])
  y_tr = np.asarray([np.int32(line.split(',')[0]) for line in lines_tr])

  # Load testing data.
  with open(file_te, 'r') as f:
    lines_te = f.readlines()
  X_te = np.asarray([np.float32(line.split(',')[1:]) for line in lines_te])
  y_te = np.asarray([np.int32(line.split(',')[0]) for line in lines_te])

  # Expand last axis for channels.
  X_tr, X_te = X_tr[..., None], X_te[..., None]

  # If `split is not None`, resplit the data.
  if split is not None:
    X = np.concatenate((X_tr, X_te), axis=0)
    y = np.concatenate((y_tr, y_te), axis=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
      X, Y, test_size=split, random_state=experiment_id, stratify=y)
  return X_tr, X_te, y_tr, y_te


def preprocess_dataset(X_tr: ArrayOnCPUOrGPU, X_te: ArrayOnCPUOrGPU,
                       config: dict = {}) -> Tuple[ArrayOnCPUOrGPU]:
  """Preprocesses the dataset.

  Args:
      config: Preprocess config to use.
  """
  if 'preprocess' not in config:
    config['preprocess'] = {}
  for key, val in _DEFAULT_DATA_PREPROCESS_CONFIG.items():
    config['preprocess'].setdefault(key, val)
  augmentor = SequenceAugmentor(**config['preprocess'],
                                max_len=config['load']['max_len'])
  X_tr = augmentor.fit_transform(X_tr)
  X_te = augmentor.transform(X_te)
  return X_tr, X_te

# -----------------------------------------------------------------------------