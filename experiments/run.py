"""Runner script for the experiments."""

import argparse
import os
import yaml

from experiment import run_experiment
from utils import search_space_to_configs


if __name__ == '__main__':
  # Parse arguments.
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu_id', '-g', type=str, default='')
  parser.add_argument('--config', '-c', type=str, default='small')
  args = parser.parse_args()

  # Set which GPU to use.
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

  # Load search space from config file.
  config_file = f'./configs/{args.config}.yml'
  # Load base configs.
  with open(config_file, 'r') as f:
    search_spaces = yaml.load(f, yaml.FullLoader)

  # Iterate over search spaces.
  for ss_name in search_spaces:
    # Create a generator for all possible configs.
    confs_as_tuples = search_space_to_configs(search_spaces[ss_name])
    # Iterate over the generator.
    for conf in confs_as_tuples:
      result = run_experiment(dict(conf))

# ------------------------------------------------------------------------------