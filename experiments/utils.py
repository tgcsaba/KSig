"""Utils for the experiments."""

import itertools

from copy import deepcopy
from typing import Dict, Generator


def search_space_to_configs(search_space: Dict) -> Generator:
  """Takes a search space and transforms it into a generator of configs.
  
  Args:
    search_space: Flat dictionary containing the search space definition
      with values as lists specifying the possible options for a given key.

  Returns:
    confs_as_generator: A generator of configs, each corresponding to a setting.
  """
  search_space_as_tuples = [
    [(k, v2) for v2 in v] if isinstance(v, list) else [(k, v)]
    for k, v in search_space.items()]
  confs_as_generator = itertools.product(*search_space_as_tuples)
  return confs_as_generator


def flat_to_nested_dict(flat_dict: Dict, sep: str = '__', base_dict: Dict = {}
                        ) -> Dict:
  """Converts a flat dictionary into a (hierarchically) nested one.
  
  Args:
    flat_dict: A flat dictionary with concatenated keys across the hierarchy.
    sep: The key separator used to indicate nesting.
    base_dict: A base dictionary into which the converted nested dict is merged.

  Returns:
    nested_dict: A nested dictionary by splitting each of the keys along `sep`. 
  """
  branch_keys = {k.split(sep)[0] for k in flat_dict if sep in k}
  nested_dict = deepcopy(base_dict)
  for branch_k in branch_keys:
    branch_dict = {
      sep.join(k.split(sep)[1:]): v for k, v in flat_dict.items()
      if k.split(sep)[0] == branch_k}
    if branch_k in base_dict:
      nested_dict[branch_k] = flat_to_nested_dict(
        branch_dict, base_dict=base_dict[branch_k])
    else:
      nested_dict[branch_k] = flat_to_nested_dict(branch_dict)
  leaf_dict = {k: v for k, v in flat_dict.items() if sep not in k}
  nested_dict = {**leaf_dict, **nested_dict}
  return nested_dict

# ------------------------------------------------------------------------------