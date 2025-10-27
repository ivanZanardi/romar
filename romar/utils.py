"""
Utility module for class management, file I/O, data processing, and
numerical utilities.

This module provides functionality for:
- Dynamically retrieving classes from modules.
- Saving and loading simulation cases using `dill`.
- Parallel loading and generation of data with `joblib`.
- Nested dictionary operations and key manipulation.
- Error metrics and timeout-safe ODE solving.

Used throughout ROMAr workflows for data handling and simulation orchestration.
"""

import os
import sys
import types
import signal
import inspect
import collections
import numpy as np
import scipy as sp
import joblib as jl
import dill as pickle

from romar import env
from tqdm import tqdm
from typing import *


# Classes
# =====================================
def get_class(
  modules: Union[types.ModuleType, List[types.ModuleType]],
  name: Optional[str] = None,
  kwargs: Union[dict, None] = None
) -> callable:
  """
  Return a class object given its name and the module it belongs to.

  This function searches for a class with the specified name within the given
  module(s). If found, it can return an instance of the class with optional
  keyword arguments provided in `kwargs`. If no class is found, an error is
  raised.

  :param modules: A module or a list of modules to search for the class.
  :type modules: list or module
  :param name: The name of the class to retrieve.
  :type name: str, optional
  :param kwargs: Optional keyword arguments to pass when initializing the
                 class (it can contain the name of the class if 'name'
                 is not provided).
  :type kwargs: dict, optional

  :return: An instance of the class if found, or the class itself.
  :rtype: object or class
  """
  # Check class name
  if ((name is None) and (kwargs is not None)):
    if ("name" in kwargs.keys()):
      name = kwargs.pop("name")
    else:
      raise ValueError("Class name not provided.")
  # Loop over modules to find class
  if (not isinstance(modules, (list, tuple))):
    modules = [modules]
  for module in modules:
    members = inspect.getmembers(module, inspect.isclass)
    for (name_i, cls_i) in members:
      if (name_i == name):
        if (kwargs is not None):
          return cls_i(**kwargs)
        else:
          return cls_i
  # Raise error if class not found
  names = [module.__name__ for module in modules]
  raise ValueError(f"Class `{name}` not found in modules: {names}.")

def check_path(path: str) -> None:
  """
  Check if the specified path exists.

  :param path: The path to check.
  :type path: str

  :return: None
  :rtype: None

  :raises IOError: If the path does not exist.
  """
  if (not os.path.exists(path)):
    raise IOError(f"Path '{path}' does not exist.")

# Data
# =====================================
def save_case(
  path: str,
  index: int,
  data: Any,
  filename: Optional[str] = None
) -> None:
  """
  Save a simulation case to disk using `dill`.

  :param path: Output directory.
  :type path: str
  :param index: Integer index for file naming.
  :type index: int
  :param data: Data to be serialized and saved.
  :type data: Any
  :param filename: Optional explicit filename. If None, uses `case_{index}.p`.
  :type filename: str, optional
  """
  if (filename is None):
    filename = path + f"/case_{str(index).zfill(4)}.p"
  with open(filename, "wb") as file:
    pickle.dump(data, file)

def load_case(
  path: Optional[str] = None,
  index: Optional[int] = 0,
  key: Optional[str] = None,
  filename: Optional[str] = None
) -> Any:
  """
  Load a simulation case from file, optionally extracting a specific field.

  :param path: Directory containing the case.
  :type path: str, optional
  :param index: Index to construct the filename.
  :type index: int, optional
  :param key: If provided, returns only the specified key from the saved object.
  :type key: str, optional
  :param filename: Explicit path to the file. Overrides `path` and `index`.
  :type filename: str, optional

  :return: Loaded data or field from the case.
  :rtype: Any
  """
  if (filename is None):
    filename = path + f"/case_{str(index).zfill(4)}.p"
  if os.path.exists(filename):
    with open(filename, "rb") as file:
      data = pickle.load(file)
    if (key is None):
      return data
    else:
      return data[key]

def load_case_parallel(
  path: str,
  irange: List[int],
  key: Optional[str] = None,
  nb_workers: int = 1,
  desc: Optional[str] = "Cases",
  delimiter: str = "  ",
  verbose: bool = True
) -> List[Any]:
  """
  Load multiple cases in parallel or sequentially.

  :param path: Directory containing cases.
  :type path: str
  :param irange: Index range [start, stop) of case files to load.
  :type irange: List[int]
  :param key: Optional key to extract from each case.
  :type key: str, optional
  :param nb_workers: Number of parallel workers (default: 1).
  :type nb_workers: int
  :param desc: Description for progress bar.
  :type desc: str, optional
  :param delimiter: Prefix spacing for the progress bar.
  :type delimiter: str, optional
  :param verbose: Whether to display progress.
  :type verbose: bool

  :return: List of loaded cases.
  :rtype: List[Any]
  """
  irange = np.sort(irange)
  iterable = tqdm(
    iterable=range(*irange),
    ncols=80,
    desc=delimiter+desc if (desc is not None) else None,
    file=sys.stdout,
    disable=(not verbose)
  )
  if (nb_workers > 1):
    return jl.Parallel(nb_workers)(
      jl.delayed(load_case)(path=path, index=i, key=key) for i in iterable
    )
  else:
    return [load_case(path=path, index=i, key=key) for i in iterable]

def generate_case_parallel(
  sol_fun: callable,
  irange: List[int],
  sol_kwargs: Dict[str, Any] = {},
  nb_workers: int = 1,
  desc: Optional[str] = "Cases",
  verbose: bool = True,
  delimiter: str = "  "
) -> float:
  """
  Run a solver in parallel over a range of indices, collecting convergence
  statistics.

  :param sol_fun: Solver function. Must return 0 or 1 to indicate convergence.
  :type sol_fun: callable
  :param irange: Index range [start, stop) for sample IDs.
  :type irange: List[int]
  :param sol_kwargs: Keyword arguments for the solver function.
  :type sol_kwargs: Dict[str, Any]
  :param nb_workers: Number of parallel workers.
  :type nb_workers: int
  :param desc: Description for progress bar.
  :type desc: str, optional
  :param verbose: Whether to print convergence statistics.
  :type verbose: bool
  :param delimiter: Prefix for progress output.
  :type delimiter: str, optional

  :return: Mean convergence rate across all runs.
  :rtype: float

  This function uses `joblib` for parallel processing and `tqdm` for showing
  a progress bar. It applies the `sol_fun` function to a range of sample
  indices and collects convergence results. If `verbose` is True, it prints
  the total number of converged cases.
  """
  irange = np.sort(irange)
  iterable = tqdm(
    iterable=range(*irange),
    ncols=80,
    desc=delimiter+desc if (desc is not None) else None,
    file=sys.stdout
  )
  if (nb_workers > 1):
    # Define parallel function
    sol_fun = env.make_fun_parallel(sol_fun)
    # Run parallel jobs
    runtime = jl.Parallel(nb_workers)(
      jl.delayed(sol_fun)(index=i, **sol_kwargs) for i in iterable
    )
  else:
    runtime = [sol_fun(index=i, **sol_kwargs) for i in iterable]
  runtime = [rt for rt in runtime if (rt is not None)]
  if verbose:
    nb_samples = irange[1]-irange[0]
    print(delimiter + f"Total converged cases: {len(runtime)}/{nb_samples}")
  return np.mean(runtime)

# Operations
# =====================================
def map_nested_dict(
  fun: callable,
  obj: Any
) -> Any:
  """
  Recursively apply a function to all values in a nested dictionary.

  This function traverses a nested dictionary and applies the given
  function to each value. It supports dictionaries, lists, and tuples.

  :param fun: The function to apply to each value.
  :type fun: Callable[[Any], Any]
  :param obj: The nested dictionary or other container to map.
  :type obj: dict or list or tuple or Any

  :return: A new nested structure with the function applied to all values.
  :rtype: Any
  """
  if isinstance(obj, collections.Mapping):
    return {k: map_nested_dict(fun, v) for (k, v) in obj.items()}
  else:
    if isinstance(obj, (list, tuple)):
      return [fun(x) for x in obj]
    else:
      return fun(obj)

def is_nan_inf(x: np.ndarray) -> np.ndarray:
  """
  Check whether entries in an array are NaN or Inf.

  :param x: Input array.
  :type x: np.ndarray

  :return: Boolean array of same shape.
  :rtype: np.ndarray
  """
  return (np.isnan(x)+np.isinf(x)).astype(bool)

def replace_keys(
  d: dict,
  key_map: Dict[str, str]
) -> dict:
  """
  Replace keys in a (possibly nested) dictionary using a mapping.

  :param d: Input dictionary.
  :type d: dict
  :param key_map: Dictionary mapping old keys to new keys.
  :type key_map: Dict[str, str]

  :return: Dictionary with keys replaced.
  :rtype: dict
  """
  new_d = {}
  for (k, v) in d.items():
    if isinstance(v, dict):
      v = replace_keys(v, key_map)
    new_key = key_map.get(k, k)
    new_d[new_key] = v
  return new_d

# Statistics
# =====================================
def absolute_percentage_error(y_true, y_pred, eps=1e-7):
  """
  Compute absolute percentage error.

  :param y_true: Ground truth values.
  :type y_true: np.ndarray
  :param y_pred: Predicted values.
  :type y_pred: np.ndarray
  :param eps: Small number to prevent division by zero.
  :type eps: float

  :return: Absolute percentage error (APE).
  :rtype: np.ndarray
  """
  return 100*np.abs(y_true-y_pred)/(np.abs(y_true)+eps)

def mape(y_true, y_pred, eps=1e-7, axis=0):
  """
  Compute mean absolute percentage error (MAPE).

  :param y_true: Ground truth values.
  :type y_true: np.ndarray
  :param y_pred: Predicted values.
  :type y_pred: np.ndarray
  :param eps: Small value to avoid division by zero.
  :type eps: float
  :param axis: Axis along which to average.
  :type axis: int

  :return: MAPE value.
  :rtype: float
  """
  err = absolute_percentage_error(y_true, y_pred, eps)
  return np.mean(err, axis=axis)

def l2_relative_error(y_true, y_pred, axis=-1, eps=1e-7):
  """
  Compute L2 relative error.

  :param y_true: Ground truth values.
  :type y_true: np.ndarray
  :param y_pred: Predicted values.
  :type y_pred: np.ndarray
  :param axis: Axis over which to compute norms.
  :type axis: int
  :param eps: Small value for numerical stability.
  :type eps: float

  :return: Relative error.
  :rtype: np.ndarray
  """
  err = np.linalg.norm(y_true-y_pred, axis=axis)
  err /= (np.linalg.norm(y_true, axis=axis) + eps)
  return err

# Timeout
# =====================================
class TimeoutException(Exception):
  """
  Custom exception raised when solver execution exceeds allowed time.
  """
  pass

def timeout_handler(signum, frame):
  """
  Signal handler for triggering timeout exception.

  :param signum: Signal number.
  :param frame: Execution frame.
  """
  raise TimeoutException("Solver exceeded the allowed execution time.")

def make_solve_ivp(
  tout: float = 0.0
) -> callable:
  """
  Wrap `scipy.integrate.solve_ivp` with a timeout handler.

  :param tout: Timeout in seconds (0 disables timeout).
  :type tout: float

  :return: A callable wrapper around `solve_ivp`.
  :rtype: callable
  """
  def solve_ivp(*args, **kwargs):
    # Timeout active
    active = (tout > 0.0)
    # Set signal alarm for timeout
    if active:
      signal.signal(signal.SIGALRM, timeout_handler)
      signal.alarm(int(tout))
    try:
      # Call function
      sol = sp.integrate.solve_ivp(*args, **kwargs)
      # Disable alarm after successful execution
      if active:
        signal.alarm(0)
    except:
      # Handle all type of errors
      sol = None
    finally:
      # Ensure alarm is disabled in case of early return
      if active:
        signal.alarm(0)
    return sol
  return solve_ivp
