import torch
import functools
import numpy as np
import tensorflow as tf


# Global
# -------------------------------------
_VALID_BKD = {"numpy", "torch"}
_VALID_DEVICE = {"cpu", "cuda"}
_VALID_DTYPE = {"float32", "float64"}

# Setting
# -------------------------------------
def set(
  backend="torch",
  device="cpu",
  device_idx=1,
  nb_threads=4,
  epsilon=None,
  floatx="float64"
):
  set_backend(backend)
  set_device(device, device_idx, nb_threads)
  set_floatx(floatx)
  set_epsilon(epsilon)

def get_backend():
  return _BKD

def set_backend(value="torch"):
  global _BKD
  _BKD = value
  if (value not in _VALID_BKD):
    raise ValueError(
      f"Unknown backend: '{value}'. Valid options are: {_VALID_BKD}"
    )

# Conversion
# -------------------------------------
def to_numpy(x):
  if (x is not None):
    if (torch.is_tensor(x)):
      return x.numpy(force=True)
    elif isinstance(x, (int, float, list, tuple)):
      return np.asarray(x, dtype=floatx("numpy"))
    else:
      return x

def to_torch(x):
  if (x is not None):
    if torch.is_tensor(x):
      return x
    else:
      x = to_numpy(x)
      if isinstance(x, np.ndarray):
        x = torch.as_tensor(x, dtype=floatx("torch"))
      return x

def to_backend(x):
  if (x is not None):
    return to_torch(x) if (_BKD == "torch") else to_numpy(x)

# def make_fun_np(fun_torch):
#   def fun_np(*args, **kwargs):
#     args = [to_torch(x) for x in args]
#     kwargs = {k: to_torch(x) for (k, x) in kwargs.items()}
#     output = tf.nest.flatten(fun_torch(*args, **kwargs))
#     output = [to_numpy(y) for y in output]
#     return output if (len(output) > 1) else output[0]
#   return fun_np

def make_fun_np(fun_torch):
  """
  Decorator to convert function inputs from NumPy to PyTorch tensors and
  convert function outputs from PyTorch tensors to NumPy arrays.
  
  :param fun_torch: Function that operates on PyTorch tensors.
  :type fun_torch: callable
  :return: A wrapped function that accepts NumPy arrays and returns NumPy arrays.
  :type fun_torch: callable
  """
  @functools.wraps(fun_torch)
  def wrapper(*args, **kwargs):
    args   = [to_torch(x) for x in args]
    kwargs = {k: to_torch(x) for (k, x) in kwargs.items()}
    output = tf.nest.flatten(fun_torch(*args, **kwargs))
    output = [to_numpy(y) for y in output]
    return output if len(output) > 1 else output[0]
  return wrapper

# Device
# -------------------------------------
def device():
  return _DEVICE

def set_device(value=None, index=0, nb_threads=4):
  if (value is None):
    value = "cuda" if torch.cuda.is_available() else "cpu"
  if (value not in _VALID_DEVICE):
    raise ValueError(
      f"Unknown device: '{value}'. Valid options are: {_VALID_DEVICE}"
    )
  if (value == "cuda"):
    value += f":{index}"
  global _DEVICE
  _DEVICE = value
  # Set default device
  try:
    torch.set_default_device(torch.device(_DEVICE))
    torch.set_num_interop_threads(nb_threads)
    torch.set_num_threads(nb_threads)
  except:
    pass

# Epsilon
# -------------------------------------
def machine_eps():
  return float(np.finfo(
    {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  ).eps)

def epsilon():
  return _EPSILON

def set_epsilon(value):
  if (value is None):
    value = machine_eps()
  global _EPSILON
  _EPSILON = value

# Float
# -------------------------------------
def floatx(bkd="torch"):
  if (bkd == "torch"):
    return {
      "float16": torch.float16,
      "float32": torch.float32,
      "float64": torch.float64
    }[_FLOATX]
  elif (bkd == "numpy"):
    return {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  else:
    return _FLOATX

def set_floatx(value):
  global _FLOATX
  _FLOATX = value
  if (value not in _VALID_DTYPE):
    raise ValueError(
      f"Unknown dtype: '{value}'. Valid options are: {_VALID_DTYPE}"
    )
  try:
    torch.set_default_dtype(floatx())
  except:
    pass
