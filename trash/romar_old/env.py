import os


def set(
  backend="torch",
  device="cpu",
  device_idx=1,
  nb_threads=4,
  epsilon=None,
  floatx="float64"
):
  nb_threads = int(nb_threads)
  _set_cpu_threads(nb_threads)
  from . import backend as bkd
  global _ENV_OPTS
  _ENV_OPTS = dict(
    backend=backend,
    device=device,
    device_idx=device_idx,
    nb_threads=nb_threads,
    epsilon=epsilon,
    floatx=floatx
  )
  bkd.set(**_ENV_OPTS)

def get():
  return _ENV_OPTS

def make_fun_parallel(fun):
  def fun_parallel(*args, **kwargs):
    set(**_ENV_OPTS)
    return fun(*args, **kwargs)
  return fun_parallel

def _set_cpu_threads(nb_threads):
  keys = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
  )
  for k in keys:
    os.environ[k] = str(nb_threads)
