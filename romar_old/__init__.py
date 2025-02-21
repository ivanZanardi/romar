# Future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Package
__author__ = "Ivan Zanardi"
__email__ = "zanardi3@illinois.edu"
__url__ = "https://github.com/ivanZanardi/romar"
__license__ = "Apache-2.0"
__version__ = "0.0.1"

__all__ = [
  "backend",
  "const",
  "env",
  "ops",
  "postproc",
  "roms",
  "systems",
  "utils"
]

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# Ignore all warnings from R
import rpy2.robjects as ro
ro.r('options(warn=-1)')

# Logging
from absl import logging
logging.set_verbosity(logging.ERROR)
