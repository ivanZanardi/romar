__all__ = [
  "BoxAd",
  "BoxIso"
]

from .box_ad import BoxAd
from .box_iso import BoxIso

# Data types
from typing import Union
SYS_TYPES = Union[BoxAd, BoxIso]
