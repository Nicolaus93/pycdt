from typing import TypeAlias
from numpy.typing import NDArray
import numpy as np

EPS = 1e-6
Vec2d: TypeAlias = tuple[float, float] | NDArray[np.floating]
Triangle: TypeAlias = tuple[Vec2d, Vec2d, Vec2d] | NDArray[np.floating]
