# typings/skimage/transform.pyi
import numpy.typing as npt
import numpy as np
from typing import Any

class ProjectiveTransform:
    def __init__(self, matrix: npt.NDArray[np.float64] | None = ...) -> None: ...

def warp(
    image: npt.NDArray[np.float64],
    inverse_map: Any,
    *,
    output_shape: tuple[int, int] | None = ...,
    **kwargs: Any,
) -> npt.NDArray[np.float64]: ...