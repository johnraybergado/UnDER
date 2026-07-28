# typings/laspy/__init__.pyi
import numpy.typing as npt
import numpy as np

__version__: str

class LasData:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]

def read(source: str) -> LasData: ...