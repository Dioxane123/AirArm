"""跨模块共享的 NumPy 数据类型；数组的具体形状仍由运行时检查。"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
"""双精度坐标/关节数组，例如 XY 的 shape=(2,) 或关节角的 shape=(6,)。"""

ImageArray: TypeAlias = NDArray[np.uint8]
"""uint8 图像；本项目的彩色帧为 shape=(height, width, 3)。"""
