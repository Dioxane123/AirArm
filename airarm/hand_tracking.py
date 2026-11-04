"""把每一帧的手部关键点转换成可检查的二维输入。

坐标约定：左上角为 (0, 0)，右下角为 (1, 1)，x 向右，y 向下。
这里不缓存上一次有效位置，也不控制机械臂。
"""

from dataclasses import dataclass
import math
from typing import Sequence


Point = tuple[float, float]
PALM_IDS = (0, 5, 9, 13, 17)  # 手腕，以及食指、中指、无名指、小指的指根。
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass(frozen=True)
class HandObservation:
    timestamp_ms: int
    status: str
    detected_hands: int
    palm_xy: Point | None = None
    landmarks_xy: tuple[Point, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status == "TRACKING"


def observe_hand(
    hands: Sequence[Sequence[Point]], timestamp_ms: int, *, mirror: bool = True
) -> HandObservation:
    """只接受当前帧中的一只手；无手、多手和异常坐标都不输出掌心。"""
    count = len(hands)
    if count == 0:
        return HandObservation(timestamp_ms, "NO_HAND", count)
    if count > 1:
        return HandObservation(timestamp_ms, "MULTIPLE_HANDS", count)

    points = hands[0]
    if len(points) != 21 or any(
        len(point) != 2 or not all(math.isfinite(value) for value in point)
        for point in points
    ):
        return HandObservation(timestamp_ms, "INVALID_HAND", count)

    # 对坐标进行与显示画面相同的一次水平镜像，不改变 y 的方向。
    displayed = tuple((1.0 - x if mirror else x, y) for x, y in points)
    palm = (
        sum(displayed[index][0] for index in PALM_IDS) / len(PALM_IDS),
        sum(displayed[index][1] for index in PALM_IDS) / len(PALM_IDS),
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in palm):
        return HandObservation(timestamp_ms, "OUT_OF_FRAME", count)
    return HandObservation(timestamp_ms, "TRACKING", count, palm, displayed)


def next_timestamp_ms(seconds: float, previous: int) -> int:
    """MediaPipe 的视频模式要求时间戳递增，快速处理也不能复用毫秒值。"""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Time must be finite and non-negative.")
    return max(previous + 1, int(seconds * 1000))
