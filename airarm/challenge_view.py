"""机械臂区域中的工作平面、目标圈、实际轨迹和挑战结果。"""

from collections.abc import Sequence

import mujoco
import numpy as np

from .challenge import ChallengeWorkspace, Point3
from .data_types import ImageArray


def add_scene_overlay(scene: mujoco.MjvScene, workspace: ChallengeWorkspace,
                      target: Point3 | None, radius: float, path: Sequence[Point3]) -> None:
    def line(a: Point3, b: Point3, color: tuple[float, float, float, float], width: float) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                          np.eye(3).ravel(), np.array(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.array(a), np.array(b))
        scene.ngeom += 1

    lo, hi, z = workspace.low, workspace.high, workspace.height
    corners: list[Point3] = [(lo[0], lo[1], z), (hi[0], lo[1], z),
                             (hi[0], hi[1], z), (lo[0], hi[1], z)]
    for i in range(4):
        line(corners[i], corners[(i + 1) % 4], (0.0, 0.8, 1.0, 0.8), 0.0012)
    if target is not None:
        for i in range(32):
            a, b = 2 * np.pi * i / 32, 2 * np.pi * (i + 1) / 32
            line((target[0] + radius * np.cos(a), target[1] + radius * np.sin(a), z),
                 (target[0] + radius * np.cos(b), target[1] + radius * np.sin(b), z),
                 (1.0, 0.65, 0.05, 1.0), 0.0015)
    # 完整记录保留所有点，显示时限制线段数，避免长挑战拖慢窗口。
    stride = max(1, len(path) // 250)
    shown = list(path[::stride])
    if path and (not shown or shown[-1] != path[-1]):
        shown.append(path[-1])
    for a, b in zip(shown, shown[1:]):
        if np.linalg.norm(np.asarray(a) - b) > 1e-5:
            line(a, b, (0.2, 1.0, 0.4, 0.9), 0.0008)


def draw_challenge_hud(frame: ImageArray, workspace: ChallengeWorkspace, targets: tuple[Point3, ...],
                       index: int, radius: float, path: Sequence[Point3], actual: Point3,
                       elapsed: float, dwell_fraction: float, *, complete: bool = False,
                       saved_name: str | None = None, replay: bool = False) -> None:
    import cv2

    # XY 俯视小图让狭小工作区域中的目标可清楚辨认，不会被机械臂遮挡。
    x0, y0, size = 405, 365, 215
    cv2.rectangle(frame, (x0 - 10, y0 - 30), (635, 605), (25, 32, 35), -1)
    cv2.putText(frame, "Reachable XY (fixed pose)", (x0 - 7, y0 - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1)

    def pixel(point: Point3) -> tuple[int, int]:
        u = (workspace.high[1] - point[1]) / (workspace.high[1] - workspace.low[1])
        v = (workspace.high[0] - point[0]) / (workspace.high[0] - workspace.low[0])
        return x0 + round(np.clip(u, 0, 1) * size), y0 + round(np.clip(v, 0, 1) * size)

    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (230, 190, 0), 1)
    for p in workspace.reachable_points:
        cv2.circle(frame, pixel(p), 2, (90, 95, 90), -1)
    if len(path) > 1:
        points = np.array([pixel(p) for p in path[::max(1, len(path) // 500)]], dtype=np.int32)
        cv2.polylines(frame, [points], False, (110, 230, 110), 1)
    if index < len(targets):
        target = targets[index]
        r = round(radius / (workspace.high[1] - workspace.low[1]) * size)
        cv2.circle(frame, pixel(target), r, (0, 180, 255), 2)
        cv2.putText(frame, str(index + 1), pixel(target), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1)
    cv2.circle(frame, pixel(actual), 5, (255, 140, 255), -1)
    cv2.rectangle(frame, (0, 128), (639, 177), (25, 25, 25), -1)
    title = f"{'REPLAY' if replay else 'CHALLENGE'}  {min(index, 3)}/3  |  {elapsed:.2f} s"
    cv2.putText(frame, title, (10, 149), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 230, 255), 1)
    cv2.putText(frame, f"Stay inside the circle: {min(100, round(dwell_fraction * 100))}%",
                (10, 171), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)
    if complete:
        cv2.rectangle(frame, (55, 220), (585, 420), (30, 40, 32), -1)
        cv2.rectangle(frame, (55, 220), (585, 420), (110, 230, 110), 2)
        cv2.putText(frame, "REPLAY FINISHED" if replay else "3 / 3 COMPLETE!", (115, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (110, 240, 110), 2)
        cv2.putText(frame, f"Time: {elapsed:.2f} s", (160, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if saved_name:
            cv2.putText(frame, "Saved: " + saved_name[-53:], (70, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
        for left, right, label in [(100, 300, "R: Replay" if replay else "R: Retry"), (340, 540, "Q: Quit")]:
            cv2.rectangle(frame, (left, 350), (right, 395), (65, 100, 70), -1)
            cv2.putText(frame, label, (left + 23, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)


def completion_action(x: int, y: int, *, panel_offset: int = 640) -> str | None:
    """完成页按钮位置，默认位于并排窗口的右半边。"""
    x -= panel_offset
    if 350 <= y <= 395:
        if 100 <= x <= 300:
            return "retry"
        if 340 <= x <= 540:
            return "quit"
    return None
