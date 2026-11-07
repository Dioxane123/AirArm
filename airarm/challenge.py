"""固定末端位姿的可达性采样，以及三目标挑战的纯状态逻辑。"""

from dataclasses import dataclass
import math
import random
from typing import Literal

import mink
import numpy as np

from .arm_control import ArmController, ArmState


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class ChallengeWorkspace:
    low: tuple[float, float]
    high: tuple[float, float]
    height: float
    reachable_points: tuple[Point3, ...]

    @classmethod
    def sample(cls, arm: ArmController, radius: float = 0.015) -> "ChallengeWorkspace":
        """采样当前控制区域内的 IK 解；不把这个局部区域称为完整三维工作空间。"""
        if not math.isfinite(radius) or not 0 < radius < float(np.min(arm.workspace_high - arm.workspace_low)) / 4:
            raise ValueError("Target radius must fit inside the XY workspace (default: 0.015 m).")
        configuration = mink.Configuration(arm.model, arm.data.qpos.copy())
        task = mink.FrameTask("attachment_site", "site", position_cost=1.0,
                              orientation_cost=1.0, lm_damping=1e-3)
        posture = mink.PostureTask(arm.model, cost=1e-4)
        posture.set_target_from_configuration(configuration)
        points: list[Point3] = []
        inset = radius + 0.003
        for x in np.linspace(arm.workspace_low[0] + inset, arm.workspace_high[0] - inset, 5):
            for y in np.linspace(arm.workspace_low[1] + inset, arm.workspace_high[1] - inset, 5):
                target = np.array([x, y, arm.home_xyz[2]])
                configuration.update(arm.data.qpos.copy())
                task.set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(arm.home_rotation), target))
                try:
                    for _ in range(200):
                        velocity = mink.solve_ik(configuration, [task, posture], arm.control_dt,
                                                 "daqp", limits=arm.limits)
                        configuration.integrate_inplace(velocity, arm.control_dt)
                        error = task.compute_error(configuration)
                        if np.linalg.norm(error[:3]) < 0.0005 and np.linalg.norm(error[3:]) < 0.01:
                            points.append((float(x), float(y), float(target[2])))
                            break
                except mink.MinkError:
                    continue
        if len(points) < 3:
            raise ValueError("Not enough reachable target locations were found for the challenge.")
        return cls((float(arm.workspace_low[0]), float(arm.workspace_low[1])),
                   (float(arm.workspace_high[0]), float(arm.workspace_high[1])), float(arm.home_xyz[2]), tuple(points))

    def choose_targets(self, start: Point3, rng: random.Random, radius: float) -> tuple[Point3, ...]:
        chosen: list[Point3] = []
        previous = np.asarray(start[:2])
        for _ in range(3):
            candidates = [point for point in self.reachable_points if point not in chosen
                          and np.linalg.norm(np.asarray(point[:2]) - previous) >= 2 * radius + 0.01]
            if not candidates:
                raise ValueError("The workspace is too small to separate three target regions.")
            selected = rng.choice(candidates)
            chosen.append(selected)
            previous = np.asarray(selected[:2])
        return tuple(chosen)


class ThreeTargetChallenge:
    def __init__(self, targets: tuple[Point3, ...], *, radius: float = 0.015,
                 height_tolerance: float = 0.025, dwell_seconds: float = 0.5) -> None:
        if len(targets) != 3 or not np.all(np.isfinite(targets)):
            raise ValueError("A challenge needs exactly three finite XYZ targets.")
        for value in (radius, height_tolerance, dwell_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Challenge tolerances must be finite and positive.")
        self.targets = targets
        self.radius = radius
        self.height_tolerance = height_tolerance
        self.dwell_seconds = dwell_seconds
        self.status: Literal["ready", "running", "complete"] = "ready"
        self.index = 0
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.inside_since: float | None = None
        self.hit_times: list[float] = []

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    @property
    def current_target(self) -> Point3 | None:
        return None if self.complete else self.targets[self.index]

    def start(self, now: float) -> None:
        if not math.isfinite(now):
            raise ValueError("Challenge clock must be finite.")
        if self.status == "ready":
            self.started_at = now
            self.status = "running"

    def elapsed(self, now: float) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, (self.finished_at if self.finished_at is not None else now) - self.started_at)

    def update(self, state: ArmState, now: float, *, enabled: bool) -> bool:
        """返回本次是否命中；只有实际位置、有效控制及连续停留都满足才计数。"""
        if self.status != "running":
            return False
        if not math.isfinite(now) or self.started_at is None or now < self.started_at:
            raise ValueError("Challenge clock moved backwards or became invalid.")
        target = self.targets[self.index]
        actual = np.asarray(state.actual_xyz)
        inside = (
            enabled and not state.paused and bool(np.all(np.isfinite(actual)))
            and float(np.linalg.norm(actual[:2] - target[:2])) <= self.radius
            and abs(float(actual[2]) - target[2]) <= self.height_tolerance
        )
        if not inside:
            self.inside_since = None
            return False
        if self.inside_since is None:
            self.inside_since = now
        if now - self.inside_since + 1e-9 < self.dwell_seconds:
            return False
        self.hit_times.append(self.elapsed(now))
        self.index += 1
        self.inside_since = None
        if self.index == 3:
            self.status = "complete"
            self.finished_at = now
        return True
