"""一轮挑战：首次启用开始计时，每步记录，第三次命中后只保存一次。"""

from pathlib import Path
import random

from .arm_control import ArmController
from .challenge import ChallengeWorkspace, Point3, ThreeTargetChallenge
from .trajectory import TrajectoryRecorder, point3


class ChallengeSession:
    def __init__(self, arm: ArmController, directory: Path, rng: random.Random) -> None:
        self.directory = directory
        self.workspace = ChallengeWorkspace.sample(arm)
        targets = self.workspace.choose_targets(point3(arm.state().actual_xyz, "start"), rng, 0.015)
        self.challenge = ThreeTargetChallenge(targets)
        self.recorder: TrajectoryRecorder | None = None
        self.saved_path: Path | None = None
        self.path: list[Point3] = []

    @property
    def complete(self) -> bool:
        return self.challenge.complete

    def begin_if_enabled(self, arm: ArmController, now: float, enabled: bool) -> None:
        if enabled and self.challenge.status == "ready":
            self.challenge.start(now)
            self.recorder = TrajectoryRecorder(arm, self.workspace, self.challenge, now)
            self.path.append(point3(arm.state().actual_xyz, "actual"))

    def after_step(self, arm: ArmController, now: float, enabled: bool) -> bool:
        if self.recorder is None or self.complete:
            return False
        self.recorder.capture(arm, now)
        self.path.append(point3(arm.state().actual_xyz, "actual"))
        hit = self.challenge.update(arm.state(), now, enabled=enabled)
        if self.complete:
            self.saved_path = self.recorder.save(self.directory)
        return hit
