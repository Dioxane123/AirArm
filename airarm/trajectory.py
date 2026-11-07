"""版本化轨迹文件：完整初始物理状态 + 每个控制周期的命令与实际轨迹。"""

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import cast
from uuid import uuid4

import mujoco
import numpy as np

from .arm_control import ArmController, ArmControlError, JOINT_NAMES
from .challenge import ChallengeWorkspace, Point3, ThreeTargetChallenge


STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def scene_fingerprint(scene: Path) -> str:
    """对场景及其模型资源内容取摘要，路径移动到其他机器不影响摘要。"""
    digest = hashlib.sha256()
    files = sorted(p for p in scene.parent.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".xml", ".obj", ".stl", ".msh", ".png", ".jpg"})
    for path in files:
        digest.update(path.relative_to(scene.parent).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TrajectoryFrame:
    wall_time: float
    sim_time: float
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    ctrl: tuple[float, ...]
    actual_xyz: Point3
    target_xyz: Point3
    target_index: int


def point3(values: object, name: str) -> Point3:
    result = numbers(values, 3, name)
    return (result[0], result[1], result[2])


def numbers(value: object, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError(f"{name} contains invalid numbers.")
    return tuple(float(v) for v in value)


def number(value: object, name: str) -> float:
    return numbers([value], 1, name)[0]


def mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object.")
    return cast(dict[str, object], value)


class TrajectoryRecorder:
    def __init__(self, arm: ArmController, workspace: ChallengeWorkspace,
                 challenge: ThreeTargetChallenge, now: float) -> None:
        self.started_at = now
        self.sim_started_at = float(arm.data.time)
        self.initial_state = np.zeros(mujoco.mj_stateSize(arm.model, STATE_SPEC))
        mujoco.mj_getState(arm.model, arm.data, self.initial_state, STATE_SPEC)
        self.workspace = workspace
        self.challenge = challenge
        self.model_digest = scene_fingerprint(arm.scene)
        self.control_dt = arm.control_dt
        self.physics_dt = float(arm.model.opt.timestep)
        self.frames: list[TrajectoryFrame] = []
        self.capture(arm, now)

    def capture(self, arm: ArmController, now: float) -> None:
        wall = now - self.started_at
        sim = float(arm.data.time) - self.sim_started_at
        if not math.isfinite(wall) or wall < 0:
            raise ValueError("Recording clock is invalid.")
        if self.frames and (wall < self.frames[-1].wall_time or sim <= self.frames[-1].sim_time):
            raise ValueError("Recording frames must advance simulation time and preserve wall-clock order.")
        s = arm.state()
        self.frames.append(TrajectoryFrame(
            wall, sim, tuple(float(v) for v in arm.data.qpos), tuple(float(v) for v in arm.data.qvel),
            tuple(float(v) for v in arm.data.ctrl), point3(s.actual_xyz, "actual_xyz"),
            point3(s.target_xyz, "target_xyz"), min(self.challenge.index, 2),
        ))

    def save(self, directory: Path) -> Path:
        if not self.challenge.complete:
            raise ValueError("Only completed challenges are saved as replayable recordings.")
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"airarm-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.json"
        destination = directory / filename
        payload = {
            "format": "airarm.trajectory", "version": 1, "mujoco_version": mujoco.__version__,
            "scene_fingerprint": self.model_digest, "joint_names": list(JOINT_NAMES),
            "state_spec": STATE_SPEC, "initial_state": self.initial_state.tolist(),
            "control_dt": self.control_dt, "physics_dt": self.physics_dt,
            "workspace": asdict(self.workspace), "targets": self.challenge.targets,
            "target_radius": self.challenge.radius, "height_tolerance": self.challenge.height_tolerance,
            "dwell_seconds": self.challenge.dwell_seconds, "completed": True,
            "elapsed_seconds": self.challenge.elapsed(self.started_at),
            "hit_times": self.challenge.hit_times, "frames": [asdict(frame) for frame in self.frames],
        }
        temporary = destination.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


class TrajectoryReplay:
    def __init__(self, path: Path, arm: ArmController) -> None:
        self.arm = arm
        raw = mapping(json.loads(path.read_text(encoding="utf-8")), "recording")
        if raw.get("format") != "airarm.trajectory" or raw.get("version") != 1 or raw.get("completed") is not True:
            raise ValueError("Unsupported or incomplete trajectory recording.")
        if raw.get("mujoco_version") != mujoco.__version__:
            raise ValueError("Replay requires the recorded MuJoCo version.")
        if raw.get("scene_fingerprint") != scene_fingerprint(arm.scene) or raw.get("joint_names") != list(JOINT_NAMES):
            raise ValueError("Trajectory does not match this robot model or joint order.")
        if raw.get("state_spec") != STATE_SPEC:
            raise ValueError("Trajectory uses a different state format.")
        self.control_dt = number(raw.get("control_dt"), "control_dt")
        physics_dt = number(raw.get("physics_dt"), "physics_dt")
        if not math.isclose(self.control_dt, arm.control_dt, abs_tol=1e-12) or not math.isclose(physics_dt, arm.model.opt.timestep, abs_tol=1e-12):
            raise ValueError("Trajectory and robot timesteps do not match.")
        self.initial_state = np.array(numbers(raw.get("initial_state"), mujoco.mj_stateSize(arm.model, STATE_SPEC), "initial_state"))
        workspace = mapping(raw.get("workspace"), "workspace")
        points = workspace.get("reachable_points")
        targets = raw.get("targets")
        frames = raw.get("frames")
        if not isinstance(points, list) or not isinstance(targets, list) or len(targets) != 3 or not isinstance(frames, list) or len(frames) < 2:
            raise ValueError("Trajectory is missing workspace, three targets, or frames.")
        low = numbers(workspace.get("low"), 2, "workspace.low")
        high = numbers(workspace.get("high"), 2, "workspace.high")
        self.workspace = ChallengeWorkspace((low[0], low[1]), (high[0], high[1]),
            number(workspace.get("height"), "workspace.height"), tuple(point3(p, "reachable point") for p in points))
        self.targets = tuple(point3(p, "target") for p in targets)
        self.radius = number(raw.get("target_radius"), "target_radius")
        self.elapsed_seconds = number(raw.get("elapsed_seconds"), "elapsed_seconds")
        if not 0 < self.radius < 0.1 or self.elapsed_seconds <= 0:
            raise ValueError("Invalid target radius or elapsed time.")
        hit_times = numbers(raw.get("hit_times"), 3, "hit_times")
        if not 0 < hit_times[0] < hit_times[1] < hit_times[2] or not math.isclose(hit_times[-1], self.elapsed_seconds, abs_tol=1e-7):
            raise ValueError("Completion times are inconsistent.")
        self.frames: list[TrajectoryFrame] = []
        previous_wall = -1.0
        for i, item in enumerate(frames):
            row = mapping(item, "frame")
            wall = number(row.get("wall_time"), "wall_time")
            sim = number(row.get("sim_time"), "sim_time")
            index = row.get("target_index")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 3:
                raise ValueError("Invalid frame target index.")
            if wall < previous_wall or wall < 0 or not math.isclose(sim, i * self.control_dt, rel_tol=0, abs_tol=1e-7):
                raise ValueError("Trajectory timestamps are out of order or have missing physics steps.")
            if i == 0 and not math.isclose(wall, 0.0, abs_tol=1e-9):
                raise ValueError("The first recording frame must start at zero.")
            ctrl = numbers(row.get("ctrl"), arm.model.nu, "ctrl")
            if np.any(np.asarray(ctrl)[arm.actuator_ids] < arm.lower) or np.any(np.asarray(ctrl)[arm.actuator_ids] > arm.upper):
                raise ValueError("Recorded actuator commands exceed limits.")
            self.frames.append(TrajectoryFrame(wall, sim,
                numbers(row.get("qpos"), arm.model.nq, "qpos"), numbers(row.get("qvel"), arm.model.nv, "qvel"),
                ctrl, point3(row.get("actual_xyz"), "actual_xyz"), point3(row.get("target_xyz"), "target_xyz"), index))
            previous_wall = wall
        if not math.isclose(self.frames[-1].wall_time, self.elapsed_seconds, abs_tol=1e-7):
            raise ValueError("The recording does not end at challenge completion.")
        self.index = 0
        self.max_joint_error = 0.0
        self.reset()

    @property
    def complete(self) -> bool:
        return self.index == len(self.frames) - 1

    def reset(self) -> None:
        mujoco.mj_setState(self.arm.model, self.arm.data, self.initial_state, STATE_SPEC)
        mujoco.mj_forward(self.arm.model, self.arm.data)
        self.arm.paused = False
        self.index = 0
        self.max_joint_error = 0.0
        if not np.allclose(self.arm.data.qpos, self.frames[0].qpos, rtol=0, atol=1e-9):
            raise ValueError("Initial physics state does not match the first recorded frame.")
        self._visual_target(self.frames[0])

    def _visual_target(self, frame: TrajectoryFrame) -> None:
        self.arm.target_xyz = np.array(frame.target_xyz)
        self.arm.data.mocap_pos[self.arm.marker_id] = frame.target_xyz
        reference = np.asarray(frame.qpos).copy()
        reference[self.arm.qpos_ids] = np.asarray(frame.ctrl)[self.arm.actuator_ids]
        self.arm.configuration.update(reference)

    def step(self) -> TrajectoryFrame:
        if self.complete:
            return self.frames[self.index]
        frame = self.frames[self.index + 1]
        self.arm.data.ctrl[:] = frame.ctrl
        self._visual_target(frame)
        mujoco.mj_step(self.arm.model, self.arm.data, nstep=self.arm.physics_steps)
        mujoco.mj_forward(self.arm.model, self.arm.data)
        error = float(np.max(np.abs(self.arm.data.qpos - frame.qpos)))
        if not np.all(np.isfinite(self.arm.data.qpos)) or error > 0.02:
            raise ArmControlError("Replay physics diverged from the recorded trajectory.")
        self.max_joint_error = max(self.max_joint_error, error)
        self.index += 1
        self.arm.paused = self.complete
        return frame
