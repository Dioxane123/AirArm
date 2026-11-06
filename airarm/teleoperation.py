"""手掌二维观测 → 对齐、平滑、限速 → 机械臂 XY 目标。

此模块不操作摄像头或窗口。时间使用同一单调时钟，便于检查旧输入。
"""

from dataclasses import dataclass
import math

import numpy as np

from .arm_control import ArmController, ArmState
from .data_types import FloatArray
from .hand_tracking import HandObservation


INPUT_PAUSE_REASONS: frozenset[str] = frozenset({
    "WAITING_FOR_HAND", "NO_HAND", "MULTIPLE_HANDS", "INVALID_HAND",
    "OUT_OF_FRAME", "STALE_INPUT", "INVALID_TIME", "TRACKING_INTERRUPTED",
})


@dataclass(frozen=True)
class TeleopConfig:
    gain: float = 0.20           # 米 / 归一化图像距离。
    smoothing_seconds: float = 0.12
    max_target_speed: float = 0.10  # 米 / 秒；限制笛卡尔目标，不是实测末端速度。
    stale_seconds: float = 0.30

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")


class HandTeleopController:
    def __init__(self, arm: ArmController, config: TeleopConfig | None = None) -> None:
        self.arm: ArmController = arm
        self.config: TeleopConfig = config or TeleopConfig()
        self.enabled: bool = False
        self.manual_paused: bool = False
        self._resume_on_valid_input: bool = False
        self.reason: str = "WAITING_FOR_HAND"
        self.observation: HandObservation | None = None
        self.captured_at: float | None = None
        self.hand_anchor: FloatArray | None = None
        self.arm_anchor: FloatArray | None = None
        self.filtered_xy: FloatArray | None = None
        self.command_xy: FloatArray | None = None
        self.enable_count: int = 0
        self.active_steps: int = 0
        self.loss_count: int = 0
        self.arm.pause()

    def age(self, now: float) -> float | None:
        return None if self.captured_at is None else now - self.captured_at

    def _input_problem(self, now: float) -> str | None:
        if not math.isfinite(now):
            return "INVALID_TIME"
        if self.observation is None or self.captured_at is None:
            return "WAITING_FOR_HAND"
        age = now - self.captured_at
        if age < 0 or age >= self.config.stale_seconds:
            return "STALE_INPUT"
        obs = self.observation
        if not obs.valid:
            return obs.status
        if obs.detected_hands != 1 or obs.palm_xy is None:
            return "INVALID_HAND"
        palm = np.asarray(obs.palm_xy, dtype=float)
        if palm.shape != (2,) or not np.all(np.isfinite(palm)) or np.any(palm < 0) or np.any(palm > 1):
            return "INVALID_HAND"
        return None

    def update(self, observation: HandObservation, captured_at: float, now: float, *, loss_count: int = 0) -> bool:
        """接收新帧；检测恢复后自动继续，但不会解除空格造成的手动暂停。"""
        if not math.isfinite(captured_at) or not math.isfinite(now) or captured_at > now:
            self.observation = None
            self.disable("INVALID_TIME")
            return False
        if self.captured_at is not None and captured_at <= self.captured_at:
            return False
        if self.enabled and loss_count > self.loss_count:
            self.disable("TRACKING_INTERRUPTED")
        if self.enabled and self.captured_at is not None and now - self.captured_at >= self.config.stale_seconds:
            self.disable("STALE_INPUT")
        self.observation = observation
        self.captured_at = captured_at
        self.loss_count = loss_count
        problem = self._input_problem(now)
        if problem is not None:
            self.disable(problem)
        elif not self.enabled and self._resume_on_valid_input and not self.manual_paused:
            # 恢复时重新对齐当前手位与实际机械臂位置，不追赶丢手前的目标。
            self.enable(now)
        elif not self.enabled and self.reason == "WAITING_FOR_HAND":
            self.reason = "PRESS_SPACE"
        return True

    def enable(self, now: float) -> bool:
        if self.manual_paused:
            return False
        problem = self._input_problem(now)
        if problem is not None:
            self.disable(problem)
            return False
        # _input_problem 已验证输入；显式断言让类型检查器也能收窄 Optional。
        assert self.observation is not None and self.observation.palm_xy is not None
        # 每次重新启用都使用当前实际 XY，丢弃上次的手位基准和过滤器历史。
        self.hand_anchor = np.asarray(self.observation.palm_xy, dtype=float).copy()
        self.arm_anchor = np.clip(
            np.asarray(self.arm.state().actual_xyz[:2]),
            self.arm.workspace_low, self.arm.workspace_high,
        )
        self.filtered_xy = self.arm_anchor.copy()
        self.command_xy = self.arm_anchor.copy()
        self.arm.set_target(self.command_xy)
        self.arm.resume()
        self.enabled = True
        self._resume_on_valid_input = False
        self.reason = "ACTIVE"
        self.enable_count += 1
        return True

    def disable(self, reason: str = "PAUSED") -> None:
        resume_requested = self.enabled or self._resume_on_valid_input
        if reason == "PAUSED":
            self.manual_paused = True
        self._resume_on_valid_input = (
            reason in INPUT_PAUSE_REASONS and resume_requested and not self.manual_paused
        )
        if self.enabled or not self.arm.paused:
            self.arm.pause()  # 仅在状态改变时保持，不能每帧重设而不断下沉。
        self.enabled = False
        # 后续的丢手、超时等事件不能覆盖手动暂停的来源。
        self.reason = "PAUSED" if self.manual_paused and reason != "STOPPED" else reason
        self.hand_anchor = self.arm_anchor = self.filtered_xy = self.command_xy = None

    def toggle(self, now: float) -> None:
        if self.manual_paused:
            self.manual_paused = False
            self._resume_on_valid_input = True
            self.enable(now)
        elif self.enabled or self._resume_on_valid_input:
            self.disable()
        else:
            # 首次启用时若暂时没有有效手，等待下一帧正常检测即可。
            self._resume_on_valid_input = True
            self.enable(now)

    def step(self, now: float) -> ArmState:
        problem = self._input_problem(now)
        if self.enabled and problem is not None:
            self.disable(problem)
        if self.enabled:
            # enabled 只在 enable() 初始化完全部基准后才设为 True。
            assert self.observation is not None and self.observation.palm_xy is not None
            assert self.hand_anchor is not None and self.arm_anchor is not None
            assert self.filtered_xy is not None and self.command_xy is not None
            delta = np.asarray(self.observation.palm_xy) - self.hand_anchor
            # 屏幕上移(y减小) → 世界 +X；屏幕右移(x增加) → 世界 -Y。
            raw_xy = self.arm_anchor + self.config.gain * np.array([-delta[1], -delta[0]])
            bounded_xy = np.clip(raw_xy, self.arm.workspace_low, self.arm.workspace_high)
            dt = self.arm.control_dt
            alpha = -math.expm1(-dt / self.config.smoothing_seconds)
            self.filtered_xy += alpha * (bounded_xy - self.filtered_xy)
            difference = self.filtered_xy - self.command_xy
            distance = float(np.linalg.norm(difference))
            max_distance = self.config.max_target_speed * dt
            self.command_xy += difference * min(1.0, max_distance / max(distance, 1e-12))
            self.arm.set_target(self.command_xy)
            self.active_steps += 1
        return self.arm.step()
