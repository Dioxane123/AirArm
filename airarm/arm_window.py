from dataclasses import asdict
from importlib import metadata
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from .arguments import ProgramArgs
from .arm_control import ArmControlError, ArmController
from .data_types import ImageArray


WINDOW = "AirArm - IK and position actuators"
PRESET_OFFSETS = ((0.04, 0.0), (0.04, 0.04), (-0.04, 0.04), (-0.04, -0.04), (0.0, 0.0))


class PresetSequence:
    """在实际末端稳定到位后切换目标；每个目标最多等待 5 秒模拟时间。"""

    def __init__(self, arm: ArmController, *, dwell_seconds: float = 0.2, timeout_seconds: float = 5.0) -> None:
        if not (0 < dwell_seconds < timeout_seconds < float("inf")):
            raise ValueError("Require 0 < dwell_seconds < timeout_seconds.")
        self.targets = [arm.home_xyz[:2] + offset for offset in PRESET_OFFSETS]
        self.index = 0
        self.dwell_seconds = dwell_seconds
        self.timeout_seconds = timeout_seconds
        self.stable_time = 0.0
        self.active_time = 0.0
        self.reached = []
        arm.set_target(self.targets[0])

    @property
    def complete(self) -> bool:
        return self.index == len(self.targets)

    def update(self, arm: ArmController) -> None:
        if self.complete:
            return
        if arm.paused:
            self.stable_time = 0.0
            return
        self.active_time += arm.control_dt
        self.stable_time = self.stable_time + arm.control_dt if arm.at_target() else 0.0
        if self.stable_time + 1e-12 >= self.dwell_seconds:
            self.reached.append({"target_index": self.index + 1, **asdict(arm.state())})
            self.index += 1
            self.stable_time = self.active_time = 0.0
            if not self.complete:
                arm.set_target(self.targets[self.index])
        elif self.active_time >= self.timeout_seconds:
            raise ArmControlError(f"Target {self.index + 1} did not settle within {self.timeout_seconds:g} simulated seconds.")


def handle_key(arm: ArmController, key: int, *, keyboard_control: bool) -> bool:
    """返回是否退出；键盘目标增量用世界坐标，H 是平滑返回 home 目标。"""
    if key in (27, ord("q")):
        return True
    if key == ord(" "):
        arm.resume() if arm.paused else arm.pause()
    if keyboard_control:
        offsets = {ord("w"): (0.01, 0), ord("s"): (-0.01, 0),
                   ord("a"): (0, -0.01), ord("d"): (0, 0.01)}
        if key in offsets:
            arm.nudge(*offsets[key])
        elif key == ord("h"):
            arm.set_target(arm.home_xyz[:2])
    return False


class ArmRenderer:
    """用 OpenCV 显示；无窗口控制不实例化此对象。"""

    def __init__(self, arm: ArmController) -> None:
        self.arm = arm
        self.renderer = mujoco.Renderer(arm.model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(arm.model, self.camera)
        self.camera.azimuth = 120
        self.camera.elevation = -25
        self.camera.distance = 1.7
        self.options = mujoco.MjvOption()
        self.options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.options.sitegroup[1] = 1
        # site 仅用于观测和显示，调整它的外观不改变动力学。
        arm.model.site_rgba[arm.site_id] = [0.1, 1.0, 0.3, 1.0]
        arm.model.site_size[arm.site_id] = [0.018, 0.018, 0.018]
        self.window_open = False

    def frame(self, label: str, *, footer: str = "W/S: X +/- | A/D: Y -/+ | H: home | Space: hold | Q: quit") -> ImageArray:
        import cv2

        arm = self.arm
        self.renderer.update_scene(arm.data, camera=self.camera, scene_option=self.options)
        scene_frame = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        # 把读数放在图像外，避免挡住机械臂本体。
        frame = np.full((640, 640, 3), 25, dtype=np.uint8)
        frame[128:608] = scene_frame
        s = arm.state()
        lines = [
            f"{label} | {'HOLD' if s.paused else 'RUN'} | sim {s.sim_time:.2f}s",
            "Target XYZ m: " + " ".join(f"{v:.3f}" for v in s.target_xyz),
            "Actual XYZ m: " + " ".join(f"{v:.3f}" for v in s.actual_xyz),
            f"Actual error {s.position_error_m * 1000:.1f} mm | IK {s.ik_position_error_m * 1000:.2f} mm",
            f"Orientation error {np.degrees(s.orientation_error_rad):.2f} deg",
        ]
        for row, text in enumerate(lines):
            cv2.putText(frame, text, (10, 22 + row * 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, footer,
                    (8, 628), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)
        return frame

    def show(self, label: str) -> int:
        import cv2

        if not self.window_open:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            self.window_open = True
        cv2.imshow(WINDOW, self.frame(label))
        key = cv2.waitKey(1) & 0xFF
        return ord("q") if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 else key

    def save(self, path: Path, label: str) -> None:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), self.frame(label)):
            raise RuntimeError(f"Could not save snapshot: {path}")

    def close(self) -> None:
        self.renderer.close()
        if self.window_open:
            import cv2
            cv2.destroyAllWindows()


def run_arm(args: ProgramArgs) -> int:
    arm = ArmController(args.scene)
    sequence = PresetSequence(arm) if args.control == "preset" else None
    renderer: ArmRenderer | None = None
    samples = [asdict(arm.state())]
    reason = "unknown"
    error_message: str | None = None
    tick = 0
    started = time.monotonic()
    # 所有模式都按模拟时间限制；无窗口时不按墙钟休眠。
    max_seconds = args.seconds if args.seconds is not None else (30.0 if sequence else None)
    label = "PRESET 1/5" if sequence else "KEYBOARD"
    print("Day 3: position-actuator control. W/S: world X, A/D: world Y, H: home, Space: hold, Q: quit.")
    try:
        if not args.headless:
            renderer = ArmRenderer(arm)
            started = time.monotonic()
        while True:
            if max_seconds is not None and arm.data.time >= max_seconds - 1e-9:
                reason = "time_limit"
                break
            state = arm.step()
            tick += 1
            if sequence is not None:
                reached_before = len(sequence.reached)
                sequence.update(arm)
                if len(sequence.reached) > reached_before:
                    result = sequence.reached[-1]
                    print(f"Reached {result['target_index']}/5: actual error {result['position_error_m'] * 1000:.2f} mm")
                label = f"PRESET {min(sequence.index + 1, 5)}/5"
                if sequence.complete:
                    label = "PRESET COMPLETE"
                    reason = "completed"
                    break
            if tick % 4 == 0:
                samples.append(asdict(state))
                if renderer is not None:
                    if handle_key(arm, renderer.show(label), keyboard_control=sequence is None):
                        reason = "user_exit"
                        break
                    # 25 FPS 显示、100 Hz 控制、500 Hz 物理；硬件不足时允许运行慢于实时。
                    time.sleep(max(0.0, started + tick * arm.control_dt - time.monotonic()))
    except ArmControlError as error:
        reason = "control_error"
        error_message = str(error)
        print(error_message)
    finally:
        if renderer is not None:
            renderer.close()

    final_state = asdict(arm.state())
    samples.append(final_state)
    complete = bool(sequence and sequence.complete)
    passed = reason == "completed" if args.headless else reason != "control_error"
    report = {
        "mode": "arm", "control": args.control, "scene": str(arm.scene),
        "platform": platform.platform(), "architecture": platform.machine(),
        "python": platform.python_version(),
        "versions": {name: metadata.version(name) for name in ("mujoco", "mink", "numpy", "qpsolvers", "daqp")},
        "control_dt": arm.control_dt, "physics_dt": float(arm.model.opt.timestep),
        "max_reference_joint_speed_rad_s": arm.max_joint_speed,
        "tolerances": {"position_m": 0.015, "orientation_rad": 0.04, "ik_position_m": 0.001, "actual_joint_speed_rad_s": 0.05},
        "workspace_xy": {"low": arm.workspace_low.tolist(), "high": arm.workspace_high.tolist()},
        "initial_state": samples[0], "final_state": final_state,
        "samples": samples, "reached_targets": sequence.reached if sequence else [],
        "completed": complete, "exit_reason": reason, "error": error_message,
        "elapsed_wall_seconds": time.monotonic() - started,
        "manual_acceptance": "not_assessed",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved report: {args.report.resolve()}")
    if args.snapshot:
        # 只有显式要求图片时，无窗口模式才需要图形后端。
        screenshot_renderer = ArmRenderer(arm)
        try:
            screenshot_renderer.save(args.snapshot, label)
        finally:
            screenshot_renderer.close()
        print(f"Saved snapshot: {args.snapshot.resolve()}")
    print(f"Stopped: {reason}; simulated {arm.data.time:.2f}s; reached {len(sequence.reached) if sequence else 0}/5")
    return 0 if passed else 1
