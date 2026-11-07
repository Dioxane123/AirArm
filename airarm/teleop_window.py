"""把后台手部输入接到主线程的 IK/物理控制，提供并排显示与运行报告。"""

from collections import Counter
from dataclasses import asdict
from importlib import metadata
import json
import platform
import random
import time
from typing import TypedDict

import numpy as np

from .arguments import ProgramArgs
from .arm_control import ArmController
from .arm_window import ArmRenderer
from .hand_input import HandFrame, HandInput
from .hand_tracking import HandObservation
from .data_types import ImageArray
from .teleoperation import HandTeleopController, TeleopConfig
from .challenge_session import ChallengeSession
from .challenge_view import add_scene_overlay, completion_action, draw_challenge_hud
from .trajectory import point3


WINDOW = "AirArm - Hand teleoperation | Space: enable/pause | I: IDs | Q: quit"


class ControlTransition(TypedDict):
    wall_seconds: float
    enabled: bool
    reason: str


def combined_frame(
    renderer: ArmRenderer,
    controller: HandTeleopController,
    packet: HandFrame | None,
    args: ProgramArgs,
    now: float,
    fps: float,
    challenge: ChallengeSession | None = None,
) -> ImageArray:
    import cv2
    from .window import draw_landmarks

    left = np.full((640, 640, 3), 25, dtype=np.uint8)
    age = controller.age(now)
    age_text = "--" if age is None else f"{max(0, age) * 1000:.0f} ms"
    enabled = controller.enabled
    lines = [
        f"{'ACTIVE' if enabled else 'HOLD'} | {controller.reason}",
        "Space: align and enable / pause | I: landmark IDs",
        f"Input age: {age_text} | gain: {controller.config.gain:.2f} m / image",
        "Hand UP -> +X | Hand RIGHT -> -Y | fixed height",
    ]
    for row, text in enumerate(lines):
        cv2.putText(left, text, (10, 23 + row * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (100, 240, 100) if row == 0 and enabled else (235, 235, 235), 1, cv2.LINE_AA)
    if packet is not None:
        view = cv2.flip(packet.frame, 1) if not args.no_mirror else packet.frame.copy()
        observation = packet.observation
        if age is not None and age >= controller.config.stale_seconds:
            # 冻结的画面不能继续显示一个仿佛仍有效的掌心。
            observation = HandObservation(observation.timestamp_ms, "STALE_INPUT", observation.detected_hands)
        view = draw_landmarks(view, observation, mirror=not args.no_mirror,
                              show_ids=args.show_ids, fps=fps, inference_ms=packet.inference_ms)
        h, w = view.shape[:2]
        scale = min(640 / w, 480 / h)
        resized = cv2.resize(view, (round(w * scale), round(h * scale)))
        rh, rw = resized.shape[:2]
        left[128 + (480 - rh) // 2:128 + (480 - rh) // 2 + rh,
             (640 - rw) // 2:(640 - rw) // 2 + rw] = resized
    else:
        cv2.putText(left, "Waiting for camera and model...", (55, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    cv2.putText(left, "Tracking loss auto-resumes. Space pause requires Space to resume.",
                (10, 628), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1)
    if challenge is None:
        right = renderer.frame("HAND TELEOP", footer="Green: actual end effector | Red box: target | Q/Esc: exit")
    else:
        game = challenge.challenge
        right = renderer.frame("THREE TARGETS", footer="Reach orange circles | Green: actual path | Cyan: workspace",
            overlay=lambda scene: add_scene_overlay(scene, challenge.workspace, game.current_target,
                                                     game.radius, challenge.path))
        fraction = 1.0 if game.complete else (0.0 if game.inside_since is None else (now - game.inside_since) / game.dwell_seconds)
        draw_challenge_hud(right, challenge.workspace, game.targets, game.index, game.radius,
                           challenge.path, point3(controller.arm.state().actual_xyz, "actual"),
                           game.elapsed(now), fraction, complete=game.complete,
                           saved_name=challenge.saved_path.name if challenge.saved_path else None)
    return np.hstack((left, right))


def _run_round(args: ProgramArgs, rng: random.Random) -> tuple[bool, int]:
    if not args.model.is_file():
        raise FileNotFoundError("Hand model missing; run scripts/download_hand_model.py.")
    if args.video is not None and not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    arm = ArmController(args.scene)
    config = TeleopConfig(gain=args.gain, smoothing_seconds=args.smoothing,
                         max_target_speed=args.target_speed, stale_seconds=args.stale_seconds)
    controller = HandTeleopController(arm, config)
    challenge = ChallengeSession(arm, args.recordings_dir, rng) if args.mode == "challenge" else None
    source = HandInput(args)
    renderer: ArmRenderer | None = None
    window_open = False
    source_started = False
    source_closed = True
    packet: HandFrame | None = None
    sequence = 0
    frames_observed = 0
    status_counts: Counter[str] = Counter()
    transitions: list[ControlTransition] = []
    samples: list[dict[str, object]] = []
    previous_status: tuple[bool, str] | None = None
    initial_state = arm.state()
    initial = asdict(initial_state)
    auto_used = False
    reason = "unknown"
    error_message: str | None = None
    interrupted = False
    peak_displacement = 0.0
    peak_target_displacement = 0.0
    started = time.monotonic()
    next_tick = next_draw = next_sample = started
    frozen_view: ImageArray | None = None
    clicked_action: str | None = None

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        nonlocal clicked_action
        import cv2
        if event == cv2.EVENT_LBUTTONUP and challenge is not None and challenge.complete:
            clicked_action = completion_action(x, y)

    def record_transition(now: float) -> None:
        nonlocal previous_status
        state = (controller.enabled, controller.reason)
        if state != previous_status:
            item: ControlTransition = {"wall_seconds": now - started, "enabled": state[0], "reason": state[1]}
            transitions.append(item)
            print(f"{item['wall_seconds']:.2f}s: {'ACTIVE' if state[0] else 'HOLD'} - {state[1]}")
            previous_status = state

    def make_renderer() -> ArmRenderer:
        result = ArmRenderer(arm)
        # 与映射方向匹配：世界 -Y 显示为右，+X 显示为前方/上方。
        result.camera.azimuth = 0
        result.camera.elevation = -55
        return result

    try:
        if not args.headless or args.snapshot:
            renderer = make_renderer()
        if not args.headless:
            import cv2
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW, on_mouse)
            window_open = True
        source.start()
        source_started = True
        started = next_tick = next_draw = next_sample = time.monotonic()
        print("Hand teleoperation starts in HOLD. Show one hand, then press Space.")
        while True:
            if challenge is not None and challenge.complete:
                if args.headless:
                    reason = "challenge_complete"
                    break
                import cv2
                assert renderer is not None
                if frozen_view is None:
                    frozen_view = combined_frame(renderer, controller, packet, args, time.monotonic(),
                        frames_observed / max(time.monotonic() - started, 0.001), challenge)
                cv2.imshow(WINDOW, frozen_view)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("r") or clicked_action == "retry":
                    reason = "retry"
                    break
                if key in (27, ord("q")) or clicked_action == "quit" or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    reason = "challenge_complete"
                    break
                continue  # 完成页不再更新手部输入、物理状态或挑战时间。
            latest, finished, input_reason, input_error = source.snapshot()
            now = time.monotonic()
            if args.seconds is not None and now - started >= args.seconds:
                reason = "time_limit"
                break
            if latest is not None and latest.sequence != sequence:
                packet = latest
                sequence = latest.sequence
                frames_observed += 1
                status_counts[packet.observation.status] += 1
                previous_enables = controller.enable_count
                controller.update(packet.observation, packet.captured_at, now, loss_count=packet.loss_count)
                if challenge is not None and controller.enable_count != previous_enables:
                    challenge.challenge.inside_since = None
                record_transition(now)
                # 视频参数只负责首次启用；之后的检测恢复由 update() 处理。
                if args.auto_enable and not auto_used and not controller.manual_paused:
                    auto_used = controller.enabled or controller.enable(now)
                    record_transition(now)
            if finished:
                reason = input_reason or "input_ended"
                error_message = input_error
                break

            if now >= next_tick:
                # 不无限追赶慢帧积压。每批最多推进 40 ms 模拟时间。
                steps = min(4, int((now - next_tick) / arm.control_dt) + 1)
                for _ in range(steps):
                    if challenge is not None:
                        challenge.begin_if_enabled(arm, time.monotonic(), controller.enabled)
                    state = controller.step(time.monotonic())
                    peak_displacement = max(peak_displacement, float(np.linalg.norm(
                        np.asarray(state.actual_xyz[:2]) - np.asarray(initial_state.actual_xyz[:2]))))
                    peak_target_displacement = max(peak_target_displacement, float(np.linalg.norm(
                        np.asarray(state.target_xyz[:2]) - np.asarray(initial_state.target_xyz[:2]))))
                    next_tick += arm.control_dt
                    if challenge is not None:
                        hit = challenge.after_step(arm, time.monotonic(), controller.enabled)
                        if hit:
                            print(f"Challenge target {challenge.challenge.index}/3 reached.")
                        if challenge.complete:
                            controller.disable("CHALLENGE_COMPLETE")
                            record_transition(time.monotonic())
                            source_closed = source.close()
                            source_started = False
                            print(f"Challenge complete: {challenge.challenge.elapsed(time.monotonic()):.2f}s")
                            print(f"Saved trajectory: {challenge.saved_path}")
                            break
                if next_tick <= time.monotonic():
                    next_tick = time.monotonic() + arm.control_dt
                record_transition(time.monotonic())

            if args.report and now >= next_sample:
                samples.append({"wall_seconds": now - started, "enabled": controller.enabled,
                                "reason": controller.reason, "input_age_seconds": controller.age(time.monotonic()),
                                **asdict(arm.state())})
                next_sample = now + 0.1
            if renderer is not None and not args.headless and now >= next_draw:
                import cv2
                now = time.monotonic()
                if challenge is not None and challenge.complete:
                    continue
                view = combined_frame(renderer, controller, packet, args, now, frames_observed / max(now - started, 0.001), challenge)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    reason = "user_exit"
                    break
                if key == ord(" "):
                    controller.toggle(time.monotonic())
                    if challenge is not None:
                        challenge.challenge.inside_since = None
                    record_transition(time.monotonic())
                elif key == ord("i"):
                    args.show_ids = not args.show_ids
                next_draw = time.monotonic() + 0.04
            time.sleep(max(0.0, min(0.005, next_tick - time.monotonic())))
    except KeyboardInterrupt:
        reason, interrupted = "interrupted", True
    except Exception as error:
        reason = "runtime_error"
        error_message = f"{type(error).__name__}: {error}"
    finally:
        controller.disable("STOPPED")
        record_transition(time.monotonic())
        if source_started:
            source_closed = source.close()
        try:
            if args.snapshot and packet is not None:
                import cv2
                if renderer is None:
                    renderer = make_renderer()
                view = frozen_view if frozen_view is not None else combined_frame(
                    renderer, controller, packet, args, time.monotonic(),
                    frames_observed / max(time.monotonic() - started, 0.001), challenge)
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.snapshot), view):
                    raise RuntimeError(f"Cannot save snapshot: {args.snapshot}")
                print(f"Saved snapshot: {args.snapshot.resolve()}")
        except Exception as error:
            reason = "snapshot_error"
            error_message = f"{type(error).__name__}: {error}"
        finally:
            if renderer is not None:
                renderer.close()
            if window_open:
                import cv2
                cv2.destroyAllWindows()

    if not source_closed:
        error_message = "Capture worker did not finish shutdown; the camera backend may be blocked."
    if args.headless and args.auto_enable and not auto_used and error_message is None:
        error_message = "No fresh single-hand input was available to enable the video demonstration."
    if challenge is not None and args.headless and not challenge.complete and error_message is None:
        error_message = "The challenge ended before all three targets were reached."
    report: dict[str, object] = {
        "mode": args.mode, "source_type": "video" if args.video is not None else "camera",
        "platform": platform.platform(), "architecture": platform.machine(),
        "versions": {name: metadata.version(name) for name in ("mediapipe", "mujoco", "mink", "opencv-contrib-python")},
        "config": asdict(config), "mirror": not args.no_mirror,
        "mapping": "display up -> world +X; display right -> world -Y",
        "frames_observed": frames_observed, "frames_processed": sequence,
        "status_counts": dict(status_counts), "transitions": transitions,
        "enable_count": controller.enable_count, "active_control_steps": controller.active_steps,
        "peak_actual_xy_displacement_m": peak_displacement,
        "peak_target_xy_displacement_m": peak_target_displacement,
        "initial_state": initial, "final_state": asdict(arm.state()), "samples": samples,
        "wall_seconds": time.monotonic() - started, "exit_reason": reason,
        "error": error_message, "worker_closed": source_closed, "manual_acceptance": "not_assessed",
    }
    if challenge is not None:
        report["challenge"] = {
            "completed": challenge.complete, "targets": challenge.challenge.targets,
            "hit_times": challenge.challenge.hit_times,
            "elapsed_seconds": challenge.challenge.elapsed(time.monotonic()),
            "trajectory": str(challenge.saved_path) if challenge.saved_path else None,
        }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved report: {args.report.resolve()}")
    if error_message:
        print(f"FAILED: {error_message}")
    print(f"Stopped: {reason}; enables={controller.enable_count}; active steps={controller.active_steps}")
    return reason == "retry" and source_closed, 130 if interrupted else (1 if error_message else 0)


def run_teleop(args: ProgramArgs) -> int:
    rng = random.Random(args.seed)
    while True:
        retry, result = _run_round(args, rng)
        if not retry or result != 0:
            return result
