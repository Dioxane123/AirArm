"""不使用摄像头，按已保存的命令与时间顺序重放机械臂运动。"""

import json
import time

from .arguments import ProgramArgs
from .arm_control import ArmController
from .arm_window import ArmRenderer
from .challenge import Point3
from .challenge_view import add_scene_overlay, completion_action, draw_challenge_hud
from .data_types import ImageArray
from .trajectory import TrajectoryReplay, point3


WINDOW = "AirArm trajectory replay - Space: pause | R: replay | Q: exit"


def run_replay(args: ProgramArgs) -> int:
    assert args.trajectory is not None
    arm = ArmController(args.scene)
    replay = TrajectoryReplay(args.trajectory, arm)
    renderer: ArmRenderer | None = None
    window_open = False
    paused = False
    paused_at = 0.0
    paused_total = 0.0
    clicked: str | None = None
    path: list[Point3] = [point3(arm.state().actual_xyz, "actual")]
    started = time.monotonic()
    next_draw = started
    reason = "unknown"

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        nonlocal clicked
        import cv2
        if event == cv2.EVENT_LBUTTONUP and replay.complete:
            clicked = completion_action(x, y, panel_offset=0)

    def frame() -> ImageArray:
        assert renderer is not None
        current = replay.frames[replay.index]
        index = 3 if replay.complete else current.target_index
        target = None if replay.complete else replay.targets[index]
        result = renderer.frame("REPLAY", footer="Space: pause/resume | R: replay after finish | Q: quit",
            overlay=lambda scene: add_scene_overlay(scene, replay.workspace, target, replay.radius, path))
        draw_challenge_hud(result, replay.workspace, replay.targets, index, replay.radius, path,
                           point3(arm.state().actual_xyz, "actual"),
                           replay.elapsed_seconds if replay.complete else current.wall_time,
                           1.0 if replay.complete else 0.0, complete=replay.complete,
                           saved_name=args.trajectory.name if args.trajectory else None, replay=True)
        return result

    try:
        if not args.headless or args.snapshot:
            renderer = ArmRenderer(arm)
            renderer.camera.azimuth = 0
            renderer.camera.elevation = -55
        if not args.headless:
            import cv2
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW, on_mouse)
            window_open = True
        started = next_draw = time.monotonic()
        while True:
            now = time.monotonic()
            if args.seconds is not None and now - started >= args.seconds:
                reason = "time_limit"
                break
            if not paused and not replay.complete:
                if args.headless or replay.frames[replay.index + 1].wall_time <= (now - started - paused_total) * args.replay_speed:
                    replay.step()
                    path.append(point3(arm.state().actual_xyz, "actual"))
            if replay.complete and args.headless:
                reason = "completed"
                break
            if not args.headless and now >= next_draw:
                import cv2
                cv2.imshow(WINDOW, frame())
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or clicked == "quit" or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    reason = "completed" if replay.complete else "user_exit"
                    break
                if (key == ord("r") or clicked == "retry") and replay.complete:
                    replay.reset()
                    path = [point3(arm.state().actual_xyz, "actual")]
                    started = time.monotonic()
                    paused, paused_total, clicked = False, 0.0, None
                elif key == ord(" ") and not replay.complete:
                    if paused:
                        paused_total += now - paused_at
                    else:
                        paused_at = now
                    paused = not paused
                    arm.paused = paused
                next_draw = time.monotonic() + 0.04
            if not args.headless:
                time.sleep(0.001)
        if args.snapshot:
            import cv2
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.snapshot), frame()):
                raise RuntimeError("Could not save replay snapshot.")
        if args.report:
            report = {
                "mode": "replay", "trajectory": str(args.trajectory), "completed": replay.complete,
                "frames_replayed": replay.index + 1, "max_joint_error_rad": replay.max_joint_error,
                "recorded_challenge_seconds": replay.elapsed_seconds, "exit_reason": reason,
                "final_actual_xyz": arm.state().actual_xyz, "manual_acceptance": "not_assessed",
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Replay {reason}: {replay.index + 1}/{len(replay.frames)} frames; max joint error {replay.max_joint_error:.3g} rad")
        return 1 if args.headless and not replay.complete else 0
    finally:
        if renderer is not None:
            renderer.close()
        if window_open:
            import cv2
            cv2.destroyAllWindows()
