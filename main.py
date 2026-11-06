"""
This is the main entry point for the AirArm application.
"""
import argparse
from collections.abc import Sequence
import math
import sys
from pathlib import Path

from airarm.arguments import ProgramArgs

ROOT = Path(__file__).resolve().parents[0]

def confidence(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"Confidence must be between 0 and 1, but got {value}")
    return value

def positive_number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive, but got {value}")
    return value

def positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Value must be a positive integer, but got {value}")
    return value

def parse_args(argv: Sequence[str] | None = None) -> ProgramArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["hand", "arm", "teleop"], default="hand", help="hand: camera; arm: IK demo; teleop: hand-controlled arm")
    parser.add_argument("--scene", type=Path, default=ROOT / "third_party/mink/examples/universal_robots_ur5e/scene.xml")
    parser.add_argument("--control", choices=["preset", "keyboard"], default="preset", help="Arm mode target source")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--index", type=int, default=0, help="Camera index, default: 0")
    source.add_argument("--video", type=Path, help="Use a local video instead of the camera")
    parser.add_argument("--model", type=Path, default=ROOT / "models/hand_landmarker.task")
    parser.add_argument(
        "--backend", default="auto",
        choices=["auto", "dshow", "msmf", "avfoundation", "v4l2"],
    )
    parser.add_argument("--no-mirror", action="store_true", help="Show camera coordinates instead of a mirror view")
    parser.add_argument("--show-ids", action="store_true", help="Show landmark indices 0-20")
    parser.add_argument("--confidence", type=confidence, default=0.5, help="Model detection, presence and tracking thresholds")
    parser.add_argument("--seconds", type=positive_number, help="Time limit: wall-clock seconds for hand/teleop; simulated seconds for arm")
    parser.add_argument("--max-frames", type=positive_integer, help="Stop after this many input frames")
    parser.add_argument("--headless", action="store_true", help="Run hand/teleop video or arm presets without a display window")
    parser.add_argument("--report", type=Path, help="Save environment and summary statistics as JSON")
    parser.add_argument("--snapshot", type=Path, help="Save the last annotated frame on normal exit")
    parser.add_argument("--gain", type=positive_number, default=0.20, help="Teleop meters per normalized hand displacement")
    parser.add_argument("--smoothing", type=positive_number, default=0.12, help="Teleop smoothing time constant in seconds")
    parser.add_argument("--target-speed", type=positive_number, default=0.10, help="Teleop Cartesian target speed limit in m/s")
    parser.add_argument("--stale-seconds", type=positive_number, default=0.30, help="Pause teleop when input is this old")
    parser.add_argument("--auto-enable", action="store_true", help="Teleop local video only: initially enable on the first valid hand")
    args = parser.parse_args(argv, namespace=ProgramArgs())
    if args.mode == "hand" and args.headless and args.video is None:
        parser.error("--headless requires --video; use a visible window for camera practice.")
    if args.mode == "arm":
        if args.video is not None or args.max_frames is not None:
            parser.error("--video and --max-frames belong to hand mode; use --seconds for arm mode.")
        if args.headless and args.control == "keyboard":
            parser.error("Headless arm mode requires --control preset.")
    if args.mode == "teleop" and args.headless and args.video is None:
        parser.error("Headless teleop requires --video; live camera control uses Space in the window.")
    if args.auto_enable and (args.mode != "teleop" or args.video is None):
        parser.error("--auto-enable is only available for teleop with a local --video.")
    if args.index < 0:
        parser.error("--index must be zero or greater.")
    return args

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "teleop":
            from airarm.teleop_window import run_teleop
            return run_teleop(args)
        if args.mode == "arm":
            from airarm.arm_window import run_arm
            return run_arm(args)
        # 延迟导入使机械臂模式与无窗口测试不依赖摄像头或 MediaPipe。
        from airarm.window import run
        return run(args) or 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
