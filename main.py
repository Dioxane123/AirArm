"""
This is the main entry point for the AirArm application.
"""
import argparse
import sys
from pathlib import Path
from typing import Literal

from airarm.arguments import ProgramArgs
from airarm.window import run

ROOT = Path(__file__).resolve().parents[0]

def confidence(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"Confidence must be between 0 and 1, but got {value}")
    return value

def positive_number(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive, but got {value}")
    return value

def positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Value must be a positive integer, but got {value}")
    return value

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seconds", type=positive_number, help="Stop after this many wall-clock seconds")
    parser.add_argument("--max-frames", type=positive_integer, help="Stop after this many input frames")
    parser.add_argument("--headless", action="store_true", help="Process a video without a display window")
    parser.add_argument("--report", type=Path, help="Save environment and summary statistics as JSON")
    parser.add_argument("--snapshot", type=Path, help="Save the last annotated frame on normal exit")
    args = parser.parse_args(argv, namespace=ProgramArgs())
    if args.headless and args.video is None:
        parser.error("--headless requires --video; use a visible window for camera practice.")
    if args.index < 0:
        parser.error("--index must be zero or greater.")
    return args

def main(argv=None):
    args = parse_args(argv)
    print(args.model.resolve())
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)

if __name__ == "__main__":
    main()
