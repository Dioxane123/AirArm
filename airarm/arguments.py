"""
for type checking of argparse
"""
import argparse
from pathlib import Path
from typing import Literal

class ProgramArgs(argparse.Namespace):
    mode: Literal["hand", "arm", "teleop"]
    scene: Path
    control: Literal["preset", "keyboard"]
    index: int
    video: Path | None
    model: Path
    backend: Literal["auto", "dshow", "msmf", "avfoundation", "v4l2"]
    no_mirror: bool
    show_ids: bool
    confidence: float
    seconds: float | None
    max_frames: int | None
    headless: bool
    report: Path | None
    snapshot: Path | None
    gain: float
    smoothing: float
    target_speed: float
    stale_seconds: float
    auto_enable: bool
