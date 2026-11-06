"""后台采集与手部识别；控制线程只读取最新结果，不等待相机或推理。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Self

import numpy as np

from .arguments import ProgramArgs
from .data_types import ImageArray
from .hand_tracking import HandObservation, next_timestamp_ms, observe_hand

if TYPE_CHECKING:
    from cv2 import VideoCapture


@dataclass(frozen=True)
class HandFrame:
    sequence: int
    observation: HandObservation
    captured_at: float  # capture.read() 返回的单调时间；不等同于硬件曝光时间。
    frame: ImageArray
    inference_ms: float
    loss_count: int = 0  # 累积失效帧数，避免“只读最新帧”漏掉短暂丢手事件。


class HandInput:
    def __init__(self, args: ProgramArgs) -> None:
        self.args: ProgramArgs = args
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="airarm-hand-input", daemon=True)
        self._latest: HandFrame | None = None
        self._finished: bool = False
        self._reason: str | None = None
        self._error: str | None = None

    def start(self) -> Self:
        self._thread.start()
        return self

    def snapshot(self) -> tuple[HandFrame | None, bool, str | None, str | None]:
        """按顺序返回：最新帧、是否结束、结束原因、错误信息。"""
        with self._lock:
            return self._latest, self._finished, self._reason, self._error

    def close(self) -> bool:
        self._stop.set()
        self._thread.join(timeout=2)
        return not self._thread.is_alive()

    def _run(self) -> None:
        capture: VideoCapture | None = None
        frames = 0
        loss_count = 0
        last_timestamp = -1
        reason = "stopped"
        error_message: str | None = None
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(Path(self.args.model).resolve()),
                    delegate=python.BaseOptions.Delegate.CPU,
                ),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=self.args.confidence,
                min_hand_presence_confidence=self.args.confidence,
                min_tracking_confidence=self.args.confidence,
            )
            with vision.HandLandmarker.create_from_options(options) as detector:
                if self.args.video is None:
                    backend = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                               "avfoundation": cv2.CAP_AVFOUNDATION, "v4l2": cv2.CAP_V4L2}[self.args.backend]
                    capture = cv2.VideoCapture(self.args.index, backend)
                    if capture.isOpened():
                        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    capture = cv2.VideoCapture(str(self.args.video.resolve()))
                if not capture.isOpened():
                    raise RuntimeError("Cannot open camera/video. Check permissions, --index and --backend.")
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or fps <= 0:
                    fps = 30.0
                started = time.monotonic()
                while not self._stop.is_set():
                    # 视频按自身帧率播放，避免脱离物理时间瞬间读完。
                    if self.args.video is not None:
                        if self._stop.wait(max(0.0, started + frames / fps - time.monotonic())):
                            break
                    ok, captured_frame = capture.read()
                    captured_at = time.monotonic()
                    if not ok or captured_frame is None:
                        if self.args.video is not None and frames:
                            reason = "end_of_video"
                            break
                        raise RuntimeError("Camera/video stopped returning frames.")
                    # OpenCV 的 MatLike 标注较宽；明确边界之后跨线程传递的图像类型。
                    frame: ImageArray = np.asarray(captured_frame, dtype=np.uint8)
                    timestamp = next_timestamp_ms(captured_at - started, last_timestamp)
                    last_timestamp = timestamp
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    before = time.monotonic()
                    result = detector.detect_for_video(image, timestamp)
                    elapsed_ms = (time.monotonic() - before) * 1000
                    hands = [[(point.x, point.y) for point in hand] for hand in result.hand_landmarks]
                    observation = observe_hand(hands, timestamp, mirror=not self.args.no_mirror)
                    frames += 1
                    if not observation.valid:
                        loss_count += 1
                    with self._lock:
                        self._latest = HandFrame(frames, observation, captured_at, frame, elapsed_ms, loss_count)
                    if self.args.max_frames is not None and frames >= self.args.max_frames:
                        reason = "frame_limit"
                        break
        except Exception as error:
            reason = "input_error"
            error_message = f"{type(error).__name__}: {error}"
        finally:
            if capture is not None:
                capture.release()
            with self._lock:
                self._reason, self._error = reason, error_message
                self._finished = True
