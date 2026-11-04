import cv2
from cv2.typing import MatLike
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter
from dataclasses import asdict
import time
import platform
import json

from .hand_tracking import HAND_EDGES, PALM_IDS, HandObservation ,next_timestamp_ms, observe_hand
from .arguments import ProgramArgs

WINDOW = "AirArm - I: landmark IDs | Q/Esc: quit"

def draw_landmarks(frame: MatLike, obs: HandObservation, *,
                    mirror: bool, show_ids: bool, fps: float, inference_ms: float) -> MatLike:
    """
    Draw the hand landmarks and edges on the frame.
    """
    height, width = frame.shape[:2]

    def to_pixel(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return (
            round(min(1.0, max(0.0, x)) * (width - 1)),
            round(min(1.0, max(0.0, y)) * (height - 1)),
        )

    if obs.valid:
        points = [to_pixel(point) for point in obs.landmarks_xy]
        for start, end in HAND_EDGES:
            cv2.line(frame, points[start], points[end], (90, 200, 90), 2)
        for index, point in enumerate(points):
            color = (255, 255, 0) if index in PALM_IDS else (255, 255, 255)
            cv2.circle(frame, point, 3, color, -1)
            if show_ids:
                cv2.putText(frame, str(index), (point[0] + 5, point[1] - 5),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 90, 90), 1)
        if obs.palm_xy is not None:
            cv2.drawMarker(frame, to_pixel(obs.palm_xy), (153, 153, 255),
                            cv2.MARKER_CROSS, 10, 2)

    status_color = (100, 240, 100) if obs.valid else (0, 190, 255)
    palm_text = "--" if obs.palm_xy is None else (
        f"x={obs.palm_xy[0]:.3f} y={obs.palm_xy[1]:.3f}"
    )
    lines = [
        (f"{obs.status} | hands={obs.detected_hands}", status_color),
        (f"Palm: {palm_text} | view={'MIRROR' if mirror else 'CAMERA'}", (240, 240, 240)),
        (f"Loop FPS: {fps:.1f} | inference: {inference_ms:.1f} ms", (240, 240, 240)),
    ]
    cv2.rectangle(frame, (0, 0), (width - 1, 76), (25, 25, 25), -1)
    for row, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10, 21 + row * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

def run(args: ProgramArgs) -> None:
    if not args.model.is_file():
        raise FileNotFoundError("Hand model missing; run: uv run python scripts/download_hand_model.py")
    if args.video is not None and not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    
    frames = 0
    counts = Counter()
    transitions = []
    inference_total_ms = 0.0
    observation: None | HandObservation = None
    previous_status: None | str = None
    last_timestamp = -1
    last_frame: None | MatLike = None
    dimensions: None | dict[str, int] = None
    window_open = False
    reason = "unknown"

    if args.video is None:
        backend = {
            "auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF, "avfoundation": cv2.CAP_AVFOUNDATION,
            "v4l2": cv2.CAP_V4L2,
        }[args.backend]
        capture = cv2.VideoCapture(args.index, backend)
    else:
        capture = cv2.VideoCapture(str(args.video.resolve()))
    
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {args.video or args.index}")
        if args.video is None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) #escape latency

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0
            if args.video is not None:
                print("Warning: Video source did not report FPS; assuming 30.0")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(args.model.resolve()),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,  # refuse to detect more than 2 hands
            min_hand_detection_confidence=args.confidence,
            min_hand_presence_confidence=args.confidence,
            min_tracking_confidence=args.confidence,
        )

        with vision.HandLandmarker.create_from_options(options) as detector:
            if not args.headless:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                window_open = True 
            start = time.monotonic()
            while True:
                frame_started = time.monotonic()
                if args.seconds is not None and frame_started - start >= args.seconds:
                    reason = "time_limit"
                    break
                ok, raw = capture.read()
                if not ok or raw is None:
                    if args.video is not None and frames:
                        reason = "end_of_video"
                        break
                    raise RuntimeError("No image received from the input; stopping instead of reusing old coordinates.")

                source_seconds = (
                    frames / source_fps if args.video is not None
                    else time.monotonic() - start
                )
                timestamp = next_timestamp_ms(source_seconds, last_timestamp)
                last_timestamp = timestamp
                # OpenCV 提供 BGR；模型需要 RGB。先对原图推理，显示时才镜像。
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                inference_started = time.monotonic()
                result = detector.detect_for_video(image, timestamp)
                inference_ms = (time.monotonic() - inference_started) * 1000
                hands = [[(point.x, point.y) for point in hand] for hand in result.hand_landmarks]
                observation = observe_hand(hands, timestamp, mirror=not args.no_mirror)
                frames += 1
                counts[observation.status] += 1
                inference_total_ms += inference_ms
                if observation.status != previous_status:
                    print(f"{timestamp:>6} ms: {observation.status}")
                    transitions.append({
                        "frame": frames, "timestamp_ms": timestamp,
                        "status": observation.status, "palm_xy": observation.palm_xy,
                    })
                    previous_status = observation.status

                frame = cv2.flip(raw, 1) if not args.no_mirror else raw.copy()
                dimensions = {"width": frame.shape[1], "height": frame.shape[0]}
                elapsed = max(time.monotonic() - start, 0.001)
                last_frame = draw_landmarks(
                    frame, observation, mirror=not args.no_mirror, show_ids=args.show_ids,
                    fps=frames / elapsed, inference_ms=inference_ms,
                )
                if not args.headless:
                    cv2.imshow(WINDOW, last_frame)
                    wait_ms = 1
                    if args.video is not None:
                        wait_ms = max(1, round(1000 / source_fps - (time.monotonic() - frame_started) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (27, ord("q")):
                        reason = "user_exit"
                        break
                    if key == ord("i"):
                        args.show_ids = not args.show_ids
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        reason = "window_closed"
                        break
                if args.max_frames is not None and frames >= args.max_frames:
                    reason = "frame_limit"
                    break
            elapsed = max(time.monotonic() - start, 0.001)
    finally:
        capture.release()
        if window_open:
            cv2.destroyAllWindows()

    if not frames or last_frame is None:
        raise RuntimeError("No frames were processed; check the input and time limit.")
    if observation is None:
        raise RuntimeError("No hand observations were produced; check the model and input.")
    report = {
        "source_type": "video" if args.video is not None else "camera",
        "source": str(args.video) if args.video is not None else args.index,
        "platform": platform.platform(), "architecture": platform.machine(),
        "python": platform.python_version(),
        "mirror": not args.no_mirror, "confidence": args.confidence,
        "dimensions": dimensions, "frames_processed": frames,
        "status_counts": dict(counts), "status_transitions": transitions,
        "elapsed_seconds": round(elapsed, 3),
        "loop_fps": round(frames / elapsed, 2),
        "mean_inference_ms": round(inference_total_ms / frames, 2),
        "exit_reason": reason, "last_observation": asdict(observation),
        "manual_acceptance": "not_assessed",
    }
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.snapshot), last_frame):
            raise RuntimeError(f"Cannot save image: {args.snapshot}")
        print(f"Saved last frame: {args.snapshot.resolve()}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report: {args.report.resolve()}")
    print(f"Processed {frames} frames. Status counts: {dict(counts)}")