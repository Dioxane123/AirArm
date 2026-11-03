"""下载官方手部模型，并记录来源和 SHA-256；已有文件不会被覆盖。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/hand_landmarker.task")
    args = parser.parse_args()
    destination = args.output
    if destination.exists():
        print(f"Already exists; kept unchanged: {destination.resolve()}")
        print("Run day1_check.py hand to validate the model.")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        print(f"Downloading {MODEL_URL}")
        with urlopen(MODEL_URL, timeout=60) as response:
            payload = response.read()
        temporary.write_bytes(payload)
        # Hand Landmarker 的 .task 文件是包含多个模型的 ZIP 包。
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None or not any(name.endswith(".tflite") for name in bundle.namelist()):
                raise ValueError("The downloaded file is not a valid model bundle.")
        temporary.replace(destination)
        info = {
            "url": MODEL_URL,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "note": "Recorded digest identifies this download; it is not a publisher-provided checksum.",
        }
        destination.with_suffix(".source.json").write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved: {destination.resolve()}")
        print(f"SHA-256: {info['sha256']}")
        return 0
    except Exception as error:
        print(f"Download failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
