"""Download the same OpenCV ResNet-SSD assets used by the notebook."""

import hashlib
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "artifacts" / "face_detector"
FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        destination = TARGET / filename
        if not destination.exists():
            print(f"Downloading {filename}")
            urllib.request.urlretrieve(url, destination)
        print(
            f"{filename}: {destination.stat().st_size:,} bytes, "
            f"sha256={sha256_file(destination)}"
        )


if __name__ == "__main__":
    main()
