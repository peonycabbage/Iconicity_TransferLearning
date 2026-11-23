"""Extract frames from videos organised by sign class."""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing class folders of videos.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory in which to save extracted frames.")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error("--input-dir must be an existing directory")

    for category in sorted(args.input_dir.iterdir()):
        if not category.is_dir():
            continue
        for video in sorted(category.iterdir()):
            if not video.is_file():
                continue
            capture = cv2.VideoCapture(str(video))
            try:
                ok, image = capture.read()
                if not ok:
                    print(f"Skipping unreadable video: {video}")
                    continue
                output_dir = args.output_dir / category.name / video.name
                output_dir.mkdir(parents=True, exist_ok=True)
                index = 0
                while ok:
                    output_file = output_dir / f"frame{index}.jpg"
                    if not cv2.imwrite(str(output_file), image):
                        raise OSError(f"Could not write frame: {output_file}")
                    index += 1
                    ok, image = capture.read()
            finally:
                capture.release()


if __name__ == "__main__":
    main()
