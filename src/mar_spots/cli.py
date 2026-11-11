"""Inspect raw detector images and validate dataset manifests locally."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from .dataset import load_manifest, validate_frames
from .image import read_mar2300


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mar-spots")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Show image dimensions, counts and header")
    inspect.add_argument("image", type=Path)
    validate = commands.add_parser("validate", help="Check images, point coordinates and splits")
    validate.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            image = read_mar2300(args.image)
            result = {
                "width": image.counts.shape[1],
                "height": image.counts.shape[0],
                "dtype": str(image.counts.dtype),
                "min": int(image.counts.min()),
                "max": int(image.counts.max()),
                "header": image.header,
            }
        else:
            frames = load_manifest(args.manifest)
            validate_frames(frames)
            result = {
                "frames": len(frames),
                "points": sum(len(frame.points) for frame in frames),
                "frames_per_split": Counter(frame.split for frame in frames),
            }
    except (OSError, ValueError) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
