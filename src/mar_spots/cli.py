"""Inspect detector images, validate manifests and work with MVView reflections."""

import argparse
from collections import Counter
import json
from math import dist
from pathlib import Path
import sys

from .dataset import load_manifest, validate_frames
from .drx import DrxFile
from .geometry import Geometry
from .image import read_mar2300


def inspect_drx(directory: Path, reference: DrxFile, offset: tuple[float, float]) -> dict:
    frames = {}
    for path in sorted(directory.glob("*.mar2300")):
        geometry = Geometry.from_image(read_mar2300(path), offset)
        phi = round(geometry.phi, 4)
        if phi in frames:
            raise ValueError(f"Cannot identify a unique frame for PHI={phi}: {path}")
        frames[phi] = (path.name, geometry, [])
    if not frames:
        raise ValueError(f"No .mar2300 frames in {directory}")
    max_vector_error = 0.0
    for index, row in enumerate(reference.reflections, 1):
        match = frames.get(round(row.phi, 4))
        if match is None:
            raise ValueError(f"No frame for reflection {index} (PHI={row.phi})")
        name, geometry, points = match
        if geometry.wavelength != reference.wavelength:
            raise ValueError(f"Wavelength differs between the DRX and {name}")
        calculated = geometry.reflection(row.dx, row.dy, row.intensity)
        error = dist((calculated.qx, calculated.qy, calculated.qz), (row.qx, row.qy, row.qz))
        max_vector_error = max(max_vector_error, error)
        x, y = geometry.image_xy(row.dx, row.dy)
        points.append({"reflection": index, "xy": [x, y], "intensity": row.intensity})
    return {
        "reflection_count": len(reference.reflections),
        "max_vector_error_inverse_angstrom": max_vector_error,
        "frames": [
            {"image": name, "phi": geometry.phi, "points": points}
            for name, geometry, points in frames.values()
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mar-spots")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Show image dimensions, counts and header")
    inspect.add_argument("image", type=Path)
    validate = commands.add_parser("validate", help="Check images, point coordinates and splits")
    validate.add_argument("manifest", type=Path)
    export = commands.add_parser("export-drx", help="Export selected MVView points as a DRX file")
    export.add_argument("image", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"), action="append", required=True,
                        help="One-based MVView coordinates, starting at the bottom left")
    drx = commands.add_parser("inspect-drx", help="Match DRX records to frames by PHI and locate them in pixels")
    drx.add_argument("directory", type=Path, help="Directory containing one series")
    drx.add_argument("reference", type=Path)
    for command in (export, drx):
        command.add_argument("--offset-mm", nargs=2, type=float, metavar=("DX", "DY"),
                             default=(0, 0), help="Effective offsets added to detector coordinates")
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
        elif args.command == "validate":
            frames = load_manifest(args.manifest)
            validate_frames(frames)
            result = {
                "frames": len(frames),
                "points": sum(len(frame.points) for frame in frames),
                "frames_per_split": Counter(frame.split for frame in frames),
            }
        elif args.command == "export-drx":
            image = read_mar2300(args.image)
            geometry = Geometry.from_image(image, tuple(args.offset_mm))
            reflections = [geometry.pick(image, x, y) for x, y in args.point]
            DrxFile(geometry.wavelength, reflections).write(args.output)
            result = {"output": str(args.output), "reflections": len(reflections)}
        else:
            result = inspect_drx(args.directory, DrxFile.read(args.reference), tuple(args.offset_mm))
    except (OSError, ValueError) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
