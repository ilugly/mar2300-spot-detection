"""Inspect detector images, validate manifests and work with MVView reflections."""

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable
import json
from math import dist, floor, isclose, isfinite
from os.path import relpath
from pathlib import Path
import sys

import numpy as np

from .dataset import load_manifest, validate_frames
from .drx import DrxFile, WAVELENGTH_TOLERANCE
from .geometry import Geometry, calibrate_offset
from .image import read_mar2300


def series_frames(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".mar2300" and path.is_file())


def detector_offset(args: argparse.Namespace) -> tuple[float, float]:
    if args.calibration is None:
        return tuple(args.offset_mm)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8-sig"))
    offset = calibration.get("offset_mm") if isinstance(calibration, dict) else None
    if not isinstance(offset, list) or len(offset) != 2 or any(
        type(value) not in (int, float) or not -sys.float_info.max <= value <= sys.float_info.max for value in offset
    ):
        raise ValueError(f"{args.calibration}: expected two finite offset_mm values")
    return float(offset[0]), float(offset[1])


def inspect_drx(
    directory: Path, reference: DrxFile, offset: tuple[float, float],
    *, verify_pixels: bool = False, allow_missing: bool = False,
    phi_range: tuple[float, float] | None = None,
) -> dict:
    reflections_by_phi = defaultdict(list)
    for index, row in enumerate(reference.reflections, 1):
        phi = round(row.phi, 4)
        if phi_range is None or phi_range[0] <= phi <= phi_range[1]:
            reflections_by_phi[phi].append((index, row))
    reflection_count = sum(map(len, reflections_by_phi.values()))
    frames = []
    seen_phi = set()
    max_vector_error = 0.0
    mismatched = []
    for path in series_frames(directory):
        image = read_mar2300(path)
        geometry = Geometry.from_image(image, offset)
        phi = round(geometry.phi, 4)
        if phi_range is not None and not phi_range[0] <= phi <= phi_range[1]:
            continue
        if phi in seen_phi:
            raise ValueError(f"Cannot identify a unique frame for PHI={phi}: {path}")
        seen_phi.add(phi)
        points = []
        for index, row in reflections_by_phi.pop(phi, []):
            if not isclose(
                geometry.wavelength, reference.wavelength, rel_tol=0, abs_tol=WAVELENGTH_TOLERANCE,
            ):
                raise ValueError(f"Wavelength differs between the DRX and {path.name}")
            calculated = geometry.reflection(row.dx, row.dy, row.intensity)
            error = dist((calculated.qx, calculated.qy, calculated.qz), (row.qx, row.qy, row.qz))
            max_vector_error = max(max_vector_error, error)
            x, y = geometry.image_xy(row.dx, row.dy)
            if verify_pixels:
                # MVView exports half-pixel coordinates for these picks. Its
                # stored intensity identifies the pixel selected by rounding up.
                x, y = floor(x + 0.50000001), floor(y + 0.50000001)
                if not (0 <= x < geometry.width and 0 <= y < geometry.height):
                    raise ValueError(f"Reflection {index} is outside {path.name}: ({x}, {y})")
                if image.counts[y, x] != row.intensity:
                    mismatched.append(index)
            points.append({"reflection": index, "xy": [x, y], "intensity": row.intensity})
        frames.append({"image": path.name, "phi": geometry.phi, "points": points})
    if not frames:
        raise ValueError(f"No .mar2300 frames in {directory}")
    missing = sorted(
        (index, row.phi) for rows in reflections_by_phi.values() for index, row in rows
    )
    if missing and not allow_missing:
        index, phi = missing[0]
        raise ValueError(f"No frame for reflection {index} (PHI={phi})")
    result = {
        "reflection_count": reflection_count,
        "max_vector_error_inverse_angstrom": max_vector_error,
        "frames": frames,
    }
    if phi_range is not None:
        result["phi_range"] = phi_range
        result["reflections_outside_phi_range"] = len(reference.reflections) - reflection_count
    if allow_missing:
        result["missing_reflections"] = [
            {"reflection": index, "phi": phi} for index, phi in missing
        ]
    if verify_pixels:
        result["intensity_mismatches"] = sorted(mismatched)
    return result


def detect_drx(
    source: Path, output: Path, offset: tuple[float, float], find_spots: Callable[[np.ndarray], list[dict]],
) -> dict:
    """Find candidates in one frame or series and export one reflection list."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    paths = series_frames(source) if source.is_dir() else [source]
    if not paths:
        raise ValueError(f"No .mar2300 frames in {source}")

    wavelength = None
    seen_phi = set()
    reflections = []
    frames = []
    for path in paths:
        image = read_mar2300(path)
        try:
            geometry = Geometry.from_image(image, offset)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if wavelength is None:
            wavelength = geometry.wavelength
        if not isclose(geometry.wavelength, wavelength, rel_tol=0, abs_tol=WAVELENGTH_TOLERANCE):
            raise ValueError(f"Wavelength differs within the series: {path}")
        phi = round(geometry.phi, 4)
        if phi in seen_phi:
            raise ValueError(f"Cannot identify a unique frame for PHI={phi}: {path}")
        seen_phi.add(phi)

        candidates = find_spots(image.counts)
        for candidate in candidates:
            x = candidate["x"] + 1
            y = geometry.height - candidate["y"]
            reflections.append(geometry.pick(image, x, y))
        frames.append({"image": str(path), "phi": geometry.phi, "reflections": len(candidates)})

    DrxFile(wavelength, reflections).write(output)
    return {
        "output": str(output),
        "wavelength": wavelength,
        "offset_mm": offset,
        "reflections": len(reflections),
        "frames": frames,
    }


def import_drx(args: argparse.Namespace) -> dict:
    assignments = {}
    for split in ("train", "validation", "test"):
        for series in getattr(args, split):
            if Path(series).name != series or series in (".", "..") or series in assignments:
                raise ValueError(f"Expected a unique series directory name: {series!r}")
            assignments[series] = split

    phi_ranges = {}
    for series, start, end in args.phi_range:
        if series not in assignments or series in phi_ranges:
            raise ValueError(f"Expected one range for an assigned series: {series!r}")
        lower, upper = float(start), float(end)
        if not all(map(isfinite, (lower, upper))) or lower > upper:
            raise ValueError(f"Invalid PHI range for {series!r}: {start} to {end}")
        phi_ranges[series] = (lower, upper)

    rows = []
    summaries = {}
    for series, split in assignments.items():
        directory = args.directory / series
        references = sorted(directory.glob("*.drx"))
        if len(references) != 1:
            raise ValueError(f"Expected one DRX file in {directory}, found {len(references)}")
        report = inspect_drx(
            directory, DrxFile.read(references[0]), (0, 0),
            verify_pixels=True, allow_missing=args.allow_missing, phi_range=phi_ranges.get(series),
        )
        mismatches = report["intensity_mismatches"]
        if mismatches:
            raise ValueError(f"{series}: {len(mismatches)} DRX intensities disagree with the image")
        annotated = [frame for frame in report["frames"] if frame["points"]]
        if not annotated:
            raise ValueError(f"No annotated frames in {directory}")
        for frame in annotated:
            points = list(dict.fromkeys(tuple(point["xy"]) for point in frame["points"]))
            rows.append({
                "image": relpath(directory / frame["image"], args.manifest.parent),
                "experiment": series,
                "split": split,
                "points": points,
            })
        summaries[series] = {
            "split": split,
            "frames": len(report["frames"]),
            "annotated_frames": len(annotated),
            "drx_reflections": report["reflection_count"],
            "missing_reflections": report.get("missing_reflections", []),
            "max_vector_error_inverse_angstrom": report["max_vector_error_inverse_angstrom"],
        }
        if series in phi_ranges:
            summaries[series]["phi_range"] = report["phi_range"]
            summaries[series]["reflections_outside_phi_range"] = report["reflections_outside_phi_range"]
    with args.manifest.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"manifest": str(args.manifest), "series": summaries}


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
    calibrate = commands.add_parser("calibrate-offset", help="Fit detector offsets from reference MVView picks")
    calibrate.add_argument("image", type=Path)
    calibrate.add_argument("reference", type=Path)
    calibrate.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"), action="append", required=True,
                           help="One-based MVView coordinates, in reference DRX row order")
    drx = commands.add_parser("inspect-drx", help="Match DRX records to frames by PHI and locate them in pixels")
    drx.add_argument("directory", type=Path, help="Directory containing one series")
    drx.add_argument("reference", type=Path)
    drx.add_argument("--verify-pixels", action="store_true", help="Check DRX intensities at the rounded pixel coordinates")
    drx.add_argument("--allow-missing", action="store_true", help="List reflections whose frame is absent")
    importer = commands.add_parser("import-drx", help="Write a positive-only JSONL manifest from DRX series")
    importer.add_argument("directory", type=Path, help="Root containing one directory per series")
    importer.add_argument("manifest", type=Path)
    importer.add_argument("--train", nargs="+", required=True, metavar="SERIES")
    for split in ("validation", "test"):
        importer.add_argument(f"--{split}", nargs="+", default=[], metavar="SERIES")
    importer.add_argument("--allow-missing", action="store_true", help="Skip reflections whose frame is absent")
    importer.add_argument(
        "--phi-range", nargs=3, action="append", default=[], metavar=("SERIES", "FROM", "TO"),
        help="Include only this inclusive range of frame midpoint angles for the named series",
    )
    train = commands.add_parser("train", help="Train SpotCNN from annotated patches")
    train.add_argument("manifest", type=Path)
    train.add_argument("checkpoint", type=Path)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--seed", type=int, default=17)
    evaluate = commands.add_parser("evaluate", help="Measure recall of recorded spots on complete frames")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--match-radius-px", type=float, help="Distance allowed for a match (default: 5 pixels)")
    evaluate.add_argument("--details", action="store_true", help="Include counts and missed points for each frame")
    predict = commands.add_parser("predict", help="Rank spot candidates in one frame")
    predict.add_argument("image", type=Path)
    predict.add_argument("checkpoint", type=Path)
    detect = commands.add_parser("detect-drx", help="Detect spots and export one frame or a series as DRX")
    detect.add_argument("input", type=Path, help="A .mar2300 frame or a directory containing one series")
    detect.add_argument("checkpoint", type=Path)
    detect.add_argument("output", type=Path)
    for command in (train, evaluate, predict, detect):
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    for command in (predict, detect):
        command.add_argument("--limit", type=int, default=25, help="Maximum candidates per frame")
    for command in (evaluate, predict, detect):
        command.add_argument("--min-score", type=float, default=0.0, help="Minimum model score, from 0 to 1")
        command.add_argument("--ranking", choices=("cnn", "contrast"), default="contrast" if command is detect else "cnn",
                             help="Rank by CNN score or include local contrast")
    for command in (export, drx, detect):
        calibration = command.add_mutually_exclusive_group()
        calibration.add_argument("--offset-mm", nargs=2, type=float, metavar=("DX", "DY"),
                                 default=(0, 0), help="Effective offsets added to detector coordinates")
        calibration.add_argument("--calibration", type=Path, help="JSON from calibrate-offset for this setup")
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
            geometry = Geometry.from_image(image, detector_offset(args))
            reflections = [geometry.pick(image, x, y) for x, y in args.point]
            DrxFile(geometry.wavelength, reflections).write(args.output)
            result = {"output": str(args.output), "reflections": len(reflections)}
        elif args.command == "calibrate-offset":
            image = read_mar2300(args.image)
            offset = calibrate_offset(image, DrxFile.read(args.reference), args.point)
            result = {"offset_mm": offset, "image": str(args.image), "reference": str(args.reference)}
        elif args.command == "import-drx":
            result = import_drx(args)
        elif args.command in ("train", "evaluate", "predict", "detect-drx"):
            try:
                from . import learning
            except ModuleNotFoundError as exc:
                if exc.name != "torch":
                    raise
                raise ValueError("Install the ml extra to use PyTorch commands") from exc
            if args.command == "train":
                result = learning.train(
                    load_manifest(args.manifest), args.checkpoint, args.epochs, args.device, args.seed,
                )
            else:
                model = learning.load_model(args.checkpoint, args.device)
                if args.command == "evaluate":
                    result = learning.evaluate(
                        model, load_manifest(args.manifest), args.split,
                        min_score=args.min_score, ranking=args.ranking,
                        match_radius=args.match_radius_px if args.match_radius_px is not None else learning.MATCH_RADIUS,
                        details=args.details,
                    )
                else:
                    def find_spots(counts):
                        return learning.predict(
                            model, counts, args.limit, min_score=args.min_score, ranking=args.ranking,
                        )

                    if args.command == "detect-drx":
                        result = detect_drx(args.input, args.output, detector_offset(args), find_spots)
                        result.update(ranking=args.ranking, min_score=args.min_score, limit=args.limit)
                    else:
                        counts = read_mar2300(args.image).counts
                        result = {"image": str(args.image), "candidates": find_spots(counts)}
        else:
            result = inspect_drx(
                args.directory, DrxFile.read(args.reference), detector_offset(args),
                verify_pixels=args.verify_pixels, allow_missing=args.allow_missing,
            )
    except (OSError, ValueError) as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
