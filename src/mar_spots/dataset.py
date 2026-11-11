"""Validate point annotations in detector pixels, grouped by experiment."""

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path

from .image import read_mar2300


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Frame:
    image: Path
    experiment: str
    split: str
    points: tuple[Point, ...]


def validate_bounds(points: tuple[Point, ...], shape: tuple[int, int]) -> None:
    height, width = shape
    for point in points:
        if not (0 <= point.x <= width - 1 and 0 <= point.y <= height - 1):
            raise ValueError(f"Point ({point.x}, {point.y}) outside {width} × {height} pixels")


def load_manifest(path: Path) -> list[Frame]:
    frames = []
    seen_images: set[Path] = set()
    experiment_splits: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                frame = _parse_frame(json.loads(line), path.parent)
                if frame.image in seen_images:
                    raise ValueError(f"Duplicate image: {frame.image}")
                previous_split = experiment_splits.setdefault(frame.experiment, frame.split)
                if previous_split != frame.split:
                    raise ValueError(
                        f"Experiment {frame.experiment!r} occurs in {previous_split} and {frame.split}"
                    )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            seen_images.add(frame.image)
            frames.append(frame)
    if not frames:
        raise ValueError(f"No frames in {path}")
    return frames


def _parse_frame(raw: object, directory: Path) -> Frame:
    if not isinstance(raw, dict) or set(raw) != {"image", "experiment", "split", "points"}:
        raise ValueError("Expected fields image, experiment, split, points")
    image, experiment, split, points = (raw[k] for k in ("image", "experiment", "split", "points"))
    if not isinstance(image, str) or not image.strip():
        raise ValueError("image must be a non-empty path")
    image_path = directory / image
    if image_path.suffix.lower() != ".mar2300":
        raise ValueError("image must reference a .mar2300 file")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment must be a non-empty string")
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")
    if not isinstance(points, list):
        raise ValueError("points must be a list of [x, y] pairs")
    parsed_points = tuple(_parse_point(point) for point in points)
    if len(set(parsed_points)) != len(parsed_points):
        raise ValueError("Duplicate point in the same image")
    return Frame(image_path.resolve(), experiment.strip(), split, parsed_points)


def _parse_point(raw: object) -> Point:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("Each point must be an [x, y] pair")
    for value in raw:
        if type(value) not in (int, float) or abs(value) > 2**53 or not isfinite(value):
            raise ValueError("Coordinates must be finite numbers")
    return Point(float(raw[0]), float(raw[1]))


def validate_frames(frames: list[Frame]) -> None:
    for frame in frames:
        try:
            image = read_mar2300(frame.image)
            validate_bounds(frame.points, image.counts.shape)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{frame.image}: {exc}") from exc
