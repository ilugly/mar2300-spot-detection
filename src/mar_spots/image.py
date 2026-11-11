"""Read detector counts and metadata without applying intensity transforms."""

from dataclasses import dataclass
from pathlib import Path
from struct import error as StructError

import numpy as np
from fabio.mar345image import Mar345Image


@dataclass(frozen=True)
class DetectorImage:
    counts: np.ndarray
    header: dict[str, str | int | float]


def read_mar2300(path: Path) -> DetectorImage:
    if path.suffix.lower() != ".mar2300":
        raise ValueError(f"Expected a .mar2300 file: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 4096:
        raise ValueError(f"Missing MAR345 image payload: {path}")
    try:
        with Mar345Image() as image:
            if image.read(str(path)) is None:
                raise ValueError("Unsupported MAR345 encoding")
            counts = np.array(image.data, copy=True)
            header = dict(image.header)
    except (OSError, ValueError, RuntimeError, EOFError, LookupError, ArithmeticError, StructError) as exc:
        raise ValueError(f"Cannot decode {path}: {exc}") from exc
    if counts.ndim != 2 or not counts.size or counts.dtype.kind not in "ui":
        raise ValueError(f"Expected a non-empty 2D array of detector counts: {path}")
    return DetectorImage(counts, header)
