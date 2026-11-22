"""Read and write the eight-column reflection files produced by MVView."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class Reflection:
    qx: float
    qy: float
    qz: float
    intensity: int
    dx: float
    dy: float
    phi: float
    norm: float


@dataclass
class DrxFile:
    wavelength: float
    reflections: list[Reflection]

    @classmethod
    def read(cls, path: Path) -> "DrxFile":
        wavelength = None
        reflections = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.partition("!")[0].split()
            if not fields:
                continue
            try:
                values = tuple(map(float, fields))
                if not all(map(isfinite, values)):
                    raise ValueError("expected finite values")
                if wavelength is None:
                    if len(values) != 1 or values[0] <= 0:
                        raise ValueError("expected one positive wavelength")
                    wavelength = values[0]
                else:
                    if len(values) != 8 or not values[3].is_integer():
                        raise ValueError("expected eight columns with integer intensity")
                    qx, qy, qz, intensity, dx, dy, phi, norm = values
                    reflections.append(Reflection(qx, qy, qz, int(intensity), dx, dy, phi, norm))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if wavelength is None:
            raise ValueError(f"{path}: missing wavelength")
        return cls(wavelength, reflections)

    def write(self, path: Path) -> None:
        lines = [f"{self.wavelength}"]
        for row in self.reflections:
            lines.append(
                f"{row.qx:12.7f} {row.qy:12.7f} {row.qz:12.7f} {row.intensity:8d}"
                f" {row.dx:9.4f} {row.dy:9.4f} {row.phi:9.4f} {row.norm:.7f}"
            )
        with path.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
