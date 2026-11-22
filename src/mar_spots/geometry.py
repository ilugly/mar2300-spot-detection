"""Geometry verified for MAR2300 acquisitions at zero chi and omega."""

from dataclasses import dataclass
from math import cos, hypot, isfinite, radians, sin

from .drx import Reflection
from .image import DetectorImage


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    center_x: float
    center_y: float
    step_x: float
    step_y: float
    distance: float
    wavelength: float
    two_theta: float
    phi: float
    offset_x: float
    offset_y: float

    @classmethod
    def from_image(cls, image: DetectorImage, offset_mm: tuple[float, float]) -> "Geometry":
        header = image.header
        try:
            center_x, center_y = float(header["CENTER_X"]), float(header["CENTER_Y"])
            pixel_x, pixel_y = float(header["PIXEL_LENGTH"]), float(header["PIXEL_HEIGHT"])
            distance = float(header["DISTANCE"])
            wavelength = float(header["WAVELENGTH"])
            two_theta = float(header["TWOTHETA"])
            phi_start, phi_end = float(header["PHI_START"]), float(header["PHI_END"])
            chi = float(header["CHI"])
            omega_start, omega_end = float(header["OMEGA_START"]), float(header["OMEGA_END"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid MAR geometry metadata: {exc}") from exc
        if chi or omega_start or omega_end:
            raise ValueError("Geometry currently requires zero CHI and OMEGA")
        height, width = image.counts.shape
        values = (
            center_x, center_y, pixel_x, pixel_y, distance, wavelength,
            two_theta, phi_start, phi_end, *offset_mm,
        )
        if not all(map(isfinite, values)) or min(pixel_x, pixel_y, distance, wavelength) <= 0:
            raise ValueError("Invalid MAR detector geometry or offset")
        return cls(
            width, height, center_x, center_y, pixel_x / 1000, pixel_y / 1000,
            distance, wavelength, two_theta, (phi_start + phi_end) / 2, *offset_mm,
        )

    def detector_mm(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.center_x) * self.step_x + self.offset_x,
            (y - self.center_y) * self.step_y + self.offset_y,
        )

    def image_xy(self, dx: float, dy: float) -> tuple[float, float]:
        """Convert detector millimeters to zero-based NumPy column and row."""
        mv_x = self.center_x + (dx - self.offset_x) / self.step_x
        mv_y = self.center_y + (dy - self.offset_y) / self.step_y
        return mv_x - 1, self.height - mv_y

    def reflection(self, dx: float, dy: float, intensity: int) -> Reflection:
        # X points towards the source. The MAR header uses the opposite
        # sign to the detector rotation around the laboratory Z axis.
        theta = radians(-self.two_theta)
        ray_length = hypot(self.distance, dx, dy)
        outgoing_x = (-self.distance * cos(theta) - dx * sin(theta)) / ray_length
        outgoing_y = (-self.distance * sin(theta) + dx * cos(theta)) / ray_length
        q_lab_x = (outgoing_x + 1) / self.wavelength
        q_lab_y = outgoing_y / self.wavelength
        qz = dy / ray_length / self.wavelength

        # Undo the sample rotation to express all frames in the same axes.
        phi = radians(-self.phi)
        qx = q_lab_x * cos(phi) - q_lab_y * sin(phi)
        qy = q_lab_x * sin(phi) + q_lab_y * cos(phi)
        return Reflection(qx, qy, qz, intensity, dx, dy, self.phi, hypot(qx, qy, qz))

    def pick(self, image: DetectorImage, x: int, y: int) -> Reflection:
        if not (1 <= x <= self.width and 1 <= y <= self.height):
            raise ValueError(f"MVView point ({x}, {y}) is outside the image")
        intensity = int(image.counts[self.height - y, x - 1])
        return self.reflection(*self.detector_mm(x, y), intensity)
