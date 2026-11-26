from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from mar_spots.cli import detect_drx, main
from mar_spots.dataset import Point, load_manifest
from mar_spots.drx import DrxFile
from mar_spots.geometry import Geometry
from mar_spots.image import DetectorImage


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.image = DetectorImage(
            np.arange(16, dtype=np.uint32).reshape(4, 4),
            {
                "CENTER_X": "2", "CENTER_Y": "3",
                "PIXEL_LENGTH": "1000", "PIXEL_HEIGHT": "500",
                "DISTANCE": "10", "WAVELENGTH": "1", "TWOTHETA": "90",
                "PHI_START": "89", "PHI_END": "91",
                "CHI": "0", "OMEGA_START": "0", "OMEGA_END": "0",
            },
        )

    def test_export_and_locate_reflection(self):
        with TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.mar2300"
            frame.touch()
            export = Path(directory) / "reflections.drx"
            with patch("mar_spots.cli.read_mar2300", return_value=self.image):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main([
                        "export-drx", str(frame), str(export),
                        "--point", "2", "4", "--offset-mm", "0", "-0.5",
                    ]), 0)
                report = StringIO()
                with redirect_stdout(report):
                    self.assertEqual(main([
                        "inspect-drx", directory, str(export), "--offset-mm", "0", "-0.5",
                    ]), 0)
            parsed = DrxFile.read(export)
            self.assertEqual(parsed.wavelength, 1)
            row, = parsed.reflections
            self.assertEqual((row.qx, row.qy, row.qz), (1, -1, 0))
            self.assertEqual((row.dx, row.dy, row.phi, row.intensity), (0, 0, 90, 1))
            frame_report, = json.loads(report.getvalue())["frames"]
            point, = frame_report["points"]
            self.assertEqual(point["xy"], [1, 0])

    def test_import_verifies_half_pixel_pick_and_omits_unannotated_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            series = root / "experiment"
            series.mkdir()
            annotated = series / "annotated.mar2300"
            empty = series / "empty.mar2300"
            annotated.touch()
            empty.touch()
            geometry = Geometry.from_image(self.image, (0, 0))
            picked = geometry.reflection(0.5, 0.25, intensity=6)
            DrxFile(0.999999, [picked, replace(picked, phi=91)]).write(series / "experiment.drx")
            other = replace(self.image, header={**self.image.header, "PHI_START": "88", "PHI_END": "90"})
            manifest = root / "frames.jsonl"
            report = StringIO()
            with patch("mar_spots.cli.read_mar2300", side_effect=[self.image, other]):
                with redirect_stdout(report):
                    self.assertEqual(main([
                        "import-drx", str(root), str(manifest), "--train", series.name,
                        "--phi-range", series.name, "89", "90",
                    ]), 0)

            frame, = load_manifest(manifest)
            self.assertEqual(frame.image, annotated.resolve())
            self.assertEqual(frame.points, (Point(2, 1),))
            summary = json.loads(report.getvalue())["series"][series.name]
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["reflections_outside_phi_range"], 1)

    def test_detect_drx_uses_calibration_and_keeps_empty_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "a.MAR2300"
            second = root / "b.mar2300"
            frame.touch()
            second.touch()
            reference = root / "reference.drx"
            output = root / "detected.drx"
            expected = Geometry.from_image(self.image, (0.15, -0.5)).pick(self.image, 2, 4)
            DrxFile(1, [expected]).write(reference)
            calibration = StringIO()
            with patch("mar_spots.cli.read_mar2300", return_value=self.image), redirect_stdout(calibration):
                self.assertEqual(main([
                    "calibrate-offset", str(frame), str(reference), "--point", "2", "4",
                ]), 0)
            self.assertEqual(json.loads(calibration.getvalue())["offset_mm"], [0.15, -0.5])

            other = replace(self.image, header={**self.image.header, "PHI_START": "91", "PHI_END": "93"})
            with patch("mar_spots.cli.read_mar2300", side_effect=[self.image, other]):
                finder = Mock(side_effect=[[{"x": 1, "y": 0, "score": 0.9}], []])
                report = detect_drx(root, output, (0.15, -0.5), finder)
            self.assertEqual(report["reflections"], 1)
            self.assertEqual([item["reflections"] for item in report["frames"]], [1, 0])
            self.assertEqual(DrxFile.read(output).reflections, DrxFile.read(reference).reflections)

    def test_detect_drx_rejects_mixed_wavelengths_without_writing_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.mar2300").touch()
            (root / "b.mar2300").touch()
            output = root / "detected.drx"
            other = replace(self.image, header={**self.image.header, "WAVELENGTH": "1.2"})
            with patch("mar_spots.cli.read_mar2300", side_effect=[self.image, other]):
                with self.assertRaisesRegex(ValueError, "Wavelength differs"):
                    detect_drx(root, output, (0, 0), lambda counts: [{"x": 1, "y": 0}])
            self.assertFalse(output.exists())

    def test_rejects_unverified_rotation(self):
        self.image.header["OMEGA_END"] = "1"
        with self.assertRaisesRegex(ValueError, "requires zero CHI and OMEGA"):
            Geometry.from_image(self.image, (0, 0))
