from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from fabio.mar345image import Mar345Image

from mar_spots.cli import main
from mar_spots.dataset import load_manifest


class PipelineTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.image = self.root / "frame.mar2300"
        counts = np.arange(4096, dtype=np.uint32).reshape(64, 64)
        frame = Mar345Image(data=counts)
        frame.byteorder = "<"
        frame.write(str(self.image))
        self.manifest = self.root / "frames.jsonl"

    def write_manifest(self, *rows):
        self.manifest.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

    def test_inspect_preserves_original_counts(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["inspect", str(self.image)]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual((result["width"], result["height"]), (64, 64))
        self.assertEqual((result["min"], result["max"]), (0, 4095))
        self.assertEqual(result["dtype"], "uint32")
        self.assertIn("PixelLength", result["header"])

    def test_validate_reads_frame_and_reports_split(self):
        self.write_manifest(
            dict(image=self.image.name, experiment="series-a", split="train", points=[[63, 0.5]])
        )
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["validate", str(self.manifest)]), 0)
        self.assertEqual(json.loads(output.getvalue())["frames_per_split"], {"train": 1})

    def test_validate_rejects_out_of_bounds_annotation(self):
        self.write_manifest(
            dict(image=self.image.name, experiment="series-a", split="train", points=[[64, 0]])
        )
        error = StringIO()
        with redirect_stderr(error):
            self.assertEqual(main(["validate", str(self.manifest)]), 1)
        self.assertIn("outside", error.getvalue())

    def test_rejects_experiment_leakage(self):
        self.write_manifest(
            dict(image="a.mar2300", experiment="series-a", split="train", points=[]),
            dict(image="b.mar2300", experiment="series-a", split="validation", points=[]),
        )
        with self.assertRaisesRegex(ValueError, "occurs in train and validation"):
            load_manifest(self.manifest)

    def test_truncated_image_reports_error(self):
        self.image.write_bytes(b"truncated")
        error = StringIO()
        with redirect_stderr(error):
            self.assertEqual(main(["inspect", str(self.image)]), 1)
        self.assertIn("Missing MAR345 image payload", error.getvalue())
