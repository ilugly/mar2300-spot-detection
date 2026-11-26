from importlib.util import find_spec
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mar_spots.dataset import Frame, Point
from mar_spots.image import DetectorImage


@unittest.skipUnless(find_spec("torch"), "PyTorch is optional")
class ModelTest(unittest.TestCase):
    def test_contrast_ranking_can_filter_a_flat_frame(self):
        import torch

        from mar_spots.learning import predict

        model = torch.nn.Conv2d(1, 1, kernel_size=1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        counts = np.full((24, 24), 10, dtype=np.uint32)
        self.assertEqual(predict(model, counts, 10, min_score=0.51, ranking="contrast"), [])
        counts[12, 12] = 100
        first, = predict(model, counts, 1, min_score=0.51, ranking="contrast")
        self.assertLessEqual(abs(first["x"] - 12), 1)
        self.assertLessEqual(abs(first["y"] - 12), 1)

    def test_evaluation_matches_close_points_and_only_counts_confirmed_false_positives(self):
        from mar_spots.learning import evaluate

        frames = [
            Frame(Path("close.mar2300"), "partial", "test", (Point(10, 10), Point(15, 10))),
            Frame(Path("empty.mar2300"), "complete", "test", (), complete=True),
        ]
        candidates = [{"x": 11, "y": 10, "score": 0.95}, {"x": 7, "y": 10, "score": 0.9}]
        image = DetectorImage(np.zeros((20, 20), dtype=np.uint32), {})
        with patch("mar_spots.learning.read_mar2300", return_value=image), patch(
            "mar_spots.learning.predict", side_effect=[candidates, candidates[:1]],
        ):
            result = evaluate(None, frames, "test", details=True)
        self.assertEqual(result["overall"]["matched_at_100"], 2)
        self.assertEqual(result["overall"]["false_positives_at_100"], 1)
        self.assertNotIn("false_positives_at_100", result["experiments"]["partial"])
        self.assertEqual(result["frame_results"][1]["false_positives"], candidates[:1])

    def test_synthetic_patches_can_train(self):
        import torch

        from mar_spots.model import SpotCNN, point_target, spot_loss

        torch.manual_seed(17)
        torch.set_num_threads(1)
        shape = (12, 16)
        positive = point_target(shape, (Point(4, 3), Point(15, 11)), radius=1)
        negative = point_target(shape, (), radius=1)
        self.assertEqual(int(positive.sum()), 8)
        self.assertEqual(int(negative.sum()), 0)
        target = torch.from_numpy(np.stack((positive, negative)))[:, None]
        images = target + torch.rand_like(target) * 0.1

        model = SpotCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = spot_loss(model(images), target, positive_weight=10).item()
        for _ in range(20):
            optimizer.zero_grad()
            logits = model(images)
            loss = spot_loss(logits, target, positive_weight=10)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = model(images)
            self.assertEqual(logits.shape, target.shape)
            self.assertLess(spot_loss(logits, target, positive_weight=10).item(), initial)
            self.assertGreater(logits[target == 1].mean().item(), logits[target == 0].mean().item())
