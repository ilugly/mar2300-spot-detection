from importlib.util import find_spec
import unittest

import numpy as np

from mar_spots.dataset import Point


@unittest.skipUnless(find_spec("torch"), "PyTorch is optional")
class ModelTest(unittest.TestCase):
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
