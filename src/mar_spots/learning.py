"""Train the spot detector and rank candidate centers on complete frames."""

from collections import Counter, defaultdict
from math import hypot, isfinite
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .dataset import Frame, Point, validate_bounds
from .image import read_mar2300
from .model import SpotCNN, point_target, spot_loss

PATCH_SIZE = 32
TARGET_RADIUS = 2
MATCH_RADIUS = 5
EVALUATION_LIMITS = (25, 100)
RANKINGS = ("cnn", "contrast")


def _normalize(counts: np.ndarray) -> np.ndarray:
    return np.log1p(counts.astype(np.float32)) / 5


def _training_patches(frames: list[Frame], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    inputs = []
    targets = []
    half = PATCH_SIZE // 2
    for frame in frames:
        if not frame.points and not frame.complete:
            continue
        counts = read_mar2300(frame.image).counts
        height, width = counts.shape
        validate_bounds(frame.points, counts.shape)
        padded = np.pad(counts, half)
        # Use an equal number of patches near recorded spots and sampled
        # background patches. Explicitly annotated empty frames also contribute.
        labels = np.array([(p.x, p.y) for p in frame.points]).reshape(-1, 2)
        offsets = rng.integers(-6, 7, size=(len(labels), 2))
        centers = [
            (int(np.clip(round(p.x) + dx, 0, width - 1)),
             int(np.clip(round(p.y) + dy, 0, height - 1)))
            for p, (dx, dy) in zip(frame.points, offsets)
        ]
        candidates = np.column_stack((rng.integers(width, size=256), rng.integers(height, size=256)))
        distances = np.abs(candidates[:, None] - labels)
        outside_targets = (distances > half + TARGET_RADIUS).any(axis=2).all(axis=1)
        candidates = candidates[outside_targets]
        values = counts[candidates[:, 1], candidates[:, 0]]
        inside_detector = values > 0
        candidates, values = candidates[inside_detector], values[inside_detector]
        if len(candidates):
            background_count = min(max(len(labels), 1), len(candidates))
            # Alternate between ordinary background and brighter structures.
            bright = np.argsort(values)[-max(1, len(values) // 10):]
            chosen = rng.choice(len(candidates), size=(background_count + 1) // 2)
            chosen_bright = rng.choice(bright, size=background_count // 2)
            centers.extend(candidates[np.concatenate((chosen, chosen_bright))].tolist())

        for x, y in centers:
            left, top = x - half, y - half
            points = tuple(
                Point(p.x - left, p.y - top) for p in frame.points
                if left <= p.x <= left + PATCH_SIZE - 1 and top <= p.y <= top + PATCH_SIZE - 1
            )
            inputs.append(_normalize(padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE]))
            targets.append(point_target((PATCH_SIZE, PATCH_SIZE), points, TARGET_RADIUS))
    if not inputs:
        raise ValueError("Training requires usable patches")
    return torch.from_numpy(np.stack(inputs))[:, None], torch.from_numpy(np.stack(targets))[:, None]


def train(frames: list[Frame], output: Path, epochs: int, device: str, seed: int) -> dict:
    training = [frame for frame in frames if frame.split == "train"]
    if not any(frame.points for frame in training):
        raise ValueError("Training requires annotated points in the train split")
    if epochs < 1 or seed < 0:
        raise ValueError("Expected positive epochs and a nonnegative seed")
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(seed)
    inputs, targets = _training_patches(training, seed)
    positive = int(targets.sum())
    weight = (targets.numel() - positive) / positive
    inputs, targets = inputs.to(device), targets.to(device)
    model = SpotCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for _ in range(epochs):
        total = torch.zeros((), device=device)
        for batch in torch.randperm(len(inputs)).to(device).split(32):
            optimizer.zero_grad()
            loss = spot_loss(model(inputs[batch]), targets[batch], weight)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(batch)
        history.append(total.item() / len(inputs))
    with output.open("xb") as stream:
        torch.save(model.cpu().state_dict(), stream)
    return {
        "checkpoint": str(output), "device": device, "seed": seed, "epochs": epochs,
        "frames": len(training), "patches": len(inputs), "positive_weight": weight,
        "training_loss": history,
    }


def load_model(checkpoint: Path, device: str) -> SpotCNN:
    model = SpotCNN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def _box_mean(image: torch.Tensor, width: int) -> torch.Tensor:
    radius = width // 2
    horizontal = F.avg_pool2d(image, (1, width), stride=1, padding=(0, radius))
    return F.avg_pool2d(horizontal, (width, 1), stride=1, padding=(radius, 0))


def _local_contrast(log_counts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    support = valid.to(log_counts.dtype)
    center = _box_mean(log_counts, 3) / _box_mean(support, 3).clamp_min(1e-6)
    neighborhood = _box_mean(log_counts, 21) / _box_mean(support, 21).clamp_min(1e-6)
    return center - neighborhood


@torch.inference_mode()
def predict(
    model: SpotCNN, counts: np.ndarray, limit: int, *, min_score: float = 0.0, ranking: str = "cnn",
) -> list[dict]:
    if limit < 1:
        raise ValueError("Expected a positive candidate limit")
    if not 0 <= min_score <= 1:
        raise ValueError("Minimum score must be between 0 and 1")
    if ranking not in RANKINGS:
        raise ValueError(f"Unknown candidate ranking: {ranking}")
    device = next(model.parameters()).device
    image = torch.from_numpy(_normalize(counts))[None, None].to(device)
    height, width = counts.shape
    scores = torch.empty_like(image)
    # Three padded 3×3 convolutions need a three-pixel halo at tile seams.
    for top in range(0, height, 256):
        bottom = min(top + 256, height)
        start, stop = max(0, top - 3), min(height, bottom + 3)
        tile_scores = model(image[:, :, start:stop])
        scores[:, :, top:bottom] = tile_scores[:, :, top - start:bottom - start]
    valid = image > 0
    if ranking == "contrast":
        scores += _local_contrast(image * 5, valid)
    scores = scores.masked_fill(~valid, -torch.inf)
    maxima = F.max_pool2d(scores, (1, 9), stride=1, padding=(0, 4))
    maxima = F.max_pool2d(maxima, (9, 1), stride=1, padding=(4, 0))
    peaks = scores.masked_fill(scores != maxima, -torch.inf)
    values, indices = peaks.flatten().topk(min(limit, height * width))
    return [
        {"x": index % width, "y": index // width, "score": score}
        for index, score in zip(indices.cpu().tolist(), values.sigmoid().cpu().tolist())
        if score > 0 and score >= min_score
    ]


def _match_points(candidates: list[dict], points: tuple[Point, ...], radius: float) -> dict[int, int]:
    """Pair as many annotated points and candidates as possible, one-to-one."""
    distances = [
        [hypot(point.x - candidate["x"], point.y - candidate["y"]) for point in points]
        for candidate in candidates
    ]
    nearby = [
        sorted((i for i, distance in enumerate(row) if distance <= radius), key=row.__getitem__)
        for row in distances
    ]
    matched = {}

    def assign(candidate: int, seen: set[int]) -> bool:
        for point in nearby[candidate]:
            if point in seen:
                continue
            seen.add(point)
            if point not in matched or assign(matched[point], seen):
                matched[point] = candidate
                return True
        return False

    for candidate in range(len(candidates)):
        assign(candidate, set())
    return matched


def evaluate(
    model: SpotCNN, frames: list[Frame], split: str, *,
    min_score: float = 0.0, ranking: str = "cnn", match_radius: float = MATCH_RADIUS, details: bool = False,
) -> dict:
    if not isfinite(match_radius) or match_radius < 0:
        raise ValueError("Match radius must be finite and nonnegative")
    totals = defaultdict(Counter)
    frame_results = []
    for frame in frames:
        if frame.split != split or (not frame.points and not frame.complete):
            continue
        counts = read_mar2300(frame.image).counts
        validate_bounds(frame.points, counts.shape)
        candidates = predict(model, counts, max(EVALUATION_LIMITS), min_score=min_score, ranking=ranking)
        total = totals[frame.experiment]
        total["frames"] += 1
        total["points"] += len(frame.points)
        if frame.complete:
            total["complete_frames"] += 1
            total["complete_points"] += len(frame.points)
        frame_matches = {}
        for limit in EVALUATION_LIMITS:
            selected = candidates[:limit]
            matches = _match_points(selected, frame.points, match_radius)
            frame_matches[limit] = matches
            total[f"candidates_at_{limit}"] += len(selected)
            total[f"matched_at_{limit}"] += len(matches)
            if frame.complete:
                total[f"complete_matches_at_{limit}"] += len(matches)
                total[f"false_positives_at_{limit}"] += len(selected) - len(matches)
        if details:
            matched = frame_matches[max(EVALUATION_LIMITS)]
            result = {
                "image": str(frame.image),
                "experiment": frame.experiment,
                "complete": frame.complete,
                "points": len(frame.points),
                "candidates": len(candidates),
                "matched": {limit: len(pairs) for limit, pairs in frame_matches.items()},
                "missed": [
                    {"x": point.x, "y": point.y}
                    for index, point in enumerate(frame.points) if index not in matched
                ],
            }
            if frame.complete:
                matched_candidates = set(matched.values())
                result["false_positives"] = [
                    candidate for index, candidate in enumerate(candidates) if index not in matched_candidates
                ]
            frame_results.append(result)
    if not totals:
        raise ValueError(f"No annotated frames in the {split} split")

    def summarize(total: Counter) -> dict:
        report = dict(total)
        for limit in EVALUATION_LIMITS:
            report[f"recall_at_{limit}"] = (
                total[f"matched_at_{limit}"] / total["points"] if total["points"] else None
            )
            if total["complete_frames"]:
                matched = total[f"complete_matches_at_{limit}"]
                predicted = matched + total[f"false_positives_at_{limit}"]
                report[f"precision_on_complete_frames_at_{limit}"] = matched / predicted if predicted else None
        return report

    combined = Counter()
    for total in totals.values():
        combined.update(total)
    report = {
        "split": split, "match_radius_px": match_radius, "min_score": min_score, "ranking": ranking,
        "overall": summarize(combined),
        "experiments": {experiment: summarize(total) for experiment, total in totals.items()},
    }
    if details:
        report["frame_results"] = frame_results
    return report
