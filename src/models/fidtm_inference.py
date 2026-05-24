"""
FIDTM inference wrapper.

This module loads the external FIDTM repository model and provides a clean
interface for frame-by-frame inference.

Expected external FIDTM repo structure:
    FIDTM/
    └── Networks/
        └── HR_Net/
            └── seg_hrnet.py

The model checkpoint should be a .pth file, for example:
    model_best_jhu.pth

This file does not include the FIDTM repository itself. It expects the repo path
and checkpoint path to be passed at runtime.

Example usage:
    inferencer = FIDTMInferencer(
        repo_dir="/kaggle/working/repos/FIDTM",
        checkpoint_path="/kaggle/input/.../model_best_jhu.pth",
        device="cuda",
    )

    points, count, inference_time = inferencer.predict_frame(frame_bgr)
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


Point = tuple[float, float, float]


@dataclass
class FIDTMOutput:
    """
    Output from one FIDTM frame inference.

    Attributes:
        points: List of localized points as (x, y, score).
        count: Number of extracted points.
        inference_time_sec: Time spent on model inference and point extraction.
        inference_fps: 1 / inference_time_sec.
    """

    points: list[Point]
    count: int
    inference_time_sec: float
    inference_fps: float


def safe_import_fidtm_config(repo_dir: str | Path) -> Any:
    """
    Import FIDTM config.py safely.

    The original FIDTM config.py uses argparse.parse_args(), which can crash
    inside notebooks because Jupyter passes extra arguments. This function
    temporarily patches sys.argv so the import is safe.

    Args:
        repo_dir: FIDTM repository directory.

    Returns:
        Imported config module.
    """
    repo_dir = Path(repo_dir)
    config_path = repo_dir / "config.py"

    if not config_path.exists():
        raise FileNotFoundError(f"FIDTM config.py not found: {config_path}")

    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]

    try:
        spec = importlib.util.spec_from_file_location(
            "fidtm_config_module",
            str(config_path),
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import FIDTM config from {config_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    finally:
        sys.argv = old_argv

    return module


def preprocess_bgr_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    Preprocess OpenCV BGR frame for FIDTM.

    Args:
        frame_bgr: Input BGR image.

    Returns:
        Tensor of shape [1, 3, H, W].
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    x = rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))

    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    return tensor


def extract_points_from_prediction(
    pred_map: torch.Tensor,
    threshold: float = 0.20,
    nms_kernel_size: int = 3,
    max_points: int = 6000,
) -> list[Point]:
    """
    Extract local maxima points from FIDTM prediction map.

    Args:
        pred_map: Model prediction tensor.
        threshold: Minimum activation threshold.
        nms_kernel_size: Kernel size for local maxima detection.
        max_points: Maximum points to return.

    Returns:
        List of points in format (x, y, score).
    """
    if pred_map.ndim == 4:
        pred_map = pred_map[0, 0]
    elif pred_map.ndim == 3:
        pred_map = pred_map[0]

    heat = pred_map.detach().cpu().numpy().astype(np.float32)
    heat[heat < threshold] = 0

    if heat.max() <= 0:
        return []

    kernel = np.ones((nms_kernel_size, nms_kernel_size), np.uint8)
    dilated = cv2.dilate(heat, kernel)

    peaks = (heat == dilated) & (heat > threshold)

    ys, xs = np.where(peaks)
    scores = heat[ys, xs]

    order = np.argsort(scores)[::-1]

    xs = xs[order]
    ys = ys[order]
    scores = scores[order]

    points: list[Point] = []

    for x, y, score in zip(xs[:max_points], ys[:max_points], scores[:max_points]):
        points.append((float(x), float(y), float(score)))

    return points


class FIDTMInferencer:
    """
    FIDTM model wrapper for frame-by-frame inference.
    """

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        threshold: float = 0.20,
        nms_kernel_size: int = 3,
        max_points: int = 6000,
    ):
        self.repo_dir = Path(repo_dir)
        self.checkpoint_path = Path(checkpoint_path)

        if not self.repo_dir.exists():
            raise FileNotFoundError(f"FIDTM repo not found: {self.repo_dir}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"FIDTM checkpoint not found: {self.checkpoint_path}")

        self.threshold = threshold
        self.nms_kernel_size = nms_kernel_size
        self.max_points = max_points

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model()

    def _load_model(self):
        """
        Import FIDTM network and load checkpoint.
        """
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        safe_import_fidtm_config(self.repo_dir)

        from Networks.HR_Net.seg_hrnet import get_seg_model

        model = get_seg_model()

        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location=self.device,
        )

        state_dict = checkpoint.get("state_dict", checkpoint)

        cleaned_state_dict = {}

        for key, value in state_dict.items():
            cleaned_key = key.replace("module.", "")
            cleaned_state_dict[cleaned_key] = value

        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

        print("✅ FIDTM checkpoint loaded")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

        model = model.to(self.device)
        model.eval()

        return model

    def predict_frame(self, frame_bgr: np.ndarray) -> FIDTMOutput:
        """
        Run FIDTM on one BGR frame.

        Args:
            frame_bgr: OpenCV BGR frame.

        Returns:
            FIDTMOutput object.
        """
        height, width = frame_bgr.shape[:2]

        start_time = time.time()

        tensor = preprocess_bgr_frame(frame_bgr).to(self.device)

        with torch.no_grad():
            pred = self.model(tensor)

        if isinstance(pred, (tuple, list)):
            pred_map = pred[0]
        else:
            pred_map = pred

        if pred_map.ndim == 4 and (
            pred_map.shape[-2] != height or pred_map.shape[-1] != width
        ):
            pred_map = F.interpolate(
                pred_map,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        points = extract_points_from_prediction(
            pred_map=pred_map,
            threshold=self.threshold,
            nms_kernel_size=self.nms_kernel_size,
            max_points=self.max_points,
        )

        end_time = time.time()

        inference_time = end_time - start_time
        inference_fps = 1.0 / inference_time if inference_time > 0 else 0.0

        return FIDTMOutput(
            points=points,
            count=len(points),
            inference_time_sec=inference_time,
            inference_fps=inference_fps,
        )