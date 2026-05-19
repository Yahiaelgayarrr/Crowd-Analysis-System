from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class FIDTMInference:
    def __init__(
        self,
        repo_path: str,
        weight_path: str,
        device: str = None,
        threshold: float = 0.20,
        nms_kernel: int = 15,
    ):
        self.repo_path = Path(repo_path)
        self.weight_path = Path(weight_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.nms_kernel = nms_kernel

        # Compatibility patch for old FIDTM code
        np.int = int
        np.float = float
        np.bool = bool

        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        from Networks.HR_Net.seg_hrnet import get_seg_model

        self.model = get_seg_model()
        self.model = nn.DataParallel(self.model)

        ckpt = torch.load(self.weight_path, map_location="cpu")
        self.model.load_state_dict(ckpt["state_dict"], strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def resize_to_multiple_of_16(image_rgb: np.ndarray):
        h, w = image_rgb.shape[:2]
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        resized = cv2.resize(
            image_rgb,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized, (h, w), (new_h, new_w)

    def extract_points(self, pred_map: np.ndarray):
        pred_map = np.squeeze(pred_map)
        pred_map = np.maximum(pred_map, 0)

        local_max = cv2.dilate(
            pred_map,
            np.ones((self.nms_kernel, self.nms_kernel), np.uint8),
        )

        peaks = (pred_map == local_max) & (pred_map >= self.threshold)
        ys, xs = np.where(peaks)

        return [
            (float(x), float(y), float(pred_map[y, x]))
            for x, y in zip(xs, ys)
        ]

    @torch.no_grad()
    def predict_frame(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        rgb_resized, original_shape, resized_shape = self.resize_to_multiple_of_16(rgb)

        original_h, original_w = original_shape
        resized_h, resized_w = resized_shape

        x = self.transform(Image.fromarray(rgb_resized)).unsqueeze(0).to(self.device)

        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        output = self.model(x)

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        pred_map = output.squeeze().detach().cpu().numpy()
        raw_points = self.extract_points(pred_map)

        scale_x = original_w / resized_w
        scale_y = original_h / resized_h

        points = [
            (x0 * scale_x, y0 * scale_y, score)
            for x0, y0, score in raw_points
        ]

        return {
            "count": len(points),
            "points": points,
            "time_sec": elapsed,
            "fps": 1.0 / elapsed if elapsed > 0 else 0.0,
        }