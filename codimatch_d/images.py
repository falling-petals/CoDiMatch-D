import os
import sys
from hashlib import file_digest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from . import LAYERS

WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
PATCH_SIZE = 14


class ImageEncoder:
    def __init__(self, dinov2: Path, weights: Path, short_edge: int, device: str) -> None:
        with weights.open("rb") as handle:
            if file_digest(handle, "sha256").hexdigest() != WEIGHTS_SHA256:
                raise ValueError("use the official dinov2_vits14_pretrain.pth linked in the README")
        if not (dinov2 / "dinov2/models/vision_transformer.py").is_file():
            raise ValueError("--dinov2 must point to the DINOv2 source checkout")
        os.environ["XFORMERS_DISABLED"] = "1"
        sys.path.insert(0, str(dinov2.resolve()))
        from dinov2.models.vision_transformer import vit_small

        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        self.model = vit_small(
            img_size=518, patch_size=PATCH_SIZE, init_values=1.0, ffn_layer="mlp",
            block_chunks=0, num_register_tokens=0, interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.eval().requires_grad_(False).to(device)
        self.device = device
        self.preprocess = transforms.Compose([
            transforms.Resize(short_edge, transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        tensor = self.preprocess(Image.fromarray(rgb))
        height, width = (edge // PATCH_SIZE for edge in tensor.shape[-2:])
        tensor = tensor[:, :height * PATCH_SIZE, :width * PATCH_SIZE]
        outputs = self.model.get_intermediate_layers(
            tensor.unsqueeze(0).to(self.device), n=[layer - 1 for layer in LAYERS],
        )
        layers = np.stack([output[0].cpu().numpy() for output in outputs])
        return layers, (height, width)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def reference_views(rgb: np.ndarray):
    height, width = rgb.shape[:2]
    yield rgb
    for angle in range(45, 360, 45):
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        yield cv2.warpAffine(
            rgb, rotation, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT,
        )


def native_score_map(patch_scores: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    resized = cv2.resize(patch_scores, (width, height), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(resized, sigma=4)


def image_score(patch_scores: np.ndarray) -> float:
    flat = patch_scores.ravel()
    top = max(1, flat.size // 100)
    return float(np.partition(flat, flat.size - top)[-top:].mean())
