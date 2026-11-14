"""FP32 FAISS normalization and distance kernels from the project scorer."""

import faiss
import numpy as np


def _normalized(values: np.ndarray, name: str) -> np.ndarray:
    result = np.array(values, dtype=np.float32, order="C", copy=True)
    if result.ndim != 2 or not result.shape[1] or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite (N, D) features with D > 0")
    norms = np.sqrt(np.einsum("ij,ij->i", result, result, dtype=np.float64))
    if np.any(norms == 0):
        raise ValueError(f"{name} must contain nonzero feature rows")
    faiss.normalize_L2(result)
    norms = np.sqrt(np.einsum("ij,ij->i", result, result, dtype=np.float64))
    if not np.isfinite(result).all() or not np.allclose(norms, 1.0, rtol=1e-4, atol=0):
        raise ValueError(f"{name} cannot be unit normalized in FP32 FAISS arithmetic")
    return result


def _index(bank: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(bank)
    return index


def _cosine_l2(distances: np.ndarray) -> np.ndarray:
    return np.maximum(distances, 0).astype(np.float64) / 2.0
