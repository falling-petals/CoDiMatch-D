"""Normal-source-balanced shrinkage transforms for the existing cosine search.

Covariance fitting uses unit descriptors. The fitted linear map is applied to
raw descriptors without subtracting a mean; the existing retrieval path then
normalizes them. This is an angular metric, not unnormalized Mahalanobis NN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cross_layer_retrieval import _normalized


@dataclass(frozen=True)
class ShrinkageAngularMetric:
    matrix: np.ndarray
    covariance: np.ndarray
    diagnostics: dict[str, object]

    @classmethod
    def fit(cls, features: np.ndarray, source_ids: np.ndarray, *,
            shrinkage: float = .07, diagonal: bool = False) -> ShrinkageAngularMetric:
        """Fit equal original-source population moments, retaining every channel.

        Rotations and retained patches within one source share that source's
        total weight. Shrinkage is towards trace-normalized identity. Its common
        scalar is immaterial after cosine normalization; lambda=1 is identity.
        """
        if (isinstance(shrinkage, (bool, np.bool_))
                or not isinstance(shrinkage, (int, float, np.floating))
                or not np.isfinite(shrinkage) or not 0 < shrinkage <= 1
                or type(diagonal) is not bool):
            raise ValueError('finite shrinkage in (0, 1] and Boolean diagonal required')
        unit = _normalized(features, 'normal covariance features')
        if (not isinstance(source_ids, np.ndarray) or source_ids.shape != (len(unit),)
                or source_ids.dtype.kind not in 'iu' or np.any(source_ids < 0)):
            raise ValueError('one nonnegative integer original-source ID per row required')
        sources = np.unique(source_ids)
        if len(sources) < 2:
            raise ValueError('at least two original normal sources required')
        dimension = unit.shape[1]
        mean = np.zeros(dimension, np.float64)
        second = np.zeros((dimension, dimension), np.float64)
        counts = {}
        for source in sources:
            group = unit[source_ids == source].astype(np.float64)
            mean += group.mean(axis=0) / len(sources)
            second += (group.T @ group) / (len(group) * len(sources))
            counts[str(int(source))] = len(group)
        covariance = second - np.outer(mean, mean)
        covariance = (covariance + covariance.T) / 2
        scale = float(np.trace(covariance) / dimension)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('positive normal covariance trace required')
        selected = np.diag(np.diag(covariance)) if diagonal else covariance
        regularized = (1 - shrinkage) * (selected / scale) + shrinkage * np.eye(dimension)
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        if (not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0
                or eigenvalues[0] < shrinkage - 1e-8):
            raise ValueError('normal covariance is not numerically positive semidefinite')
        if shrinkage == 1:
            matrix = np.eye(dimension, dtype=np.float32)
        else:
            matrix = ((eigenvectors * (1 / np.sqrt(eigenvalues))) @ eigenvectors.T).astype(np.float32)
        matrix = np.ascontiguousarray(matrix)
        matrix.setflags(write=False)
        covariance.setflags(write=False)
        return cls(matrix, covariance, {
            'dimension': dimension, 'normal_rows': len(unit), 'source_counts': counts,
            'source_weighting': 'equal original sources; within-source population moments',
            'shrinkage': float(shrinkage), 'diagonal': diagonal, 'trace_scale': scale,
            'regularized_eigenvalue_min': float(eigenvalues[0]),
            'regularized_eigenvalue_max': float(eigenvalues[-1]),
            'center_covariance': True, 'center_transform': False,
            'post_transform_normalization': 'existing FP32 FAISS cosine retrieval',
            'trainable_parameters': 0,
        })

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Return raw transformed FP32 rows; caller retains original normalization.

        The explicit identity path preserves raw bytes for an exact AB witness.
        No caller array or fitted matrix is mutated, and no query is fitted.
        """
        values = np.array(features, dtype=np.float32, order='C', copy=True)
        if (values.ndim != 2 or values.shape[1] != self.matrix.shape[0]
                or not np.isfinite(values).all()):
            raise ValueError('finite query/reference rows matching fitted dimension required')
        if self.diagnostics['shrinkage'] == 1:
            return values
        result = np.ascontiguousarray(values @ self.matrix, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError('transformed rows are not finite')
        return result
