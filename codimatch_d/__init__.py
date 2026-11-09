"""Descriptor-level entry points for the CoDiMatch-D study variants."""

from collections.abc import Mapping, Sequence

import numpy as np

from .context import pool_grid, pool_reference_views
from .cross_layer_retrieval import _cosine_l2, _index, _normalized
from .shrinkage_metric import ShrinkageAngularMetric

Stream = tuple[int, int]
Streams = Mapping[Stream, np.ndarray]
Transforms = Mapping[int, ShrinkageAngularMetric]

LAYERS = (6, 12)
SHRINKAGE = 0.07
RAW_STREAMS = ((6, 1), (12, 1))
CONTEXT_STREAMS = RAW_STREAMS + ((6, 3), (12, 3))
VARIANT_STREAMS = {
    "A": RAW_STREAMS,
    "AB": CONTEXT_STREAMS,
    "AW": RAW_STREAMS,
    "BW": ((12, 1), (12, 3)),
    "ABW": CONTEXT_STREAMS,
}


def reference_streams(layers: np.ndarray, grids: Sequence[tuple[int, int]], *,
                      context: bool = True) -> dict[Stream, np.ndarray]:
    """Build stream banks from aligned (2, N, D) rows in reference-view order."""
    if layers.ndim != 3 or len(layers) != len(LAYERS):
        raise ValueError("expected aligned (2, N, D) reference layers")
    streams = dict(zip(RAW_STREAMS, layers, strict=True))
    if context:
        streams.update({(layer, 3): pool_reference_views(streams[layer, 1], grids, 3)
                        for layer in LAYERS})
    return streams


def query_streams(layers: np.ndarray, mask: np.ndarray, *,
                  context: bool = True) -> dict[Stream, np.ndarray]:
    """Pool on the full query grid, then select the same Boolean mask for each stream."""
    if (layers.ndim != 3 or len(layers) != len(LAYERS) or mask.ndim != 2
            or mask.dtype != np.bool_ or layers.shape[1] != mask.size):
        raise ValueError("expected (2, H*W, D) query layers and a Boolean (H, W) mask")
    retained = mask.ravel()
    streams = dict(zip(RAW_STREAMS, layers[:, retained], strict=True))
    if context:
        streams.update({(layer, 3): pool_grid(values, mask.shape, 3)[retained]
                        for layer, values in zip(LAYERS, layers, strict=True)})
    return streams


def fit_diagonal_transforms(banks: Streams, source_ids: np.ndarray) -> dict[int, ShrinkageAngularMetric]:
    """Fit fixed diagonal W from raw normal rows only, once for each layer."""
    return {layer: ShrinkageAngularMetric.fit(banks[layer, 1], source_ids,
                                             shrinkage=SHRINKAGE, diagonal=True)
            for layer in LAYERS}


class DescriptorScorer:
    """Build once per variant and reference group, then score retained query rows."""

    def __init__(self, banks: Streams, variant: str, *, transforms: Transforms | None = None) -> None:
        self.streams = VARIANT_STREAMS[variant]
        if variant.endswith("W") != (transforms is not None):
            raise ValueError("supply diagonal transforms exactly for AW, BW and ABW")
        self.transforms = None if transforms is None else dict(transforms)
        if self.transforms is not None:
            for layer in {stream[0] for stream in self.streams}:
                model = self.transforms[layer]
                if model.diagnostics.get("diagonal") is not True or model.diagnostics.get("shrinkage") != SHRINKAGE:
                    raise ValueError("W requires the fixed diagonal shrinkage transform at 0.07")
        prepared = {stream: self._prepare(banks[stream], stream, "bank") for stream in self.streams}
        shape = prepared[self.streams[0]].shape
        if not shape[0] or any(bank.shape != shape for bank in prepared.values()):
            raise ValueError("all requested reference streams must have equal, nonempty (N, D) shapes")
        self.indices = {stream: _index(bank) for stream, bank in prepared.items()}
        self.dimension = shape[1]

    def _prepare(self, values: np.ndarray, stream: Stream, label: str) -> np.ndarray:
        if self.transforms is not None:
            values = self.transforms[stream[0]].transform(values)
        return _normalized(values, f"{label}{stream}")

    def score(self, queries: Streams) -> np.ndarray:
        """Return one FP64 mean of independent cosine 1NN distances per retained row."""
        parts = []
        count = len(queries[self.streams[0]])
        for stream in self.streams:
            query = self._prepare(queries[stream], stream, "query")
            if query.shape != (count, self.dimension):
                raise ValueError("requested query streams must have equal shapes matching the bank dimension")
            distances, _ = self.indices[stream].search(query, 1)
            parts.append(_cosine_l2(distances[:, 0]))
        return np.mean(np.stack(parts), axis=0, dtype=np.float64)


def scatter_scores(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != np.bool_ or scores.shape != (int(mask.sum()),) or not np.isfinite(scores).all():
        raise ValueError("finite retained-token scores must align with the full Boolean grid")
    result = np.zeros(mask.shape, dtype=np.float64)
    result[mask] = scores
    return result
