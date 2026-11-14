"""Score artificial descriptor grids with the five study configurations."""

import faiss
import numpy as np

from codimatch_d import (
    DescriptorScorer,
    fit_diagonal_transforms,
    query_streams,
    reference_streams,
    scatter_scores,
)


def main() -> None:
    faiss.omp_set_num_threads(2)
    faiss.cvar.distance_compute_blas_threshold = 128000
    rng = np.random.default_rng(17)
    grids = [(2, 3)] * 8
    source_ids = np.repeat(np.arange(4), 12)
    bank_layers = rng.uniform(-0.4, 1, (2, 48, 4)).astype(np.float32)
    query_layers = rng.uniform(-0.4, 1, (2, 6, 4)).astype(np.float32)
    mask = np.array([[True, True, False], [True, False, True]])

    banks = reference_streams(bank_layers, grids)
    queries = query_streams(query_layers, mask)
    transforms = fit_diagonal_transforms(banks, source_ids)
    print("Artificial inputs only. One score per retained query patch.")
    for variant in ("A", "AB", "AW", "BW", "ABW"):
        scorer = DescriptorScorer(banks, variant, transforms=transforms if variant.endswith("W") else None)
        scores = scorer.score(queries)
        score_map = scatter_scores(scores, mask)
        print(f"{variant:>3} {np.array2string(score_map, precision=6)}")


if __name__ == "__main__":
    main()
