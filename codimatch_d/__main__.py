import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import faiss
import numpy as np

from . import (
    LAYERS,
    SHRINKAGE,
    VARIANT_STREAMS,
    DescriptorScorer,
    ShrinkageAngularMetric,
    fit_diagonal_transforms,
    query_streams,
    reference_streams,
    scatter_scores,
)

MODEL_FORMAT = 1


def fit(args: argparse.Namespace) -> None:
    from .images import PATCH_SIZE, ImageEncoder, read_rgb, reference_views

    if len({path.resolve() for path in args.references}) != len(args.references):
        raise ValueError("--references must list distinct normal images")
    if args.variant.endswith("W") and len(args.references) < 2:
        raise ValueError("W requires at least two normal reference images")
    if args.short_edge < PATCH_SIZE:
        raise ValueError(f"--short-edge must be at least {PATCH_SIZE}")

    encoder = ImageEncoder(args.dinov2, args.weights, args.short_edge, args.device)
    layers, grids, source_ids = [], [], []
    for source_id, path in enumerate(args.references):
        for view in reference_views(read_rgb(path)):
            descriptors, grid = encoder.encode(view)
            layers.append(descriptors)
            grids.append(grid)
            source_ids.append(np.full(descriptors.shape[1], source_id, dtype=np.int64))

    raw = np.concatenate(layers, axis=1)
    state = {
        "format": MODEL_FORMAT,
        "variant": args.variant,
        "short_edge": args.short_edge,
        "layers": raw,
        "grids": np.asarray(grids),
    }
    if args.variant.endswith("W"):
        banks = reference_streams(raw, grids, context=False)
        fitted = fit_diagonal_transforms(banks, np.concatenate(source_ids))
        for layer, metric in fitted.items():
            state[f"matrix_{layer}"] = metric.matrix
            state[f"covariance_{layer}"] = metric.covariance

    args.model.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.model.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **state)
        temporary.replace(args.model)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {args.variant} model to {args.model}")


def load_model(path: Path) -> tuple[DescriptorScorer, int, bool]:
    with np.load(path, allow_pickle=False) as state:
        if state["format"].item() != MODEL_FORMAT:
            raise ValueError("unsupported model format")
        variant = state["variant"].item()
        context = any(degree == 3 for _, degree in VARIANT_STREAMS[variant])
        banks = reference_streams(
            state["layers"], [tuple(grid) for grid in state["grids"]], context=context,
        )
        metrics = None
        if variant.endswith("W"):
            metrics = {
                layer: ShrinkageAngularMetric(
                    state[f"matrix_{layer}"],
                    state[f"covariance_{layer}"],
                    {"diagonal": True, "shrinkage": SHRINKAGE},
                )
                for layer in LAYERS
            }
        return DescriptorScorer(banks, variant, transforms=metrics), state["short_edge"].item(), context


def predict(args: argparse.Namespace) -> None:
    from .images import ImageEncoder, image_score, native_score_map, read_rgb

    if args.mask and len(args.images) != 1:
        raise ValueError("--mask takes one Boolean patch grid for a single test image")
    if len({path.name.casefold() for path in args.images}) != len(args.images):
        raise ValueError("--images must have distinct file names in a prediction batch")

    scorer, short_edge, context = load_model(args.model)
    encoder = ImageEncoder(args.dinov2, args.weights, short_edge, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    scores = []
    for path in args.images:
        rgb = read_rgb(path)
        layers, grid = encoder.encode(rgb)
        mask = np.load(args.mask, allow_pickle=False) if args.mask else np.ones(grid, dtype=bool)
        if mask.shape != grid or mask.dtype != np.bool_:
            raise ValueError(f"{path.name}: expected a Boolean patch mask with shape {grid}")
        queries = query_streams(layers, mask, context=context)
        patch_map = scatter_scores(scorer.score(queries), mask)
        map_path = args.output / f"{path.name}.npy"
        np.save(map_path, native_score_map(patch_map, rgb.shape[:2]), allow_pickle=False)
        scores.append({"image": str(path), "map": map_path.name, "score": image_score(patch_map)})

    score_path = args.output / "scores.json"
    score_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(scores)} prediction(s) to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and run CoDiMatch-D on images.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit", help="fit on normal reference images")
    train.add_argument("--references", nargs="+", required=True, type=Path)
    train.add_argument("--short-edge", type=int, default=448)
    train.add_argument("--variant", choices=VARIANT_STREAMS, default="ABW")
    train.set_defaults(run=fit)

    test = commands.add_parser("predict", help="load a model and score images")
    test.add_argument("--images", nargs="+", required=True, type=Path)
    test.add_argument("--mask", type=Path, help="optional Boolean patch-grid .npy file for one image")
    test.add_argument("--output", required=True, type=Path)
    test.set_defaults(run=predict)

    for command in (train, test):
        command.add_argument("--dinov2", required=True, type=Path, help="DINOv2 source checkout")
        command.add_argument("--weights", required=True, type=Path, help="official DINOv2-S/14 .pth")
        command.add_argument("--model", required=True, type=Path, help="CoDiMatch-D .npz checkpoint")
        command.add_argument("--device", default="cpu", help="encoder device, e.g. cuda:0")

    args = parser.parse_args()
    faiss.omp_set_num_threads(2)
    faiss.cvar.distance_compute_blas_threshold = 128000
    args.run(args)


if __name__ == "__main__":
    main()
