# CoDiMatch-D

This release implements the descriptor-level A, AB, AW, BW and ABW variants used in the CoDiMatch-D component-removal study. ABW is CoDiMatch-D: A matches layers independently, B adds a local context stream, and W conditions distances using normal-reference statistics. The bundled saved metric rows reproduce the removal table.

The main benchmark reports dual-layer A on 27 categories. The ABW follow-up reports VisA pcb1 with two reference configurations and MVTec AD tile with one configuration. These are development-exposed queries. Conditioning improves pcb1 pixel ranking and lowers tile pixel AP relative to AB.

## Run the artificial example

Use Python 3.12 and a CPU NumPy and FAISS environment.

    python -m pip install -r requirements.txt
    python demo.py
    python check_saved_table.py

The demo uses generated descriptor arrays, without loading a backbone, images, masks or experimental caches. The saved-table checker reads the bundled CSV rows, verifies their SHA256 hashes and checks 57 contrasts across 12 case rows. It does not compute new image-level predictions or metrics.

## Descriptor inputs

The layers array has shape (2, N, D), with layers 6 and 12 in that order. The study uses frozen DINOv2-S/14 and D=384. Reference rows align across layers and follow original-source, rotation-view, row-major patch order. Pass the actual height and width of every complete reference view to reference_streams. The four original-source IDs repeat over all their views and patch positions.

For queries, use the complete row-major descriptor grid and a Boolean (H, W) query mask. Pooling operates on the complete grid before mask selection. It averages neighborhoods within a view and excludes padded positions from the divisor. Pool reference views separately to preserve their boundaries.

fit_diagonal_transforms uses unit-normalized, original-scale normal descriptors and equal total weight for each original source. The study shrinkage coefficient is 0.07. The same layer transform acts on both its original and pooled streams. Application multiplies raw descriptors by the matrix and then normalizes. The mean used for fitting the covariance is not subtracted during application.

Construct one DescriptorScorer per reference group and variant. A and AB use no transform. AW, BW and ABW reuse the same saved diagonal transforms. Search uses FP32 FAISS squared Euclidean distances between unit descriptors, followed by nonnegative clipping, conversion to FP64 and division by two. The score is the equal mean across retained streams.

Set faiss.cvar.distance_compute_blas_threshold to 128000 to match the historical runner. The demo sets this threshold and two CPU threads explicitly. Set runtime threading before building the scorer if comparisons require another recorded thread budget.

scatter_scores restores a zero-filled patch grid. The experiment's native-image interpolation, smoothing, top-one-percent image score, query foreground-mask extraction, encoder preprocessing and native-pixel AP/AUPRO evaluation are documented in the manuscript reproduction inventory. This compact release starts from descriptors and does not provide an end-to-end raw-image benchmark command.

## Sources and reproducibility

context.py and shrinkage_metric.py preserve project numerical routines with their relative import adjusted. cross_layer_retrieval.py retains the normalization, distance conversion and FAISS index kernels. __init__.py supplies the small descriptor API. provenance.json records project source hashes and the saved table hashes. The manuscript bundle contains the wider experiment inventory, source snapshots and complete metric reports.

Related methods include AnomalyDINO, MuSc+ and covariance-aware patch retrieval. Their contributions are attributed in the paper bibliography. Obtain datasets and model weights from their owners under the applicable terms.

## Licensing

The CoDiMatch-D source code is released under the [MIT License](LICENSE). The dependency packages retain their own licenses. Dataset images, model weights, and experiment caches are not included.

## Results

results/case-summary.csv contains the twelve AB, AW, BW and ABW rows. results/removal-deltas.csv contains nine full-minus-removal contrasts. A higher pixel AP and AUPRO is better. A lower false-positive rate at the declared coverage is better. All values and operating-point definitions match Appendix J of the associated manuscript.
