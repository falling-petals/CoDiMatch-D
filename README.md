# CoDiMatch-D

Few-shot industrial anomaly localization with a frozen DINOv2-S/14 encoder. Fit one model per product category using normal images only.

## Install

Use Python 3.12. The commands below run from the CoDiMatch-D repository.

```sh
python -m pip install -r requirements-image.txt
git clone https://github.com/facebookresearch/dinov2.git dinov2
git -C dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
python -c "from urllib.request import urlretrieve; urlretrieve('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth', 'dinov2_vits14_pretrain.pth')"
```

The examples use CPU. For a CUDA-enabled PyTorch installation, add `--device cuda:0` to both commands below. FAISS search runs on CPU.

## Train on normal images

This example fits CoDiMatch-D (`ABW`) using four normal MVTec AD `tile` images. Set the paths to match your dataset directory.

```sh
python -m codimatch_d fit --dinov2 dinov2 --weights dinov2_vits14_pretrain.pth --model models/tile.npz --references data/mvtec_ad/tile/train/good/004.png data/mvtec_ad/tile/train/good/005.png data/mvtec_ad/tile/train/good/006.png data/mvtec_ad/tile/train/good/007.png
```

Fitting stores the normal patch bank and the diagonal distance transforms in `models/tile.npz`. DINOv2 weights stay frozen. To fit an ablation, set `--variant` to `A`, `AB`, `AW`, or `BW` and use a separate model path.

## Test with a saved model

You need the downloaded DINOv2 `.pth` and a category-specific `.npz` from fitting. Run `predict` directly to load them without refitting.

```sh
python -m codimatch_d predict --dinov2 dinov2 --weights dinov2_vits14_pretrain.pth --model models/tile.npz --images data/mvtec_ad/tile/test/crack/000.png --output predictions/tile
```

Pass more paths after `--images` to reuse the same model. The output contains `scores.json` with image scores and a native-resolution `.npy` score map for each image. All patches are scored by default. For a single image with a saved foreground mask, pass `--mask mask.npy` using a Boolean array in the encoder's patch-grid shape.

## Check the paper's ablation table

```sh
python check_saved_table.py
```

This recalculates the Appendix J ablation differences from the saved metric rows in `results/`. Exact image-level replay of that table uses archived descriptors and foreground masks. The archive and fitted category checkpoints are not bundled.

Code is available under the [MIT license](LICENSE). Obtain datasets and DINOv2 weights from their original sources.
