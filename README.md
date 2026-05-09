# Acne Detection Project

This repository contains multiple PyTorch notebook pipelines for skin condition image classification (including acne-type classes), from an initial baseline to more advanced training strategies.

The notebooks are versioned so you can see how the pipeline evolved:

- `detection.ipynb`: baseline production-style classifier.
- `detection_v2.ipynb`: stronger augmentation and EfficientNet-B2 upgrade.
- `detection_v3.ipynb`: anti-overfitting upgrades (SWA, CutMix, backbone options).
- `detection_v4.ipynb`: class balancing on disk + Noisy-Student B2 default.
- `acne_classification_pytorch (1).ipynb`: standalone acne-classification notebook using Kaggle dataset and EfficientNet-B3.

## What This Project Actually Does

Despite the folder name saying "detection," the notebooks are image classification pipelines, not object detection with boxes. The model predicts a class label for each input image.

Typical class labels in these notebooks include:

- `Blackheads`
- `Cyst`
- `Papules`
- `Pustules`
- `Whiteheads`

The core goal is to train a robust classifier that can be exported for inference use.

## Repository Contents

- `detection.ipynb`
	First complete end-to-end baseline with stratified split, EfficientNet-B0 fine-tuning, checkpointing, and export.

- `detection_v2.ipynb`
	Adds Colab-aware dataset handling, stronger augmentation, MixUp, EfficientNet-B2, and improved training control.

- `detection_v3.ipynb`
	Adds stronger regularization and anti-overfit tools: SWA, CutMix + MixUp alternation, gradient accumulation, model-backbone choices (`standard`, `noisy_student`, `dinov2`), and model summary tooling.

- `detection_v4.ipynb`
	Adds offline dataset rebalancing (copy/downsample/augment to balanced classes), conservative online augmentation, Noisy-Student EfficientNet-B2 default, warmup + cosine schedule, SWA, and TTA-oriented workflow.

- `acne_classification_pytorch (1).ipynb`
	Separate workflow that downloads `tiswan14/acne-dataset-image` via KaggleHub and trains EfficientNet-B3 with class weighting, weighted sampling, early stopping, and evaluation plots.

## Full Pipeline Explanation (What Happens Step by Step)

The notebooks follow this general sequence.

### 1) Environment Setup

- Install/import dependencies (`torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`, plus optional `timm`, `torchinfo`, `seaborn`).
- Detect runtime device:
	- CUDA GPU when available.
	- MPS for Apple silicon in some notebooks.
	- CPU fallback.

Why this matters:
- Training speed and mixed precision behavior depend on device.

### 2) Dataset Input and Path Handling

Depending on notebook version:

- Local path mode:
	- Example root like `/home/dev/Desktop/cnn/dataset`.
- Colab mode:
	- Auto-extract from `/content/dataset.zip` or download zip from Google Drive in v4.
- Kaggle mode (EfficientNet-B3 notebook):
	- Download from `tiswan14/acne-dataset-image` via `kagglehub`.

Why this matters:
- Makes the same notebook run in both local and cloud environments.

### 3) Class Discovery and Class Map Save

- Use `torchvision.datasets.ImageFolder` to infer classes from folder names.
- Save `class_to_idx` mapping to json.

Why this matters:
- Inference must use the exact same class index mapping as training.

### 4) Train/Val/Test Split (Stratified by Class)

- Build class-wise index lists.
- Split each class into train/val/test by configured ratios.
- Force at least one sample in val/test when class counts are very small.

Why this matters:
- Prevents one class from disappearing from validation/testing.
- Gives fair per-class performance visibility.

### 5) Transform and Augmentation Pipeline

- Train transforms include resize/crop + photometric and geometric augmentations.
- Eval transforms are deterministic (resize + normalization).
- Later versions add stronger techniques:
	- MixUp.
	- CutMix.
	- Test-time augmentation (TTA) helper transforms.

Why this matters:
- Improves generalization and reduces overfitting to exact training appearances.

### 6) Handling Class Imbalance

Different versions use different strategies:

- Baseline/v2/v3:
	- Stratified split + regularization (+ class weighting in some workflows).
- EfficientNet-B3 notebook:
	- `WeightedRandomSampler` and class-weighted cross-entropy.
- v4:
	- Builds `balanced_dataset` on disk first.
	- Large classes can be downsampled; minority classes are augmented into synthetic files.
	- Then trains on already-balanced data without sampler tricks.

Why this matters:
- Reduces bias toward majority classes.

### 7) Model Construction

- `detection.ipynb`: EfficientNet-B0.
- `detection_v2.ipynb`: EfficientNet-B2.
- `detection_v3.ipynb`: multiple backbone options:
	- `standard` EfficientNet-B2.
	- `noisy_student` EfficientNet-B2 NS via `timm`.
	- `dinov2` ViT option via `timm` wrapper.
- `detection_v4.ipynb`: Noisy-Student EfficientNet-B2 default.
- `acne_classification_pytorch (1).ipynb`: EfficientNet-B3.

All versions replace the final classifier head to output `num_classes`.

Why this matters:
- Transfer learning provides strong pretrained features and faster convergence.

### 8) Two-Stage Fine-Tuning

Common strategy across notebooks:

- Stage 1: freeze backbone and train head only for a few epochs.
- Stage 2: unfreeze backbone and fine-tune end-to-end at lower LR.

Why this matters:
- Stabilizes early training and protects pretrained features before full adaptation.

### 9) Optimization and Scheduling

- Optimizer: mostly `AdamW`.
- Learning-rate scheduling:
	- Baseline uses plateau/cosine-style control.
	- v2/v3/v4 use warmup + cosine-like Lambda schedules.
- Mixed precision (`autocast` + `GradScaler`) where available.
- Gradient clipping for stability.
- v3/v4 include SWA (Stochastic Weight Averaging) to improve generalization.

Why this matters:
- Better optimization stability, smoother convergence, and often better val/test results.

### 10) Training Loop and Logging

Each epoch generally does:

- Train pass.
- Validation pass.
- Scheduler step.
- Metric logging (loss/accuracy/LR).
- Best-checkpoint logic by validation accuracy with `min_delta` and patience rules.

Why this matters:
- Preserves best model state and avoids overfitting after peak validation performance.

### 11) Evaluation and Diagnostics

- Load best checkpoint.
- Evaluate on test set.
- Produce:
	- classification report.
	- confusion matrix.
	- training curves (loss/accuracy over epochs).

Why this matters:
- Reveals class-level strengths/weaknesses instead of a single accuracy number.

### 12) Export for Inference

Artifacts typically include:

- `best_model.pt` (training checkpoint).
- `model_inference.pt` (compact inference bundle: weights + class names + normalization metadata + architecture info).
- `class_to_idx.json` and optional `config.json`.

Why this matters:
- You can load these artifacts later without rerunning full training.

## Version-by-Version Summary

### `detection.ipynb` (baseline)

- EfficientNet-B0 transfer learning.
- Class-aware split.
- Basic strong training loop with checkpointing and export.
- Includes inference helper scaffold.

Use when:
- You want the simplest complete pipeline.

### `detection_v2.ipynb`

- EfficientNet-B2.
- Better augmentation and MixUp.
- Colab-friendly zip extraction.
- Better scheduling and fine-tuning control.

Use when:
- You want better baseline quality with modest complexity.

### `detection_v3.ipynb`

- Adds SWA, CutMix + MixUp alternation, gradient accumulation.
- Optional advanced backbones (`noisy_student`, `dinov2`).
- Additional anti-overfit tuning logic.

Use when:
- You are chasing higher generalization and can handle more knobs.

### `detection_v4.ipynb`

- Builds balanced dataset to disk before training.
- Uses conservative online augmentation on top of balanced data.
- Noisy-Student B2 default, warmup/cosine schedule, SWA, TTA flow.

Use when:
- Class imbalance is your main bottleneck.

### `acne_classification_pytorch (1).ipynb`

- Kaggle dataset download inside notebook.
- EfficientNet-B3 setup.
- Weighted sampling and class-weighted loss.
- Early stopping and thorough metrics.

Use when:
- You want a self-contained Kaggle-based acne classification experiment.

## Expected Dataset Structure

Most notebooks expect folder-per-class style data:

```text
dataset_root/
	ClassA/
		img1.jpg
		img2.jpg
	ClassB/
		img3.jpg
		...
	ClassC/
		...
```

If your data is in zip format for Colab, the v2/v3/v4 notebooks include extraction logic.

## How To Run

1. Open any notebook in Jupyter/VS Code.
2. Set dataset paths (or use Colab zip/Kaggle mode).
3. Run cells top to bottom.
4. Watch training metrics.
5. Use saved artifacts for evaluation or inference.

Recommended order if you are learning the project evolution:

1. `detection.ipynb`
2. `detection_v2.ipynb`
3. `detection_v3.ipynb`
4. `detection_v4.ipynb`

## Practical Notes

- Some stored notebook outputs show old runtime errors (for example, `name 'model' is not defined`). Those are from previous interrupted sessions, not necessarily current code logic.
- If you rerun notebook cells in order from a clean kernel, pipeline state is usually consistent.
- GPU strongly improves training time.

## Which Notebook Should You Use Right Now?

- Best all-around default for this repo: `detection_v4.ipynb`.
- Most configurable research notebook: `detection_v3.ipynb`.
- Simplest starting point: `detection.ipynb`.
- Kaggle-centric standalone workflow: `acne_classification_pytorch (1).ipynb`.