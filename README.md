# Diabetic Retinopathy Detection from Fundus Images

Binary classification of diabetic retinopathy (DR vs. No DR) from color fundus
photographs, using two independent PyTorch pipelines: a convolutional network
trained entirely from scratch, and an ensemble of ImageNet-pretrained
backbones adapted via progressive fine-tuning. Built for a Kaggle-style
image-classification competition (Codabench) as a 2-person team project for
a Computer Vision graduate course.

**Final results (ROC-AUC on held-out test set):**

| Track | Public AUC | Private AUC |
|---|---|---|
| **Custom CNN** (from scratch) | 0.757 | 0.780 |
| **Fine-tuned ensemble** | 0.835 | 0.826 |

## Problem

Diabetic retinopathy is a leading cause of preventable blindness, and
progression can be slowed if the disease is caught early — but screening at
scale requires a lot of ophthalmologist time. The task here is to predict,
from a single fundus photograph, whether a patient shows any sign of
retinopathy.

- **Data**: 3,500 fundus images (2,000 train / 500 val / 1,000 test),
  labeled 0–4 on the clinical DR severity scale and binarized to `DR` /
  `No DR` (~27% positive). Left/right eye is recorded per image; right-eye
  images are horizontally mirrored so orientation is consistent across the
  dataset.
- **Metric**: ROC-AUC (`sklearn.metrics.roc_auc_score`), which sidesteps
  threshold selection and is robust to the class imbalance.
- **Two independent submission tracks**:
  - **CUSTOM** — a CNN architecture designed and trained from scratch, no
    pretrained weights allowed.
  - **FINE-TUNING** — any ImageNet-pretrained backbone, adapted to this task.

## Approach

### Shared preprocessing

```
CropByEye → Ben Graham normalization → Rescale → augmentation → crop → Normalize(ImageNet stats)
```

- **`CropByEye`** — intensity-threshold segmentation that crops each image
  down to the retinal disc, removing the black background border.
- **Ben Graham normalization** (`4×img − 4×GaussianBlur(img) + 128`, circular
  mask) — a local contrast-enhancement formula from the original Kaggle DR
  competition that makes microaneurysms and vessel edges much more visible.
  Must run on `uint8` input; see [What didn't work](#what-didnt-work) for why.
- **Class imbalance** is handled two ways at once: `BCEWithLogitsLoss` with
  `pos_weight ≈ 2.76` (loss-level reweighting) and a `WeightedRandomSampler`
  with inverse class-frequency weights (batch-level rebalancing).

### Track 1 — Custom CNN (`CustomNetV2`)

A compact 5-block CNN, kept deliberately small to avoid overfitting a
2,000-image training set with no pretrained prior:

```
[Conv3x3(same, no bias) → BatchNorm → ReLU → MaxPool2] × 5   (channels: 3→32→64→128→256→256)
→ Global Average Pooling
→ Dropout(0.2) → Linear(256, 128) → Dropout(0.4) → Linear(128, 1)
```

~1.0M parameters. Trained with AdamW (`lr=1e-3`, `wd=5e-3`),
`CosineAnnealingLR(T_max=50)`, up to 50 epochs with early stopping
(patience 10). Augmentation at 224px: random flips, ±15° rotation, color
jitter, random crop.

**Final submission**: 3 independent runs (seeds 42, 123, 456), each scored
with 4-pass test-time augmentation (original + h-flip + v-flip + rot180),
then uniformly averaged. Independent weight initialization gives enough
diversity for the ensemble to help; a fixed uniform average was used instead
of tuning per-model weights, to avoid overfitting the 500-image validation
set.

### Track 2 — Fine-tuned backbone ensemble

Three ImageNet-pretrained backbones, chosen for complementary inductive
biases, each fine-tuned independently and then blended:

| Backbone | Idea |
|---|---|
| DenseNet-121 | dense feature reuse across layers |
| DenseNet-169 | deeper dense reuse |
| ResNeXt-50 (32×4d) | grouped ("cardinality") convolutions |

Each backbone gets the same head: `Dropout(0.5) → Linear(features, 1)`.

**Two-stage progressive unfreezing**, per backbone:
- *Stage 1* — freeze everything except the classification head and the
  deepest block. AdamW `lr=3e-4`, 30 epochs, label smoothing `ε=0.05`.
- *Stage 2* — unfreeze one more block, drop the backbone LR to `3e-5`
  (head: `1e-4`), 3-epoch linear warmup then cosine decay. Gated behind a
  validation check (only runs if Stage 1 val AUC ≥ 0.752) and rolled back if
  it regresses relative to Stage 1.

Inputs are processed at higher resolution (`Rescale(416) → RandomCrop(380)`)
with a wide `±180°` rotation range — fundus images have no canonical
orientation, and early lesions (microaneurysms) are small enough that they
benefit from the extra resolution. All three backbones are fully
convolutional (GAP-based heads), so no architecture change was needed to
raise the input resolution.

**Final submission**: 4-pass TTA per backbone, then a validation-set grid
search over blend weights. Best found:
`0.35 × DenseNet121 + 0.15 × DenseNet169 + 0.50 × ResNeXt50` (val AUC 0.867).
A fourth backbone (SE-ResNeXt50) was trained but excluded — its predictions
were too correlated with plain ResNeXt50 to add ensemble diversity.

### What didn't work

Kept here because the debugging is arguably more informative than the final
recipe:

| Change tried | Why it was dropped |
|---|---|
| CLAHE before `CropByEye` | Alters global intensity so the fixed 0.10 threshold mis-segments the retinal disc. |
| Ben Graham without a `uint8` dtype guard | Silent numpy integer overflow collapsed images to `{0, 128}` — a near-uniform gray square. Tanked both tracks (custom 0.541, FT 0.558) until traced back to the dtype. |
| Pre-caching images at 224px on disk (5× training speedup) | Removes the per-epoch `RandomCrop` jitter, which turned out to be the main spatial regularizer on a dataset this small (−0.028 AUC). |
| SE blocks + residual connections on the custom CNN | Extra capacity overfits with no pretrained prior — SE alone dropped to 0.52 val AUC. |
| 380px input / ±180° rotation on the *custom* CNN | The custom architecture's pooling stack is tuned for a 224px input; at 380px, training from scratch failed to converge (0.53 val AUC). Aggressive augmentation only paid off on the pretrained backbones. |
| 8-pass TTA (adding rot90/rot270) | Ben Graham's circular mask + rectangular crops are not invariant to 90° rotation, so those extra views fell outside the training distribution and hurt both tracks. |
| Unfreezing a 3rd DenseNet block (Stage 3) | Over-parameterized for 2,000 training images (−0.013 FT AUC vs. stopping at two stages). |

### Iteration trail

Notebooks in `notebooks/archive/` are numbered in the order they were run,
each named after its one key change. A condensed view of how the score moved
(public / private AUC):

| Notebook | Key change | Custom AUC | FT AUC |
|---|---|---|---|
| `01_baseline_customnet_efficientnetb0` | CustomNet baseline + EfficientNet-B0 (SGD + StepLR) | 0.570 / 0.597 | 0.698 / 0.708 |
| `06_customnetv2_bengraham_weightedsampler_tta` | CustomNetV2 + Ben Graham + WeightedRandomSampler + 4-pass TTA | 0.766 / 0.743 | 0.770 / 0.744 |
| `17_se_v2_v3_ensemble_4pass_tta` | Custom: SE-ensemble; FT: ResNet50 two-stage unfreezing | 0.756 / 0.764 | 0.780 / 0.791 |
| `19_customnetv2_3seed_ensemble_final_custom` | **Final CUSTOM** — 3-seed `CustomNetV2` ensemble | **0.757 / 0.780** | — |
| `20_3backbone_weighted_ensemble_final_ft` → `final_notebook` | **Final FT** — validation-grid 3-backbone weighted ensemble, 380px | — | **0.835 / 0.826** |

The full numbered sequence (01–25) covers every architecture and
preprocessing variant that was tried, including the dead ends listed above;
`notebooks/archive/logs/` has the raw per-notebook score dumps
(`scores.txt`) these numbers are drawn from.

## Repo structure

```
.
├── notebooks/
│   ├── final_notebook.ipynb   # canonical, self-contained: data → both training pipelines → val AUC → submission
│   └── archive/                # iteration history, numbered 01–25 in run order, one key change each
│       ├── logs/                # score logs and run summaries the table above is drawn from
│       └── starter/             # course-provided starter notebook the project began from
├── docs/
│   └── history/edits/           # per-notebook change logs, numbered 01–13 in chronological order
├── data/                        # train/val/test CSVs + images (not committed, course-provided dataset)
└── CLAUDE.md                    # working notes for AI-assisted development on this repo
```
