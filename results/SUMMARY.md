# Summary — Comparison of CNN models in pneumonia image classification

**Question.** Which convolutional neural network best separates pneumonia from normal chest X-rays of children, when
every model gets exactly the same data, training settings and evaluation?

**What we did.** Six models — a small CNN trained from scratch and five ImageNet-pretrained networks — were trained on
the Kermany *Chest X-Ray Images (Pneumonia)* dataset with one shared pipeline: same patient-level train/val split,
224×224 images, augmentation, class-balancing sampler, AdamW, learning-rate schedule, 25-epoch budget with early
stopping, one Tesla T4 GPU. Each model was trained **3 times (seeds 42, 43, 44)**: 18 runs, ~1.7 GPU-hours of training
on Kaggle. All models were scored on the untouched official test set (624 images: 234 normal, 390 pneumonia).

## Main numbers (test set, mean ± std over 3 seeds)

| Model | **ROC-AUC** (primary) | Recall | Specificity | Accuracy | Train time | ms / image | Parameters |
|---|---|---|---|---|---|---|---|
| Baseline CNN (from scratch) | 0.935 ± 0.008 | 0.979 | 0.530 | 0.810 | 3.3 min | **1.0** | 0.4 M |
| VGG16 | **0.978** ± 0.008 | 0.980 | **0.803** | **0.914** | 10.5 min | 9.7 | 134.3 M |
| ResNet50 | **0.978** ± 0.001 | 0.985 | 0.768 | 0.904 | 5.3 min | 8.4 | 23.5 M |
| DenseNet121 | **0.978** ± 0.002 | 0.991 | 0.746 | 0.900 | 6.3 min | 24.5 | 7.0 M |
| EfficientNet-B0 | 0.971 ± 0.002 | **0.994** | 0.701 | 0.884 | 5.5 min | 13.1 | 4.0 M |
| MobileNetV3-Large | 0.975 ± 0.004 | 0.983 | 0.775 | 0.905 | 3.3 min | 10.3 | 4.2 M |

Full table with standard deviations of every metric: [comparison.md](comparison.md).
**Recall** = share of sick children found; **specificity** = share of healthy children correctly called healthy.

## The best model

- **By quality, there is no single winner.** VGG16, ResNet50 and DenseNet121 reach the same test ROC-AUC (0.978); the
  differences between them are smaller than the variation between seeds.
- **Recommended choice: ResNet50.** Top ROC-AUC, the most stable across seeds (± 0.001), mid-size and fast (8.4 ms per
  image), and the only model whose Grad-CAM attention stays on the lungs (see below).
- **Best for weak hardware: MobileNetV3-Large.** Almost the same quality (0.975) with 4.2 M parameters, the shortest
  training of the pretrained models and ~10 ms per image.
- **Transfer learning matters most:** every ImageNet-pretrained model beats the from-scratch baseline clearly
  (ROC-AUC ≈ 0.97–0.98 vs 0.935).

## Interesting findings

1. **Almost no sick child is missed, but many healthy children are flagged.** Recall is 0.98–0.99 for every model, while
   specificity at the standard 0.5 threshold is only 0.53–0.80: on average 46–110 of 234 healthy children are called
   sick. The models assign too high pneumonia probabilities; the ROC curves show that a better threshold would trade a
   little recall for much higher specificity.
2. **One training run is not enough.** Threshold-based metrics jump between seeds — ResNet50's specificity was 0.61 with
   seed 42 but 0.88 with seed 43; the baseline's ranged from 0.11 to 0.77. ROC-AUC barely moves (std ≤ 0.008). That is
   why ROC-AUC averaged over 3 seeds is our primary metric (team decision, [Plan.md](../docs/Plan.md) section 7).
3. **Validation looks much easier than test.** Validation ROC-AUC is ~0.999 for all pretrained models, test ROC-AUC
   ~0.975: the validation images come from the same source folders as training; the official test set is harder.
   Because validation is saturated, early stopping chooses between epochs that differ only in the 4th–5th decimal.
4. **Same score, different reasoning (Grad-CAM).** Share of Grad-CAM "heat" on the outer 10 % of the image (corners,
   "R" side marker, shoulders, neck — not lungs; 0.36 = evenly spread): ResNet50 **0.14**, EfficientNet-B0 0.29,
   DenseNet121 0.32, baseline 0.36, MobileNetV3 0.39, VGG16 0.42 (VGG16 also gave empty maps for 6 of 24 images). Some
   models may partly rely on shortcuts that would not transfer to X-rays from another hospital.
5. **Parameters ≠ speed.** DenseNet121 has 7 M parameters but is the slowest (24.5 ms per image); VGG16 has 134 M and
   takes 9.7 ms.
6. **Possible data overlap.** 170 pneumonia patient numbers appear in both the official training and test folders; the
   dataset does not say whether these are the same children. The test set was kept as published (comparability with
   other work), so this is listed as a limitation.

## Figures for the report

| Figure | Use it for |
|---|---|
| [comparison_metrics.png](figures/comparison_metrics.png) | Main results slide: all metrics per model, bars = mean, dots = seeds |
| [comparison_roc.png](figures/comparison_roc.png) | ROC curves of all models (with zoom) — quality independent of the threshold |
| [comparison_efficiency.png](figures/comparison_efficiency.png) | Quality vs speed vs size — the model-choice argument |
| [comparison_confusion_matrices.png](figures/comparison_confusion_matrices.png) | Where the errors are (false alarms on healthy children) |
| [comparison_training_curves.png](figures/comparison_training_curves.png) | Training behaviour, over-fitting, early stopping |
| [gradcam_comparison_seed42.jpg](figures/gradcam_comparison_seed42.jpg) | Where each model looks, same 8 test images |
| [gradcam_border_share.png](figures/gradcam_border_share.png) | The shortcut check in one chart |
| [eda_class_counts.png](figures/eda_class_counts.png), [eda_images_per_patient.png](figures/eda_images_per_patient.png) | Dataset section: imbalance, why we split by patient |

All figures are rebuilt from the committed result files with `python scripts/compare_results.py --data-dir
data/raw/chest_xray`; [notebooks/03_results.ipynb](../notebooks/03_results.ipynb) shows them with explanations.

## Limitations and possible next steps

- Decision threshold fixed at 0.5 — choosing it per model on the validation set (never on test) would raise
  specificity; listed as future work.
- One dataset from one hospital; an external test set (e.g. RSNA Pneumonia Detection) would show real generalization.
- Grad-CAM analysed on 8 fixed images per run; a lung-segmentation mask would make the shortcut check more precise.
- 3 seeds give a rough spread estimate; more seeds would allow formal significance tests.
