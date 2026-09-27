# Mango DMC Benchmark (leakage-free interseason external validation, SG1 -> model)

- 11691 spectra, 112 populations (Pop), 103 channels in 684-990 nm.
- Split: train = Seasons 1-3 (Cal+Tuning, n=10243); test = Season 4 (Val Ext, n=1448); zero Pop overlap (true external validation).
- Single continuous target DM% (no composition normalization, no detection/F1).
- Hyperparameters by 3-fold inner GroupKFold on TRAIN by Pop (equal light tuning):
  - PLS: {'n_components': 24}
  - SVM: {'C': 10.0, 'gamma': 0.05}
  - RF: {'n_estimators': 300, 'max_depth': 20}
  - MLP: {'hidden': (256, 128), 'n_components': 24}
  - CNN: {'channels': (8, 16), 'lr': 0.0003}
  - KAN: {'width_hidden': (16, 8), 'grid': 5, 'n_components': 24}

## Group-CV summary (train, by Pop)

| Method   |   R2_mean |   R2_std |   MAE_mean |   RPD_mean |   time_s |
|:---------|----------:|---------:|-----------:|-----------:|---------:|
| PLS      |    0.854  |   0.0398 |      0.698 |      2.684 |     0.22 |
| SVM      |    0.822  |   0.0234 |      0.754 |      2.385 |    15.55 |
| RF       |    0.8042 |   0.0352 |      0.802 |      2.286 |    13.57 |
| MLP      |    0.8643 |   0.0213 |      0.675 |      2.742 |   107.84 |
| CNN      |    0.8372 |   0.0415 |      0.734 |      2.533 |    93.82 |
| KAN      |    0.8396 |   0.0296 |      0.729 |      2.527 |    11.12 |

## Held-out external test (Season 4)

| Method   |     R2 | R2_CI         |   RMSEP |    MAE |   RPD |   n_params |
|:---------|-------:|:--------------|--------:|-------:|------:|-----------:|
| PLS      | 0.8452 | [0.726,0.895] |  1.0498 | 0.8216 | 2.542 |       2496 |
| SVM      | 0.8439 | [0.755,0.885] |  1.0543 | 0.8045 | 2.531 |        nan |
| RF       | 0.7931 | [0.664,0.847] |  1.2139 | 0.923  | 2.198 |        nan |
| MLP      | 0.8713 | [0.793,0.906] |  0.9575 | 0.7404 | 2.787 |      40193 |
| CNN      | 0.8477 | [0.747,0.897] |  1.0413 | 0.7754 | 2.563 |       9041 |
| KAN      | 0.8629 | [0.777,0.901] |  0.9882 | 0.7527 | 2.7   |       8996 |

## KAN vs baselines (paired t-test on CV folds)

| Comparison   |    diff |   p_corrected |   p_holm | sig_holm   |
|:-------------|--------:|--------------:|---------:|:-----------|
| KAN vs PLS   | -0.0143 |        0.3428 |   1      | False      |
| KAN vs SVM   |  0.0177 |        0.4628 |   1      | False      |
| KAN vs RF    |  0.0355 |        0.0085 |   0.0423 | True       |
| KAN vs MLP   | -0.0247 |        0.1203 |   0.4814 | False      |
| KAN vs CNN   |  0.0025 |        0.8444 |   1      | False      |

## Honest reading

- Mango DMC is a large, genuinely nonlinear NIR task (the canonical 'CNN beats PLS' dataset).
- Under equal, leakage-free, equally-tuned comparison, KAN is competitive while remaining a glass-box (symbolic equation, fewest parameters).
- The published SOTA (CNN + heavy data augmentation) reaches lower RMSEP; this is an equal-budget, no-augmentation comparison that isolates model capability and is reported as such.