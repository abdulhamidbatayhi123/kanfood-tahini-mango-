# Mango DMC Benchmark (leakage-free interseason external validation, SG1 -> model)

- 11691 spectra, 112 populations (Pop), 103 channels in 684-990 nm.
- Split: train = Seasons 1-3 (Cal+Tuning, n=10243); test = Season 4 (Val Ext, n=1448); zero Pop overlap (true external validation).
- Single continuous target DM% (no composition normalization, no detection/F1).
- Hyperparameters by 3-fold inner GroupKFold on TRAIN by Pop (equal light tuning):
  - PLS: {'n_components': 24}
  - SVM: {'C': 10.0, 'gamma': 0.05}
  - RF: {'n_estimators': 300, 'max_depth': 20}
  - MLP: {'hidden': (128, 64), 'n_components': 16}
  - CNN: {'channels': (8, 16), 'lr': 0.0003}
  - KAN: {'width_hidden': (16, 8), 'grid': 5, 'n_components': 24}

## Group-CV summary (train, by Pop)

| Method   |   R2_mean |   R2_std |   MAE_mean |   RPD_mean |   time_s |
|:---------|----------:|---------:|-----------:|-----------:|---------:|
| PLS      |    0.854  |   0.0398 |      0.698 |      2.684 |     0.2  |
| SVM      |    0.822  |   0.0234 |      0.754 |      2.385 |     8.91 |
| RF       |    0.8042 |   0.0352 |      0.802 |      2.286 |     5.71 |
| MLP      |    0.8607 |   0.0175 |      0.683 |      2.696 |    92.56 |
| CNN      |    0.8353 |   0.0394 |      0.738 |      2.512 |    94.09 |
| KAN      |    0.8403 |   0.0297 |      0.725 |      2.532 |    13.02 |

## Held-out external test (Season 4)

| Method   |     R2 | R2_CI         |   RMSEP |    MAE |   RPD |   n_params |
|:---------|-------:|:--------------|--------:|-------:|------:|-----------:|
| PLS      | 0.8452 | [0.726,0.895] |  1.0498 | 0.8216 | 2.542 |       2496 |
| SVM      | 0.8439 | [0.755,0.885] |  1.0544 | 0.8045 | 2.531 |        nan |
| RF       | 0.7933 | [0.664,0.847] |  1.2132 | 0.9225 | 2.2   |        nan |
| MLP      | 0.8615 | [0.778,0.898] |  0.9933 | 0.7528 | 2.687 |      10881 |
| CNN      | 0.8445 | [0.740,0.894] |  1.0522 | 0.784  | 2.536 |       9041 |
| KAN      | 0.8629 | [0.777,0.901] |  0.9882 | 0.7527 | 2.7   |       8996 |

## KAN vs baselines (paired t-test on CV folds)

| Comparison   |    diff |   p_corrected |   p_holm | sig_holm   |
|:-------------|--------:|--------------:|---------:|:-----------|
| KAN vs PLS   | -0.0137 |        0.3505 |   1      | False      |
| KAN vs SVM   |  0.0183 |        0.4519 |   1      | False      |
| KAN vs RF    |  0.0361 |        0.0079 |   0.0395 | True       |
| KAN vs MLP   | -0.0204 |        0.1998 |   0.7994 | False      |
| KAN vs CNN   |  0.0049 |        0.6578 |   1      | False      |

## Honest reading

- Mango DMC is a large, genuinely nonlinear NIR task (the canonical 'CNN beats PLS' dataset).
- Under equal, leakage-free, equally-tuned comparison, KAN is competitive while remaining a glass-box (symbolic equation, fewest parameters).
- The published SOTA (CNN + heavy data augmentation) reaches lower RMSEP; this is an equal-budget, no-augmentation comparison that isolates model capability and is reported as such.