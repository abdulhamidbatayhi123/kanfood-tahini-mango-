# Phase 1 - Tahini Benchmark (leakage-free, tuned, SNV -> PLS -> model)

- Samples: 1554 spectra, 55 physical samples (isim)
- Split: GROUP hold-out by isim - train 1146 / test 408 spectra, no shared sample
- Pipeline: SNV -> PLS latent scores -> model. Score models use a 12-component PLS front-end by default; KAN's compression is a tuned hyperparameter. PLS & 1-D CNN use the full spectrum.
- Hyperparameters chosen by 3-fold inner GroupKFold on TRAIN only (equal light tuning):
  - PLS: {'n_components': 12}
  - SVM: {'C': 100.0, 'gamma': 0.05}
  - RF: {'n_estimators': 300, 'max_depth': 10}
  - MLP: {'hidden': (64, 32), 'n_components': 8}
  - CNN: {'channels': (16, 32), 'lr': 0.001}
  - KAN: {'width_hidden': (3,), 'grid': 5, 'n_components': 8}
- Detection threshold 94.0% (chosen on train CV).

## Group-CV summary (train, tuned configs)

| Method   |   R2_mean |   R2_std |   MAE_mean |   RPD_mean |   F1_mean |   time_s |
|:---------|----------:|---------:|-----------:|-----------:|----------:|---------:|
| PLS      |    0.9916 |   0.0046 |      1.619 |     13.837 |    0.9327 |     0.34 |
| SVM      |    0.8952 |   0.0542 |      6.492 |      3.559 |    0.6898 |     0.15 |
| RF       |    0.8871 |   0.0283 |      5.438 |      3.233 |    0.8457 |     0.9  |
| MLP      |    0.9799 |   0.006  |      2.44  |      7.881 |    0.9176 |    18.11 |
| CNN      |    0.946  |   0.0187 |      3.886 |      4.944 |    0.8199 |    48.11 |
| KAN      |    0.9421 |   0.031  |      3.702 |      6.487 |    0.8623 |     7.25 |

## Held-out test (group-independent)

| Method   |   R2_tahini | R2_CI         |   MAE_tahini |   RPD_tahini |   R2_mean3 |     F1 |    Acc |   n_params |
|:---------|------------:|:--------------|-------------:|-------------:|-----------:|-------:|-------:|-----------:|
| PLS      |      0.9943 | [0.978,0.995] |        2.111 |       13.271 |     0.9953 | 1      | 1      |      21180 |
| SVM      |      0.9246 | [0.816,0.973] |        7.003 |        3.641 |     0.9469 | 0.7883 | 0.6618 |        nan |
| RF       |      0.919  | [0.662,0.947] |        6.945 |        3.514 |     0.9351 | 0.9298 | 0.9093 |        nan |
| MLP      |      0.9882 | [0.955,0.992] |        3.157 |        9.208 |     0.9907 | 0.9981 | 0.9975 |       2947 |
| CNN      |      0.9674 | [0.836,0.986] |        3.934 |        5.537 |     0.9761 | 0.9804 | 0.9755 |      19363 |
| KAN      |      0.9869 | [0.946,0.992] |        3.031 |        8.73  |     0.9881 | 1      | 1      |        684 |

## KAN vs baselines (paired t-test on CV folds)

| Comparison   |    diff |   p_corrected |   p_holm | sig_holm   |
|:-------------|--------:|--------------:|---------:|:-----------|
| KAN vs PLS   | -0.0495 |        0.0712 |   0.3562 | False      |
| KAN vs SVM   |  0.0468 |        0.0944 |   0.363  | False      |
| KAN vs RF    |  0.055  |        0.0907 |   0.363  | False      |
| KAN vs MLP   | -0.0378 |        0.1452 |   0.363  | False      |
| KAN vs CNN   | -0.0039 |        0.8522 |   0.8522 | False      |

## Honest reading

- Leakage-free (group split); every model tuned with the same inner-CV budget and trained to convergence.
- KAN is a single, stable model (no ensembling) -> one equation for interpretability.
- Tahini blending is ~linear, so PLS is very strong; KAN's value is interpretability + cross-food transfer, not winning tahini accuracy.