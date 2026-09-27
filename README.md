# kanfood — Kolmogorov–Arnold networks as closed-form calibrations for food spectroscopy

This repository holds the code, the open tahini dataset and every result file behind the article

> Batayhi, A.; Özgölet, M.; Cankurt, H.; Arici, M. *Interpretable Machine Learning in Food Analysis:
> Closed-Form Chemometric Calibration of FTIR and NIR Spectroscopy Using Kolmogorov–Arnold Networks.*
> *Foods* (MDPI), manuscript foods-4561619 (under revision).

A Kolmogorov–Arnold network (KAN) is compared with PLS regression, support-vector regression (SVR), a random
forest (RF), a multilayer perceptron (MLP) and a one-dimensional convolutional network (CNN) under leakage-free,
group-aware validation, on two problems: tahini adulteration with sunflower and peanut paste measured by ATR–FTIR
(dataset collected for the study and released here) and dry matter in intact mango measured by NIR (public
benchmark of Anderson et al., 2020).

## Summary of findings (as reported in the article)

- **No accuracy advantage is claimed.** A parameter-matched MLP had the higher mean R² in all twelve configurations
  compared; on withheld production lots no difference between the leading models was significant.
- **The contribution is the closed-form equation.** Most of the accuracy lost in converting a trained KAN to an
  equation is due to the conversion procedure; correcting it raised the median external R² of the tahini equation
  from 0.952 to 0.988. The conversion is not reproducible run to run, whereas a KAN whose edge functions are
  expanded in an elementary basis (Chebyshev, powers, Gaussian) prints an equation identical to the network.
- **Screening, not confirmation.** With every authentic lot serving as the blank, detection limits were
  8.9–21.6% adulterant; blends at 4–12% were not quantified.

## Install

```bash
python -m pip install -r requirements.txt
```

Key dependencies: numpy, scipy, scikit-learn, pandas, torch, pykan, matplotlib, seaborn. The exact versions
used for the paper are pinned in `requirements-lock.txt`.

## Quickstart

```bash
# Windows PowerShell — set once for clean UTF-8 output and OpenMP coexistence:
#   $env:PYTHONIOENCODING="utf-8"; $env:KMP_DUPLICATE_LIB_OK="TRUE"

# Tahini (FTIR) — data ships with this repository under data/tahini/
python -m kanfood.run_experiment    # leakage-free benchmark -> results_phase1/
python -m kanfood.report            # publication figures + main table
python -m kanfood.interpret         # symbolic equation + response curves
python -m kanfood.phase2_robust     # fold-stability, IUPAC LOD, intrinsic-vs-SHAP importance

# Mango (NIR) — public data, downloaded on first run
python -m kanfood.fetch_mango       # downloads the public mango NIR data -> data/mango/raw/
python -m kanfood.run_mango         # leakage-free interseason benchmark -> results_mango/
python -m kanfood.report_mango
python -m kanfood.interpret_mango
```

## Datasets (all open)

| Dataset | Modality | Availability |
|---------|----------|--------------|
| **Tahini adulteration** | FTIR (mid-IR) | **Open** — collected for this study; ships in `data/tahini/` (CC-BY-4.0; see `data/tahini/README.md`) |
| Mango dry-matter content | NIR | **Public** — Anderson et al. (2020), Mendeley `46htwnp833` (fetched by `kanfood.fetch_mango`) |
| Edible-oil adulteration | ATR-FTIR | **Public** — Gilbraith et al. (2024), Mendeley `ctgg7k4m5g` |

Data locations are configurable through environment variables (defaults are repo-relative):
`KANFOOD_TAHINI_PATH`, `KANFOOD_MANGO_PATH`, `KANFOOD_OILS_PATH`.

## Reproducibility

- Fixed random seeds (42) throughout.
- Pre-processing and PLS compression are fitted on the training folds only — no information leakage.
- Every model receives an equal-budget hyperparameter search (the exact grids are in `kanfood/tune.py` and
  the `MANGO_GRIDS` in `kanfood/run_mango.py`).
- `python -m pytest -q` runs the test suite; tests that need a dataset skip automatically when it is absent.

## Where each result comes from

| Location | What it produces |
|---|---|
| `kanfood/run_experiment.py` → `results_phase1/` | Table 2 (tahini benchmark) and the predictions used for the hold-out columns of Table 4 |
| `kanfood/run_mango.py` → `results_mango/` | Table 3 (mango benchmark) |
| `paper1_rigor/` → `results_rigor/` | Sections 3.3–3.6 and S1–S13 (below) |
| `revision/` → `revision/results/` | Section S4 (nested rows of Table S3), Table 4 (out-of-fold columns) and Sections S14–S23 |

Main analyses (`paper1_rigor/`, run as `python -m paper1_rigor.<module>` from the repository root):

| Module | Article / Supplementary |
|---|---|
| `r05_lod_ci`, `r08_lod_cv` | Table 4 (hold-out detection limits and intervals), Section S13 |
| `r12_extraction_protocol` | Table 5, Figure 4 (conversion protocols) |
| `r19_basis_ablation`, `r19b_basis_mango`, `basis_kan` | Table 6, Figure 5 (exact elementary-basis route) |
| `r13_band_equation`, `make_band_equation` | Section 3.4, Table 7, Figure 6, Equation (3) |
| `r23_equation_under_shift` | Section 3.5, Table 8, Figure 7 |
| `r14_glassbox_baselines`, `r11_equal_interpretability` | Section 3.6, Table 9, Figure 8 |
| `r09_adulterant_type` | Section S2, Table S1 |
| `r07_robustness` | Section S3, Table S2 |
| `r04_generalisation` | Section S4, frozen rows of Table S3 |
| `r02_ablation` | Section S5, Table S4 |
| `r17_preprocessing_sweep` | Section S6, Table S5 |
| `r01_matched_mlp`, `r10_fair_capacity`, `r10b_config_vs_harness` | Section S7, Table S6 |
| `r20_reproducibility`, `r22_repro_selected_config` | Section S8, Table S7 |
| `r03_stability`, `r26_conversion_determinism`, `r27_fidelity_as_a_filter` | Section S9, Tables S8–S10 |
| `band_assignments`, `f03_interpretability` | Section S10, Table S11, Figures S5–S7 |
| `r06_shap`, `r15_attribution` | Section S11, Tables S12–S13, Figure S8 |
| `r24_stats_new_comparisons` | Section S12, Tables S14–S15 |
| `make_equations`, `equation` | Equations (1) and (2) |
| `f00`–`f06`, `build_tables`, `build_tables2` | figures and formatted tables (written to `submission_foods/`) |

Revision analyses (`revision/`, run as `python -m revision.<script>`):

| Script | Supplementary section |
|---|---|
| `r01_nested_generalisation.py` | S4, Table S3: leave-one-lot-out / leave-one-season-out with selection nested in each fold (and the frozen control) |
| `r03_low_level_authentic_bootstrap.py` | S16–S17: low-level identification, false positives on authentic tahini, two-stage lot/sample bootstrap |
| `r04_repeated_grouped_splits.py` | S23: 30 repeated grouped hold-outs, corrected t-tests and equivalence tests |
| `r05_oof_lod.py` | Table 4 (out-of-fold columns) and S17 |
| `r07_band_artifact_controls.py`, `r07b_band_confound_checks.py` | S14: artefact controls for the selected channels |
| `r08_mango_covariates.py` | S18: mango residuals by cultivar, ripening treatment and region |
| `r09_training_budget.py` | S21: training-budget crossover |
| `r10_learning_curve.py`, `figS9_learning_curve.py` | S20, Figure S9: learning curve |
| `r11_mango_input_representation.py` | S19: input representations for mango |
| `r12_band_count_overlap.py` | S15: number of channels and band overlap |
| `r13_kan_best_factors.py` | S22: the KAN with the favourable design choices combined |
| `r14_mango_mlp_tie.py` | S8: the near-tie in the mango MLP selection |
| `fig1_pipeline.py` | Figure 1 |

`results_phase1/` and `results_mango/` hold the benchmark exactly as published (Tables 2–4).
`revision/results/benchmark_reproduction/` holds the rerun of that benchmark with this code in a different
software environment, described in Section S8 (PLS, SVR, RF and the KAN reproduce exactly; the MLP and the CNN
to within 0.004 R² on tahini; on mango the inner search selected a different, near-tied MLP configuration, kept
in `mango_meta.reproduced.json`). Section S18 uses the predictions of that rerun.

Deep-learning results depend on the software environment; the exact versions are pinned in
`requirements-lock.txt`. Runs are seeded, and the seed of every quoted equation is given in the article.

## Package layout

| Module | Purpose |
|--------|---------|
| `data.py` | dataset loaders and the `SpectralDataset` container |
| `split.py` | group-aware (leakage-free) train/test splits and cross-validation folds |
| `preprocess.py` | SNV / MSC / Savitzky–Golay corrections (fit on train only) |
| `features.py` | PLS-score and mutual-information feature compression |
| `models.py` | PLS, SVR, random forest, MLP, 1-D CNN and KAN |
| `tune.py`, `validate.py` | equal-budget nested group cross-validation and tuning |
| `metrics.py` | metrics, cluster bootstrap, Nadeau–Bengio corrected t-test, Holm correction |
| `bands.py`, `figures.py` | FTIR band assignments and plotting helpers |
| `run_experiment.py`, `run_mango.py` | end-to-end benchmarks (tahini / mango) |
| `report*.py`, `interpret*.py` | figures, tables, symbolic equations, response curves |
| `phase2_robust.py`, `phase3_transfer.py` | earlier fold-stability and cross-food analyses (not used in the article) |

## Citation

If you use this code or the tahini dataset, please cite the article and this repository (concept DOI
https://doi.org/10.5281/zenodo.21446757, which always resolves to the latest version) — see `CITATION.cff`.

## License

- **Code:** MIT — see `LICENSE`.
- **Tahini dataset** (`data/tahini/`): Creative Commons Attribution 4.0 (CC-BY-4.0) — see `LICENSE-DATA`.
