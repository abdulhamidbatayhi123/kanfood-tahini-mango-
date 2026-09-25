# kanfood — interpretable Kolmogorov–Arnold networks for food spectroscopy

`kanfood` is a leakage-free machine-learning pipeline for **food adulteration and quality analysis by
vibrational spectroscopy** (FTIR and NIR). It benchmarks a **Kolmogorov–Arnold network (KAN)** used as an
*interpretable glass-box* model against standard baselines (PLS, support-vector regression, random forest,
a multilayer perceptron and a 1-D CNN), and reads the trained KAN out as a **closed-form symbolic equation**
together with chemically meaningful per-feature response curves.

This is the reference implementation and open dataset for the paper
*"Interpretability without sacrificing accuracy: Kolmogorov–Arnold networks with closed-form calibration
equations for food adulteration and quality analysis by vibrational spectroscopy"* (submitted to
*Artificial Intelligence in Agriculture*). **All code and the tahini dataset are released openly** for full
reproducibility and transparency.

## Honest summary of results

The KAN **matches/ties the strongest models while staying interpretable and compact** — it does *not*
dominate on accuracy. Every model was given a **comparable, equal-budget hyperparameter search**, confidence
intervals come from a **cluster (group) bootstrap**, and model comparisons use the **Nadeau–Bengio corrected
resampled t-test with Holm correction**.

| Task | Regime | KAN test R² | Where it lands | Params | Symbolic eq. R² |
|------|--------|-------------|----------------|--------|-----------------|
| Tahini adulteration (FTIR) | ~linear | 0.987 | ties PLS (0.994) and MLP (0.988) | **684** (smallest) | 0.948 |
| Mango dry-matter (NIR) | nonlinear | **0.863** (highest) | statistical tie with MLP (0.861), CNN, PLS, SVM | 8,996 | 0.823 |

- **Tahini:** after correction, **no pairwise difference is statistically significant** — the interpretable
  KAN is on par with every baseline, including PLS, while being by far the smallest model.
- **Mango:** the KAN has the **highest external-test R²** even under the equal-budget search; in
  cross-validation it significantly exceeds only the random forest and is tied with the other models.

Uniquely among the models, the KAN yields a **closed-form symbolic equation** relating spectral features to
composition, and its intrinsic importances agree with post-hoc SHAP/permutation attributions.

All validation is **leakage-free**: tahini is split by physical sample (grouped — replicate scans of the same
sample are never split across train/test); mango uses the published across-season external test (train on
Seasons 1–3, test on Season 4, no shared population).

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

## Revision analyses (`revision/`)

The analyses reported in Sections S4 and S14–S23 of the Supplementary Materials of the revised article.
Each script writes CSV files to `revision/results/<id>/`; the tables of the Supplementary Materials are
formatted views of these files.

| Script | Supplementary section |
|---|---|
| `r01_nested_generalisation.py` | S4, Table S3: leave-one-lot-out / leave-one-season-out with selection nested in each fold (and the frozen control) |
| `r03_low_level_authentic_bootstrap.py` | S16–S17: low-level identification, false positives on authentic tahini, two-stage lot/sample bootstrap |
| `r04_repeated_grouped_splits.py` | S23: 30 repeated grouped hold-outs, corrected t-tests and equivalence tests |
| `r05_oof_lod.py` | Table 4 and S17: out-of-fold detection limits and their bootstrap |
| `r07_band_artifact_controls.py`, `r07b_band_confound_checks.py` | S14: artefact controls for the selected channels |
| `r08_mango_covariates.py` | S18: mango residuals by cultivar, ripening treatment and region |
| `r09_training_budget.py` | S21: training-budget crossover |
| `r10_learning_curve.py`, `figS9_learning_curve.py` | S20, Figure S9: learning curve |
| `r11_mango_input_representation.py` | S19: input representations for mango |
| `r12_band_count_overlap.py` | S15: number of channels and band overlap |
| `r13_kan_best_factors.py` | S22: the KAN with the favourable design choices combined |
| `r14_mango_mlp_tie.py` | S8: the near-tie in the mango MLP selection |
| `fig1_pipeline.py` | Figure 1 |

Run any of them as `python -m revision.<script>` from the repository root (mango data: `python -m kanfood.fetch_mango`).
`results_phase1/` and `results_mango/` hold the reproduced benchmark (Tables 2 and 3); the mango MLP configuration
in `results_mango/mango_meta.json` is pinned to the published one, and the configuration selected by the rerun is kept
in `mango_meta.reproduced.json` (see Section S8).

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
| `phase2_robust.py`, `phase3_transfer.py` | fold-stability / LOD and cross-food analyses |

## Citation

If you use this code or the tahini dataset, please cite the paper (forthcoming) and this repository — see
`CITATION.cff`.

## License

- **Code:** MIT — see `LICENSE`.
- **Tahini dataset** (`data/tahini/`): Creative Commons Attribution 4.0 (CC-BY-4.0) — see `LICENSE-DATA`.
