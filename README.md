# 🌌 Predicting Stellar Class — Kaggle Playground S6E6

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E6-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e6)
[![Balanced Accuracy](https://img.shields.io/badge/Balanced%20Accuracy-0.9697-success)](#results)
[![Rank](https://img.shields.io/badge/Rank-top%20~25%25%20(136%2F544)-brightgreen)](#results)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-9ACD32)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EB0F00)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Solution for [Kaggle Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6):
classify an astronomical object as **GALAXY / QSO / STAR** from SDSS photometry.
Metric: **Balanced Accuracy** (mean per-class recall — robust to class imbalance).

| | Score |
|---|---|
| **Public Balanced Accuracy (GBDT + RealMLP blend)** | **0.96968** |
| GBDT ensemble (LGBM+XGB+CatBoost) | 0.96616 |
| Single LightGBM | 0.96532 |
| Rank | **~136 / 544 (top ~25%)** |
| Public top-1 | 0.97127 |

## The task

~577k train rows, ~247k test. Features: sky coordinates (`alpha`, `delta`),
SDSS photometric magnitudes (`u, g, r, i, z`), `redshift`, plus two categoricals
(`spectral_type`, `galaxy_population`). Classes are **imbalanced**
(GALAXY 65% / QSO 20% / STAR 14%), which is exactly why the metric is *balanced*
accuracy rather than plain accuracy.

## Approach

1. **EDA** (`src/eda.py`)
   - `redshift` is the dominant separator: STAR ≈ 0.07, GALAXY ≈ 0.51, QSO ≈ 1.88.
   - No missing values; two low-cardinality categoricals.

2. **Feature engineering** (inside the train scripts)
   - **Astronomical color indices**: `u-g, g-r, r-i, i-z, u-r, g-i, ...` — the
     physically meaningful quantities for star/galaxy/QSO separation.
   - `redshift_log`, a near-zero-redshift flag (stars), magnitude summary stats.

3. **Models + metric-aware decision rule**
   - LightGBM / XGBoost / CatBoost, multi-class, 5-fold stratified CV.
   - `class_weight='balanced'` so minority classes aren't ignored.
   - **Per-class score multipliers tuned on OOF** to maximize *balanced* accuracy
     (a generalization of prior-correction `argmax(p / prior)`).

4. **Ensemble** (`src/train_ensemble_gpu.py`, runs on Kaggle GPU)
   - Probability average of the three GBDTs + tuned class weights `[0.9, 1.3, 1.7]`
     → OOF balanced accuracy **0.96521**.

## Results

| Model | OOF Balanced Acc | Public |
|---|---|---|
| LightGBM (single) | 0.96453 | 0.96532 |
| XGBoost (single) | 0.96289 | — |
| CatBoost (single) | 0.96194 | — |
| GBDT ensemble (tuned weights) | 0.96521 | 0.96616 |
| **GBDT + RealMLP blend** | **0.96911** | **0.96968** |

The CV tracks the public leaderboard closely → the validation is trustworthy.

### The blend that moved us ~100 places

GBDTs are correlated with each other, so stacking three of them barely helps.
A tabular **RealMLP neural net** is an *independent* signal — blending GBDT + RealMLP
and re-tuning per-class weights for Balanced Accuracy jumped the score from
**0.96616 → 0.96968** and the rank from ~210 to **~136 / 544**. Same lesson as on the
sibling F1 project: *independent signal (GBDT + NN) beats more of the same.*

> **Attribution:** the RealMLP OOF/test probabilities come from the public notebook
> [*PS|S6|E6: RealMLP / PyTorch* by Vladimir Demidov (yekenot)](https://www.kaggle.com/code/yekenot/ps-s6-e6-realmlp-pytorch).
> They are **not** redistributed here; `src/blend_with_realmlp.py` only reads them.
> All feature engineering, the GBDT pipeline, CV and the blend/metric logic are this repo's own work.

## Repo structure

```
src/
  eda.py                  # data exploration (class balance, redshift split)
  train_lgbm.py           # single LightGBM + color features + prior-adjusted decision
  train_ensemble_gpu.py   # LGBM + XGB + CatBoost, 5-fold CV, class-weight tuning (Kaggle GPU)
  blend_with_realmlp.py   # GBDT + RealMLP (NN) blend, OOF-verified  (best result)
kaggle/
  kernel-metadata.json    # config to run the ensemble as a Kaggle GPU notebook
results/
  results.json            # all scores in one place
```

## Reproduce

```bash
pip install -r requirements.txt
kaggle competitions download -c playground-series-s6e6 -p data && (cd data && unzip '*.zip')
python src/eda.py
python src/train_lgbm.py            # single model -> submission_lgbm.csv
# ensemble: run src/train_ensemble_gpu.py (CPU works; GPU on Kaggle is faster)
```

## Lessons

- For **balanced accuracy** on imbalanced classes, the *decision rule* matters as
  much as the model: tuning per-class multipliers on OOF beats plain `argmax`.
- Domain features win: SDSS **color indices** + `redshift` carry almost all the
  separating signal between stars, galaxies and quasars.
- A trustworthy out-of-fold CV (here matching public to ~0.0008) is worth more
  than chasing the leaderboard.

## License

MIT — see [LICENSE](LICENSE).
