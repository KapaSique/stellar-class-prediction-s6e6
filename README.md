<h1 align="center">🌌 Predicting Stellar Class — Kaggle Playground S6E6</h1>

<p align="center">
  <a href="https://www.kaggle.com/competitions/playground-series-s6e6"><img src="https://img.shields.io/badge/Kaggle-Playground%20S6E6-20BEFF?logo=kaggle&logoColor=white"></a>
  <img src="https://img.shields.io/badge/Balanced%20Accuracy-0.9711-success">
  <img src="https://img.shields.io/badge/Leaderboard-top%20~8%25%20(~108%2F1300)-brightgreen">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/LightGBM-9ACD32"><img src="https://img.shields.io/badge/XGBoost-EB0F00"><img src="https://img.shields.io/badge/CatBoost-FFCC00"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

Multiclass classification of astronomical objects (**GALAXY / QSO / STAR**) from SDSS photometry,
redshift and spectral features. Metric: **Balanced Accuracy**. ~577k train rows, 3 imbalanced
classes (65 / 20 / 14 %).

| Result | Balanced Acc |
|---|---|
| 🛠️ **My own model** (GBDT ensemble + RealMLP blend) | **0.96990** |
| 🏅 **Best leaderboard** (mega-stack incl. public stacker) | **0.97106** → top ~8% |
| GBDT ensemble (LGBM + XGB + CatBoost) | 0.96616 |
| Single LightGBM baseline | 0.96532 |

The headline isn't only the score — it's an **honest, out-of-fold-validated pipeline** plus a
documented engineering journey (including diagnosing a real Kaggle GPU/driver mismatch) and a
clear-eyed account of *what actually separates a strong solo solution from a grandmaster's*.

---

## Approach (each step OOF-validated)

1. **EDA + leakage checks** ([`src/eda.py`](src/eda.py)) — redshift almost separates the classes
   (STAR ≈ 0.07, GALAXY ≈ 0.51, QSO ≈ 1.88); no leakage, classes imbalanced.
2. **Feature engineering** — SDSS **color indices** (u−g, g−r, r−i, i−z …), redshift transforms,
   magnitude stats. The decision rule is tuned with **per-class weights** because Balanced Accuracy
   rewards minority-class recall.
3. **GBDT ensemble** ([`src/train_ensemble_gpu.py`](src/train_ensemble_gpu.py)) — LightGBM + XGBoost
   + CatBoost, 5-fold CV, GPU → **0.96616**.
4. **Add a neural net** ([`src/blend_with_realmlp.py`](src/blend_with_realmlp.py)) — a tabular
   **RealMLP** is decorrelated from GBDTs (rank-corr ~0.97); blending + class-weight tuning lifts the
   **honest OOF to 0.96911 / public 0.96990** — my best own result.
5. **My own 2-level stacker** ([`src/train_stacker_gpu.py`](src/train_stacker_gpu.py)) — implemented
   the grandmaster-style approach myself: base models (GBDTs + PyTorch MLPs) → **logistic-regression
   meta-model** on nested-OOF probabilities.
6. **Mega-stack** ([`src/meta_stack.py`](src/meta_stack.py)) — LR meta over the strongest
   decorrelated signals → **0.97106 (top ~8%)**.

## The engineering journey (this is the interesting part)

Chasing the leaderboard turned into a real systems-debugging exercise:

- **Stacking ≠ magic.** My first LR-stacker (5 GBDTs + RealMLP) gave 0.969 — *no better than a plain
  blend*. Reason: my four GBDTs were ~0.99 correlated, so the meta-model had nothing new to combine.
  Stacking only pays off with **diverse, strong base models**.
- **A genuine GPU bug.** My PyTorch NNs crashed with `CUDA error: no kernel image`. I wrote a
  [diagnostic kernel](src/gpu_diagnose.py) and found Kaggle had assigned a **Tesla P100 (sm_60)**,
  while the preinstalled `torch 2.10+cu128` only supports **sm_70+**. Fix: reinstall **torch cu118**
  before import → NNs train on GPU. ([`src/train_stacker_gpu.py`](src/train_stacker_gpu.py) ships the fix.)
- **Reproducing RealMLP is research, not a one-liner.** My out-of-the-box `pytabkit` RealMLP runs
  scored 0.946 vs. the tuned public RealMLP's 0.969. The gap is careful NN architecture tuning —
  days of work, not a flag.

**Conclusion:** anything blended on top of a single strong NN signal plateaus near **0.970 OOF**.
The real lever to climb further is **building diverse, strong base learners (especially NNs)** — that
is the grandmaster's edge, not the stacker itself.

## Results

| Model | OOF Bal-Acc | Public |
|---|---|---|
| LightGBM (single) | 0.96453 | 0.96532 |
| GBDT ensemble (tuned weights) | 0.96520 | 0.96616 |
| **GBDT + RealMLP blend (my best own)** | **0.96911** | **0.96990** |
| My LR-stacker (GBDTs + PyTorch NN) | 0.96895 | 0.96934 |
| Mega-stack (LR meta, incl. public stacker) | 0.97031 | **0.97106** |

CV tracks public to ~0.0005 → trustworthy validation, no leaderboard overfit.

## Repo structure

```
src/
  eda.py                 # EDA + class balance + leakage checks
  train_lgbm.py          # single LightGBM + color features + prior-adjusted decision
  train_ensemble_gpu.py  # LGBM + XGB + CatBoost, 5-fold CV, class-weight tuning (Kaggle GPU)
  blend_with_realmlp.py  # GBDT + RealMLP blend, OOF-verified  (best own result)
  train_stacker_gpu.py   # my 2-level stacker: GBDTs + PyTorch NN -> LR meta  (+ P100 torch cu118 fix)
  meta_stack.py          # LR meta over decorrelated signals (probs + log-probs)
  gpu_diagnose.py        # the kernel that found the P100 sm_60 vs torch mismatch
results/results.json     # all scores in one place
```

## Reproduce

```bash
pip install -r requirements.txt
kaggle competitions download -c playground-series-s6e6 && unzip '*.zip' -d data
python src/eda.py --data-dir data
python src/train_ensemble_gpu.py          # GBDT ensemble (GPU on Kaggle)
python src/blend_with_realmlp.py ...       # blend with external RealMLP OOF/test
```

## Attribution & honesty

- **Mine:** all feature engineering, the GBDT ensemble + CV, the **stacker implementation**, the
  per-class Balanced-Accuracy tuning, the GPU diagnosis/fix, and the blend logic.
- **External signals (attributed, not redistributed):** RealMLP OOF/test from the public notebook
  [*PS|S6|E6: RealMLP* by Vladimir Demidov (yekenot)](https://www.kaggle.com/code/yekenot/ps-s6-e6-realmlp-pytorch);
  the top leaderboard submission additionally blends a public LR-stacker by Chris Deotte. Competition
  data and third-party files are excluded from this repo (`.gitignore`).
- Best **fully-own** result reported separately (**0.96990**) from leaderboard-max (**0.97106**).

## Lessons learned

- A trustworthy **out-of-fold CV** beats chasing the public LB — here it predicted public to ~0.0005.
- **Stacking amplifies diversity it's given** — ten correlated GBDTs ≈ one GBDT. Independent signal wins.
- Real ML work includes **infra debugging** (the P100/torch mismatch cost more than the modeling).
- Know where your edge ends: I could *implement* everything a grandmaster does, but their advantage is
  the **base-model craft** — which is exactly the next skill to build.

## License

MIT — see [LICENSE](LICENSE).
