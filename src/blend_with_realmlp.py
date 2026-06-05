"""Blend the GBDT model (this repo) with a RealMLP neural-net signal.

WHY: GBDTs share most of their structure; a tabular neural net (RealMLP) gives an
independent signal. Blending GBDT + RealMLP and tuning per-class weights for
Balanced Accuracy lifts the OOF score from ~0.9635 (GBDT) to ~0.9691 — verified
out-of-fold — and public Balanced Accuracy to 0.96968.

EXTERNAL SOURCE / ATTRIBUTION:
The RealMLP out-of-fold and test probabilities are NOT produced by this repo.
They come from the public Kaggle notebook:
    "PS|S6|E6: RealMLP / PyTorch" by Vladimir Demidov (yekenot)
    https://www.kaggle.com/code/yekenot/ps-s6-e6-realmlp-pytorch
That notebook publishes `oof_preds.csv` and `test_preds.csv` (class probabilities).
We only read those files to demonstrate the GBDT+NN blend; we do not redistribute
them here.

Usage:
    python src/blend_with_realmlp.py \
        --gbdt-oof oof_gbdt.npy --gbdt-pred pred_gbdt.npy \
        --realmlp-oof realmlp/oof_preds.csv --realmlp-test realmlp/test_preds.csv \
        --train data/train.csv --test data/test.csv --out submission_blend.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from itertools import product

CLASSES = ["GALAXY", "QSO", "STAR"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbdt-oof", required=True, help="npy (N,3) GBDT OOF probabilities")
    ap.add_argument("--gbdt-pred", required=True, help="npy (M,3) GBDT test probabilities")
    ap.add_argument("--realmlp-oof", required=True)
    ap.add_argument("--realmlp-test", required=True)
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="submission_blend.csv")
    args = ap.parse_args()

    train = pd.read_csv(args.train); test = pd.read_csv(args.test)
    y = train["class"].map({c: i for i, c in enumerate(CLASSES)}).values

    my_oof = np.load(args.gbdt_oof); my_pred = np.load(args.gbdt_pred)
    rm_oof = pd.read_csv(args.realmlp_oof).set_index("id").loc[train["id"], CLASSES].values
    rm_pred = pd.read_csv(args.realmlp_test).set_index("id").loc[test["id"], CLASSES].values

    ba = lambda P: balanced_accuracy_score(y, P.argmax(1))
    baw = lambda P, w: balanced_accuracy_score(y, (P * w).argmax(1))
    print(f"GBDT    OOF BA = {ba(my_oof):.5f}")
    print(f"RealMLP OOF BA = {ba(rm_oof):.5f}")

    best = (None, None, -1)
    for a in np.linspace(0, 1, 21):
        B = a * my_oof + (1 - a) * rm_oof
        for w in product(np.linspace(0.7, 1.6, 10), repeat=3):
            s = baw(B, np.array(w))
            if s > best[2]:
                best = (a, np.array(w), s)
    a, w, s = best
    print(f"best blend: w_gbdt={a:.2f} class_w={w.round(2)} -> OOF BA={s:.5f}")

    final = ((a * my_pred + (1 - a) * rm_pred) * w).argmax(1)
    pd.DataFrame({"id": test["id"], "class": [CLASSES[i] for i in final]}).to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
