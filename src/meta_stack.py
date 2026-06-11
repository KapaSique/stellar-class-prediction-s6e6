"""S6E6 mega-stack — LR meta over 3 strong decorrelated signals. Runs on Kaggle.
Inputs (kernel sources):
  - ssstelmah/s6e6-stellar-gpu-ensemble     -> oof_ens.npy / pred_ens.npy      (my GBDT ensemble)
  - yekenot/ps-s6-e6-realmlp-pytorch        -> oof_preds.csv / test_preds.csv  (RealMLP NN)
  - cdeotte/gpu-logistic-regression-stacker -> oof_lr_stacker_v9.npy / pred_lr_stacker_v9.npy
LR meta (nested 5-fold, class_weight=balanced) + per-class weight tuning.
"""
import glob, numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from itertools import product

def find(pat):
    g=glob.glob(f"/kaggle/input/**/{pat}",recursive=True)
    if not g: raise FileNotFoundError(pat)
    return g[0]

classes=["GALAXY","QSO","STAR"]
train=pd.read_csv(find("train.csv"))
if "class" not in train.columns:
    for p in glob.glob("/kaggle/input/**/train.csv",recursive=True):
        t=pd.read_csv(p)
        if "class" in t.columns: train=t; break
test=pd.read_csv(find("test.csv"))
y=train["class"].map({c:i for i,c in enumerate(classes)}).values
N=len(train)

def norm(P):
    P=np.clip(P,1e-9,None); return P/P.sum(1,keepdims=True)

sigs={}
sigs["gbdt"]=(norm(np.load(find("oof_ens.npy")).astype(float)), norm(np.load(find("pred_ens.npy")).astype(float)))
sigs["realmlp"]=(norm(pd.read_csv(find("oof_preds.csv")).set_index("id").loc[train["id"],classes].values),
                 norm(pd.read_csv(find("test_preds.csv")).set_index("id").loc[test["id"],classes].values))
sigs["cdeotte"]=(norm(np.load(find("oof_lr_stacker_v9.npy")).astype(float)), norm(np.load(find("pred_lr_stacker_v9.npy")).astype(float)))

ba=lambda P: balanced_accuracy_score(y,P.argmax(1))
for n,(o,_) in sigs.items(): print(f"{n}: OOF BA={ba(o):.5f}",flush=True)

names=list(sigs.keys())
# features for meta: raw probs + log-probs (richer for LR)
def feats(mats):
    X=np.hstack(mats)
    return np.hstack([X, np.log(np.clip(X,1e-9,None))])
OOF=feats([sigs[n][0] for n in names]); PRED=feats([sigs[n][1] for n in names])

skf=StratifiedKFold(5,shuffle=True,random_state=42)
meta_oof=np.zeros((N,3))
for tr,va in skf.split(OOF,y):
    lr=LogisticRegression(C=1.0,max_iter=4000,class_weight="balanced")
    lr.fit(OOF[tr],y[tr]); meta_oof[va]=lr.predict_proba(OOF[va])
lr=LogisticRegression(C=1.0,max_iter=4000,class_weight="balanced"); lr.fit(OOF,y)
meta_pred=lr.predict_proba(PRED)
print(f"META(LR, probs+logprobs) OOF BA={ba(meta_oof):.5f}",flush=True)

baw=lambda P,w: balanced_accuracy_score(y,(P*w).argmax(1))
best=(np.ones(3),baw(meta_oof,np.ones(3)))
for w in product(np.linspace(0.7,1.6,19),repeat=3):
    s=baw(meta_oof,np.array(w))
    if s>best[1]: best=(np.array(w),s)
cw,score=best
print(f"tuned class_w={cw.round(3)} -> META OOF BA={score:.5f}",flush=True)

# fallback: if meta worse than best single (cdeotte tuned), use that
cd_oof=sigs["cdeotte"][0]
best_cd=(np.ones(3),baw(cd_oof,np.ones(3)))
for w in product(np.linspace(0.7,1.6,19),repeat=3):
    s=baw(cd_oof,np.array(w))
    if s>best_cd[1]: best_cd=(np.array(w),s)
print(f"cdeotte tuned OOF={best_cd[1]:.5f}",flush=True)

if score>=best_cd[1]:
    final=(meta_pred*cw).argmax(1); used=f"meta score={score:.5f}"
else:
    final=(sigs["cdeotte"][1]*best_cd[0]).argmax(1); used=f"cdeotte-tuned score={best_cd[1]:.5f}"
print("USED:",used,flush=True)
pd.DataFrame({"id":test["id"],"class":[classes[i] for i in final]}).to_csv("/kaggle/working/submission.csv",index=False)
with open("/kaggle/working/scores.txt","w") as f:
    f.write(", ".join(f"{n}={ba(sigs[n][0]):.5f}" for n in names)+"\n")
    f.write(f"meta={ba(meta_oof):.5f} meta_tuned={score:.5f} cdeotte_tuned={best_cd[1]:.5f} used={used}\n")
print("saved submission.csv",flush=True)
