"""S6E6 final stacker — strong PyTorch NNs + GBDT -> LR meta. Runs on Kaggle (P100).
Fix: reinstall torch cu118 first (P100 is sm_60, unsupported by default cu128 torch)."""
import subprocess, sys, glob, os, time, warnings; warnings.filterwarnings("ignore")
# --- make torch work on Tesla P100 (sm_60): install cu118 build BEFORE importing torch ---
subprocess.run([sys.executable,"-m","pip","install","-q","torch==2.4.1","--index-url","https://download.pytorch.org/whl/cu118"],check=False,timeout=900)

import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from itertools import product
t0=time.time()
def log(s): print(f"[{time.time()-t0:.0f}s] {s}",flush=True)

def find(pat):
    g=glob.glob(f"/kaggle/input/**/{pat}",recursive=True); return g[0] if g else None
classes=["GALAXY","QSO","STAR"]
tr_path=find("train.csv"); train=pd.read_csv(tr_path)
if "class" not in train.columns:
    for p in glob.glob("/kaggle/input/**/train.csv",recursive=True):
        t=pd.read_csv(p)
        if "class" in t.columns: train=t; tr_path=p; break
test=pd.read_csv(os.path.join(os.path.dirname(tr_path),"test.csv"))
y=train["class"].map({c:i for i,c in enumerate(classes)}).values
N=len(train); M=len(test)
cls,cnt=np.unique(y,return_counts=True); cw_arr=np.array([N/(3*c) for c in cnt],dtype=np.float32)
log(f"train={N} test={M}")

def fe(df):
    df=df.copy(); b=["u","g","r","i","z"]
    for a,c in [("u","g"),("g","r"),("r","i"),("i","z"),("u","r"),("g","i"),("u","z"),("u","i"),("g","z")]:
        df[f"{a}_{c}"]=df[a]-df[c]
    df["redshift_log"]=np.log1p(df["redshift"].clip(lower=0)); df["redshift_pos"]=(df["redshift"]>0.0035).astype(int)
    df["mean_mag"]=df[b].mean(1); df["std_mag"]=df[b].std(1); df["min_mag"]=df[b].min(1); df["max_mag"]=df[b].max(1)
    return df
train=fe(train); test=fe(test)
catc=["spectral_type","galaxy_population"]
for c in catc:
    cats=pd.concat([train[c],test[c]]).astype("category").cat.categories
    train[c]=pd.Categorical(train[c],categories=cats).codes; test[c]=pd.Categorical(test[c],categories=cats).codes
feats=[c for c in train.columns if c not in ["id","class"]]; cat_idx=[feats.index(c) for c in catc]
Xtr=train[feats].values.astype(np.float32); Xte=test[feats].values.astype(np.float32)
Xtr_df=train[feats].copy(); Xte_df=test[feats].copy()
for c in catc: Xtr_df[c]=Xtr_df[c].astype(int); Xte_df[c]=Xte_df[c].astype(int)
def onehot(df):
    parts=[df.drop(columns=catc).values.astype(np.float32)]
    for c in catc:
        k=int(max(train[c].max(),test[c].max()))+1
        parts.append(np.eye(k,dtype=np.float32)[df[c].astype(int).values])
    return np.hstack(parts)
Xtr_nn=onehot(train[feats]); Xte_nn=onehot(test[feats])
skf=StratifiedKFold(5,shuffle=True,random_state=42)
bases={}
def cv(fp,nm):
    oof=np.zeros((N,3)); pred=np.zeros((M,3))
    for tr,va in skf.split(Xtr,y):
        o,p=fp(tr,va); oof[va]=o; pred+=p/5
    log(f"[{nm}] OOF BA={balanced_accuracy_score(y,oof.argmax(1)):.5f}"); bases[nm]=(oof,pred)

def lgbm_fp(tr,va):
    for dev in ("gpu","cpu"):
        try:
            p=dict(objective="multiclass",num_class=3,learning_rate=0.05,num_leaves=160,min_child_samples=50,
                subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=2.0,n_estimators=2500,
                random_state=42,verbose=-1,n_jobs=-1,class_weight="balanced")
            if dev=="gpu": p.update(device="gpu",max_bin=255)
            m=lgb.LGBMClassifier(**p); m.fit(Xtr[tr],y[tr],eval_set=[(Xtr[va],y[va])],eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
            return m.predict_proba(Xtr[va]),m.predict_proba(Xte)
        except Exception as e: log(f"lgbm {dev} {e}")
    raise RuntimeError
def xgb_fp(tr,va):
    sw=cw_arr[y[tr]]
    for dev in ("cuda","cpu"):
        try:
            m=xgb.XGBClassifier(objective="multi:softprob",num_class=3,n_estimators=2500,learning_rate=0.05,max_depth=8,
                subsample=0.8,colsample_bytree=0.7,reg_lambda=2.0,tree_method="hist",device=dev,eval_metric="mlogloss",
                early_stopping_rounds=80,random_state=42,n_jobs=-1)
            m.fit(Xtr[tr],y[tr],sample_weight=sw,eval_set=[(Xtr[va],y[va])],verbose=0)
            return m.predict_proba(Xtr[va]),m.predict_proba(Xte)
        except Exception as e: log(f"xgb {dev} {e}")
    raise RuntimeError
def cat_fp(tr,va):
    for tt in ("GPU","CPU"):
        try:
            m=CatBoostClassifier(iterations=2500,learning_rate=0.05,depth=8,l2_leaf_reg=3.0,loss_function="MultiClass",
                auto_class_weights="Balanced",random_seed=42,task_type=tt,devices="0",verbose=0,
                early_stopping_rounds=80,allow_writing_files=False)
            m.fit(Pool(Xtr_df.iloc[tr],y[tr],cat_features=cat_idx),eval_set=Pool(Xtr_df.iloc[va],y[va],cat_features=cat_idx))
            return m.predict_proba(Xtr_df.iloc[va]),m.predict_proba(Xte_df)
        except Exception as e: log(f"cat {tt} {e}")
    raise RuntimeError
def hgb_fp(tr,va):
    m=HistGradientBoostingClassifier(max_iter=700,learning_rate=0.06,max_leaf_nodes=63,l2_regularization=1.0,
        class_weight="balanced",random_state=42,early_stopping=True,validation_fraction=0.1)
    m.fit(Xtr[tr],y[tr]); return m.predict_proba(Xtr[va]),m.predict_proba(Xte)

cv(lgbm_fp,"lgbm"); cv(xgb_fp,"xgb"); cv(cat_fp,"cat"); cv(hgb_fp,"hgb")

# ---- PyTorch NNs (GPU after cu118 fix; CPU fallback) ----
try:
    import torch, torch.nn as nn
    dev="cuda" if torch.cuda.is_available() else "cpu"
    try:
        _=torch.zeros(2).to(dev)+1
    except Exception:
        dev="cpu"
    log(f"torch {torch.__version__} device={dev}")
    sc_all=StandardScaler().fit(Xtr_nn); Xs=sc_all.transform(Xtr_nn).astype(np.float32); Xes=sc_all.transform(Xte_nn).astype(np.float32)
    class MLP(nn.Module):
        def __init__(s,d,hid,p):
            super().__init__(); L=[]; x=d
            for h in hid: L+=[nn.Linear(x,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(p)]; x=h
            L+=[nn.Linear(x,3)]; s.net=nn.Sequential(*L)
        def forward(s,x): return s.net(x)
    wt=torch.tensor(cw_arr,device=dev)
    Xes_t=torch.tensor(Xes,device=dev)
    def nn_factory(hid,drop,seed,epochs=45):
        def fp(tr,va):
            torch.manual_seed(seed)
            Xt=torch.tensor(Xs[tr],device=dev); yt=torch.tensor(y[tr],device=dev)
            Xv=torch.tensor(Xs[va],device=dev)
            m=MLP(Xs.shape[1],hid,drop).to(dev)
            opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
            lossf=nn.CrossEntropyLoss(weight=wt)
            n=len(tr); bs=8192; best=(-1,None); pat=0
            for ep in range(epochs):
                m.train(); perm=torch.randperm(n,device=dev)
                for i in range(0,n,bs):
                    idx=perm[i:i+bs]; opt.zero_grad(); loss=lossf(m(Xt[idx]),yt[idx]); loss.backward(); opt.step()
                sch.step(); m.eval()
                with torch.no_grad(): pv=torch.softmax(m(Xv),1).cpu().numpy()
                ba=balanced_accuracy_score(y[va],pv.argmax(1))
                if ba>best[0]: best=(ba,{k:v.clone() for k,v in m.state_dict().items()}); pat=0
                else:
                    pat+=1
                    if pat>=7: break
            m.load_state_dict(best[1]); m.eval()
            with torch.no_grad():
                ov=torch.softmax(m(Xv),1).cpu().numpy(); oe=torch.softmax(m(Xes_t),1).cpu().numpy()
            return ov,oe
        return fp
    cv(nn_factory([256,128],0.3,42),"nn1")
    cv(nn_factory([512,256,128],0.4,1),"nn2")
    cv(nn_factory([384,192],0.2,7),"nn3")
except Exception as e:
    log(f"NN block failed: {e}")

# external RealMLP
rof=find("oof_preds.csv"); rtf=find("test_preds.csv")
if rof and rtf:
    rm_oof=pd.read_csv(rof).set_index("id").loc[train["id"],classes].values
    rm_pred=pd.read_csv(rtf).set_index("id").loc[test["id"],classes].values
    bases["realmlp"]=(rm_oof,rm_pred); log(f"[realmlp] OOF BA={balanced_accuracy_score(y,rm_oof.argmax(1)):.5f}")

# LEVEL-1 meta
names=list(bases.keys())
OOF=np.hstack([bases[n][0] for n in names]); PRED=np.hstack([bases[n][1] for n in names])
meta_oof=np.zeros((N,3))
for tr,va in skf.split(OOF,y):
    lr=LogisticRegression(C=1.0,max_iter=3000,class_weight="balanced"); lr.fit(OOF[tr],y[tr]); meta_oof[va]=lr.predict_proba(OOF[va])
lr=LogisticRegression(C=1.0,max_iter=3000,class_weight="balanced"); lr.fit(OOF,y); meta_pred=lr.predict_proba(PRED)
log("bases: "+", ".join(f"{n}={balanced_accuracy_score(y,bases[n][0].argmax(1)):.4f}" for n in names))
log(f"META(LR) OOF={balanced_accuracy_score(y,meta_oof.argmax(1)):.5f}")
baw=lambda P,w: balanced_accuracy_score(y,(P*w).argmax(1))
best=(np.ones(3),baw(meta_oof,np.ones(3)))
for w in product(np.linspace(0.7,1.6,10),repeat=3):
    s=baw(meta_oof,np.array(w));
    if s>best[1]: best=(np.array(w),s)
cw,score=best
log(f"tuned class_w={cw.round(3)} META OOF={score:.5f}")
final=(meta_pred*cw).argmax(1)
pd.DataFrame({"id":test["id"],"class":[classes[i] for i in final]}).to_csv("/kaggle/working/submission.csv",index=False)
with open("/kaggle/working/scores.txt","w") as f:
    f.write("bases: "+", ".join(f"{n}={balanced_accuracy_score(y,bases[n][0].argmax(1)):.5f}" for n in names)+"\n")
    f.write(f"meta={balanced_accuracy_score(y,meta_oof.argmax(1)):.5f} tuned={score:.5f} class_w={cw.round(4).tolist()}\n")
log(f"DONE tuned META BA={score:.5f}")
