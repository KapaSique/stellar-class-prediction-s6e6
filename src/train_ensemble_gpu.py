"""S6E6 Stellar Class — GPU ensemble (LGBM+XGB+CatBoost) + class-weight tuning.
Metric: Balanced Accuracy. Runs on Kaggle GPU."""
import os, glob, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from itertools import product
t0=time.time()

c=glob.glob("/kaggle/input/**/train.csv",recursive=True)
DATA=next((os.path.dirname(p) for p in c if os.path.exists(os.path.join(os.path.dirname(p),"test.csv"))),"/kaggle/input/playground-series-s6e6")
print("DATA",DATA,flush=True)
train=pd.read_csv(f"{DATA}/train.csv"); test=pd.read_csv(f"{DATA}/test.csv")
classes=["GALAXY","QSO","STAR"]; cls2i={c:i for i,c in enumerate(classes)}
y=train["class"].map(cls2i).values; priors=np.bincount(y)/len(y); N=len(train)
print("priors",dict(zip(classes,priors.round(4))),flush=True)

def fe(df):
    df=df.copy(); bands=["u","g","r","i","z"]
    for a,b in [("u","g"),("g","r"),("r","i"),("i","z"),("u","r"),("g","i"),("u","z"),("u","i"),("g","z")]:
        df[f"{a}_{b}"]=df[a]-df[b]
    df["redshift_log"]=np.log1p(df["redshift"].clip(lower=0))
    df["redshift_pos"]=(df["redshift"]>0.0035).astype(int)
    df["mean_mag"]=df[bands].mean(axis=1); df["std_mag"]=df[bands].std(axis=1)
    df["min_mag"]=df[bands].min(axis=1); df["max_mag"]=df[bands].max(axis=1)
    return df
train=fe(train); test=fe(test)
catc=["spectral_type","galaxy_population"]
for cc in catc:
    cats=pd.concat([train[cc],test[cc]]).astype("category").cat.categories
    train[cc]=pd.Categorical(train[cc],categories=cats).codes
    test[cc]=pd.Categorical(test[cc],categories=cats).codes
feats=[c for c in train.columns if c not in ["id","class"]]
cat_idx=[feats.index(c) for c in catc]
Xtr=train[feats]; Xte=test[feats]
skf=StratifiedKFold(5, shuffle=True, random_state=42)
print(f"features={len(feats)}",flush=True)

def run_lgbm():
    oof=np.zeros((N,3)); pred=np.zeros((len(test),3))
    for dev in ("gpu","cpu"):
        try:
            for f,(tr,va) in enumerate(skf.split(Xtr,y)):
                p=dict(objective="multiclass",num_class=3,learning_rate=0.05,num_leaves=160,
                    min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=2.0,
                    n_estimators=4000,random_state=42,verbose=-1,n_jobs=-1,class_weight="balanced")
                if dev=="gpu": p.update(device="gpu",max_bin=255)
                m=lgb.LGBMClassifier(**p)
                m.fit(Xtr.iloc[tr],y[tr],eval_set=[(Xtr.iloc[va],y[va])],eval_metric="multi_logloss",
                      categorical_feature=cat_idx,callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
                oof[va]=m.predict_proba(Xtr.iloc[va]); pred+=m.predict_proba(Xte)/skf.n_splits
            print(f"  lgbm({dev}) BA={balanced_accuracy_score(y,oof.argmax(1)):.5f} ({time.time()-t0:.0f}s)",flush=True)
            return oof,pred
        except Exception as e: print("lgbm",dev,"fail",e,flush=True)
    raise RuntimeError("lgbm")

def run_xgb():
    oof=np.zeros((N,3)); pred=np.zeros((len(test),3))
    sw=np.array([1/priors[c] for c in y]); sw=sw/sw.mean()
    for dev in ("cuda","cpu"):
        try:
            for f,(tr,va) in enumerate(skf.split(Xtr,y)):
                m=xgb.XGBClassifier(objective="multi:softprob",num_class=3,n_estimators=4000,learning_rate=0.05,
                    max_depth=8,subsample=0.8,colsample_bytree=0.7,reg_lambda=2.0,tree_method="hist",device=dev,
                    eval_metric="mlogloss",early_stopping_rounds=100,random_state=42,n_jobs=-1)
                m.fit(Xtr.iloc[tr],y[tr],sample_weight=sw[tr],eval_set=[(Xtr.iloc[va],y[va])],verbose=0)
                oof[va]=m.predict_proba(Xtr.iloc[va]); pred+=m.predict_proba(Xte)/skf.n_splits
            print(f"  xgb({dev}) BA={balanced_accuracy_score(y,oof.argmax(1)):.5f} ({time.time()-t0:.0f}s)",flush=True)
            return oof,pred
        except Exception as e: print("xgb",dev,"fail",e,flush=True)
    raise RuntimeError("xgb")

def run_cat():
    oof=np.zeros((N,3)); pred=np.zeros((len(test),3))
    for tt in ("GPU","CPU"):
        try:
            for f,(tr,va) in enumerate(skf.split(Xtr,y)):
                m=CatBoostClassifier(iterations=4000,learning_rate=0.05,depth=8,l2_leaf_reg=3.0,
                    loss_function="MultiClass",auto_class_weights="Balanced",random_seed=42,
                    task_type=tt,devices="0",verbose=0,early_stopping_rounds=100,allow_writing_files=False)
                m.fit(Pool(Xtr.iloc[tr],y[tr],cat_features=cat_idx),eval_set=Pool(Xtr.iloc[va],y[va],cat_features=cat_idx))
                oof[va]=m.predict_proba(Xtr.iloc[va]); pred+=m.predict_proba(Xte)/skf.n_splits
            print(f"  cat({tt}) BA={balanced_accuracy_score(y,oof.argmax(1)):.5f} ({time.time()-t0:.0f}s)",flush=True)
            return oof,pred
        except Exception as e: print("cat",tt,"fail",e,flush=True)
    raise RuntimeError("cat")

ol,pl=run_lgbm(); ox,px=run_xgb(); oc,pc=run_cat()
oof=(ol+ox+oc)/3; pred=(pl+px+pc)/3

def ba_w(P,w): return balanced_accuracy_score(y,(P*w).argmax(1))
best=(None,-1)
for w0,w1,w2 in product(np.linspace(0.5,2.0,16),repeat=3):
    w=np.array([w0,w1,w2]); s=ba_w(oof,w)
    if s>best[1]: best=(w,s)
w=best[0]
print(f"\nBA: plain={balanced_accuracy_score(y,oof.argmax(1)):.5f} prior={balanced_accuracy_score(y,(oof/priors).argmax(1)):.5f} tuned={best[1]:.5f} w={w.round(3)}",flush=True)

final=(pred*w).argmax(1)
# сохраняем усреднённые вероятности ансамбля (компактно, для последующего бленда с NN)
np.save("/kaggle/working/oof_ens.npy", oof.astype("float32"))
np.save("/kaggle/working/pred_ens.npy", pred.astype("float32"))
pd.DataFrame({"id":test["id"],"class":[classes[i] for i in final]}).to_csv("/kaggle/working/submission.csv",index=False)
with open("/kaggle/working/scores.txt","w") as f:
    f.write(f"lgbm={balanced_accuracy_score(y,ol.argmax(1)):.5f} xgb={balanced_accuracy_score(y,ox.argmax(1)):.5f} cat={balanced_accuracy_score(y,oc.argmax(1)):.5f}\n")
    f.write(f"blend_tuned={best[1]:.5f} w={w.round(4).tolist()}\n")
print(f"DONE {time.time()-t0:.0f}s tuned BA={best[1]:.5f}",flush=True)
