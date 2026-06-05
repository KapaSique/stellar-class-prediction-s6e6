import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import warnings; warnings.filterwarnings("ignore")

train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
classes = ["GALAXY","QSO","STAR"]
cls2i = {c:i for i,c in enumerate(classes)}
y = train["class"].map(cls2i).values
priors = np.bincount(y)/len(y)
print("priors:", dict(zip(classes, priors.round(4))))

def fe(df):
    df = df.copy()
    bands = ["u","g","r","i","z"]
    # астрономические цветовые индексы
    for a,b in [("u","g"),("g","r"),("r","i"),("i","z"),("u","r"),("g","i"),("u","z")]:
        df[f"{a}_{b}"] = df[a]-df[b]
    df["redshift_log"] = np.log1p(df["redshift"].clip(lower=0))
    df["redshift_pos"] = (df["redshift"]>0.0035).astype(int)  # звёзды ~0
    df["mean_mag"] = df[bands].mean(axis=1)
    df["std_mag"] = df[bands].std(axis=1)
    df["r_minus_z"] = df["r"]-df["z"]
    return df

train = fe(train); test = fe(test)
for c in ["spectral_type","galaxy_population"]:
    cats = pd.concat([train[c],test[c]]).astype("category").cat.categories
    train[c]=pd.Categorical(train[c],categories=cats).codes
    test[c]=pd.Categorical(test[c],categories=cats).codes

feats = [c for c in train.columns if c not in ["id","class"]]
cat_idx = [feats.index(c) for c in ["spectral_type","galaxy_population"]]
print(f"features ({len(feats)})")

params = dict(objective="multiclass", num_class=3, metric="multi_logloss",
              learning_rate=0.05, num_leaves=128, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
              reg_lambda=2.0, n_estimators=2000, random_state=42, verbose=-1, n_jobs=-1,
              class_weight="balanced")

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros((len(train),3)); pred = np.zeros((len(test),3))
for f,(tr,va) in enumerate(skf.split(train[feats],y)):
    m = lgb.LGBMClassifier(**params)
    m.fit(train[feats].iloc[tr], y[tr], eval_set=[(train[feats].iloc[va],y[va])],
          eval_metric="multi_logloss", categorical_feature=cat_idx,
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof[va] = m.predict_proba(train[feats].iloc[va])
    pred += m.predict_proba(test[feats])/skf.n_splits
    ba = balanced_accuracy_score(y[va], oof[va].argmax(1))
    print(f"fold {f+1} balanced_acc={ba:.5f} it={m.best_iteration_}")

# argmax vs prior-adjusted (для balanced accuracy при дисбалансе)
ba_plain = balanced_accuracy_score(y, oof.argmax(1))
ba_prior = balanced_accuracy_score(y, (oof/priors).argmax(1))
print(f"\n>>> OOF balanced_acc plain  = {ba_plain:.5f}")
print(f">>> OOF balanced_acc prior  = {ba_prior:.5f}")

use_prior = ba_prior > ba_plain
final = (pred/priors).argmax(1) if use_prior else pred.argmax(1)
print(f"using {'prior-adjusted' if use_prior else 'plain'} argmax")
np.save("oof_lgbm.npy", oof); np.save("pred_lgbm.npy", pred)
sub = pd.DataFrame({"id":test["id"],"class":[classes[i] for i in final]})
sub.to_csv("submission_lgbm.csv", index=False)
print("saved submission_lgbm.csv  dist:", sub["class"].value_counts().to_dict())
