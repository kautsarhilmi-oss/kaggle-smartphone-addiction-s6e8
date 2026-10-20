"""
Smartphone Addiction Prediction — Kaggle Playground Series S6E8
Metric  : ROC-AUC
Model   : HistGradientBoostingClassifier (sklearn) — 3-Fold CV on 150K subsample
CV AUC  : ~0.9561 | Public LB: 0.95873 | Private: 0.95921
Author  : Kautsar Hilmi
GitHub  : https://github.com/kautsarhilmi-oss
LinkedIn: https://www.linkedin.com/in/kautsarhilmi/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
sub   = pd.read_csv('sample_submission.csv')

TARGET = 'addicted_label'

# ─── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
def feature_engineering(df):
    df = df.copy()

    # Leisure vs productive screen time — key addiction signal
    df['leisure_screen_time']    = df['social_media_hours'].fillna(0) + df['gaming_hours'].fillna(0)
    df['productive_screen_time'] = df['work_study_hours'].fillna(0)
    df['leisure_to_productive']  = df['leisure_screen_time'] / (df['productive_screen_time'] + 1)

    # Weekend binge ratio — weekend spike = strong addiction signal
    df['weekend_to_daily_ratio'] = df['weekend_screen_time'] / (df['daily_screen_time_hours'] + 1)

    # Sleep deprivation — deviasi negatif dari threshold 7 jam
    df['sleep_deficit']          = (7 - df['sleep_hours']).clip(lower=0)

    # Behavioral compulsion metrics
    df['notif_per_app_open']     = df['notifications_per_day'] / (df['app_opens_per_day'] + 1)
    df['app_opens_log']          = np.log1p(df['app_opens_per_day'])

    # Screen time per sleep hour — lebih tinggi = lebih buruk
    df['screen_per_sleep']       = df['daily_screen_time_hours'] / (df['sleep_hours'] + 1)

    # Social media dominance ratio
    df['social_media_ratio']     = df['social_media_hours'] / (df['daily_screen_time_hours'] + 1)

    return df

train = feature_engineering(train)
test  = feature_engineering(test)

# ─── 3. ENCODE CATEGORICALS ───────────────────────────────────────────────────
# HistGBM handles NaN natively — hanya encode tipe, tidak perlu imputation
cat_cols = ['gender', 'stress_level', 'academic_work_impact']
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
enc.fit(pd.concat([train[cat_cols], test[cat_cols]], axis=0))
train[cat_cols] = enc.transform(train[cat_cols])
test[cat_cols]  = enc.transform(test[cat_cols])

# ─── 4. DEFINE FEATURES ───────────────────────────────────────────────────────
FEATURES = [
    'age', 'daily_screen_time_hours', 'social_media_hours',
    'gaming_hours', 'work_study_hours', 'sleep_hours',
    'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time',
    'gender', 'stress_level', 'academic_work_impact',
    # engineered features
    'leisure_screen_time', 'productive_screen_time', 'leisure_to_productive',
    'weekend_to_daily_ratio', 'sleep_deficit', 'notif_per_app_open',
    'app_opens_log', 'screen_per_sleep', 'social_media_ratio',
]

X      = train[FEATURES]
y      = train[TARGET]
X_test = test[FEATURES]

# ─── 5. SUBSAMPLE FOR FAST CV ─────────────────────────────────────────────────
# 150K dari 691K — representatif, jauh lebih cepat di local/Colab
np.random.seed(42)
sample_idx = np.random.choice(len(train), 150000, replace=False)
X_s, y_s   = X.iloc[sample_idx], y.iloc[sample_idx]

# ─── 6. MODEL + 3-FOLD CV ─────────────────────────────────────────────────────
params = dict(
    max_iter=500,
    learning_rate=0.08,
    max_leaf_nodes=63,
    min_samples_leaf=80,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=True,
    validation_fraction=0.1,   # 10% dari data train untuk early stopping
    n_iter_no_change=20,
    random_state=42,
    verbose=0,
    class_weight='balanced',   # handle class imbalance 71% positive
)

N_SPLITS  = 3
skf       = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_s))
models    = []

print(f"\n{'='*58}")
print(f"  3-Fold CV — HistGradientBoostingClassifier")
print(f"  Features : {len(FEATURES)}")
print(f"  Train    : {len(train):,} rows (subsample: 150K)")
print(f"  Test     : {len(test):,} rows")
print(f"  Imbalance: {y.mean():.3f} positive rate")
print(f"{'='*58}")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_s, y_s), 1):
    m = HistGradientBoostingClassifier(**params)
    m.fit(X_s.iloc[tr_idx], y_s.iloc[tr_idx])

    oof_preds[val_idx] = m.predict_proba(X_s.iloc[val_idx])[:, 1]
    models.append(m)

    fold_auc = roc_auc_score(y_s.iloc[val_idx], oof_preds[val_idx])
    print(f"  Fold {fold} | AUC: {fold_auc:.5f} | Iters: {m.n_iter_}")

oof_auc = roc_auc_score(y_s, oof_preds)
print(f"{'='*58}")
print(f"  Overall CV AUC : {oof_auc:.5f}")
print(f"{'='*58}")

# ─── 7. PREDICT & SAVE SUBMISSION ─────────────────────────────────────────────
# Ensemble dari 3 fold models → lebih stabil dari single model
test_preds = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)

sub['addicted_label'] = test_preds
sub.to_csv('submission.csv', index=False)
print(f"\nDone! submission.csv saved.")
print(f"Pred mean : {test_preds.mean():.4f}")
print(f"Train mean: {y.mean():.4f}")
