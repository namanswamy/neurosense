"""
============================================================
NeuroSense — One-Click ML Setup & Training Script
Train 4 models: Random Forest, XGBoost, SVM, LightGBM
============================================================
Usage:
    cd ml
    pip install -r requirements.txt
    python setup_and_train.py
============================================================
"""

import os
import sys
import json
import urllib.request
import zipfile
import glob

# ── Step 0: Check dependencies ────────────────────────────
print("=" * 60)
print("  NeuroSense ML — Setup & Training (4-Model Ensemble)")
print("=" * 60)

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import lightgbm as lgb
    print("\n[OK] All dependencies found")
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

ML_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ML_DIR, 'autism_screening.csv')

# ── Step 1: Get the dataset ───────────────────────────────
print("\n" + "-" * 60)
print("  STEP 1: Dataset")
print("-" * 60)

if os.path.exists(DATA_FILE):
    print(f"[OK] Dataset already exists: {DATA_FILE}")
else:
    # Try to download from the public UCI mirror (same data as Kaggle)
    UCI_URL = "https://archive.ics.uci.edu/static/public/426/autism+screening+adult.zip"
    zip_path = os.path.join(ML_DIR, '_dataset.zip')

    print(f"[..] Dataset not found. Attempting download...")
    print(f"     Source: UCI ML Repository (same data as Kaggle)")
    print(f"     URL: {UCI_URL}")

    try:
        req = urllib.request.Request(UCI_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp, open(zip_path, 'wb') as f:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r     Downloading: {pct:.0f}%", end="", flush=True)
            print()
        print("[OK] Download complete")

        # Extract CSV from zip
        with zipfile.ZipFile(zip_path, 'r') as zf:
            csv_files = [n for n in zf.namelist() if n.endswith('.csv') or n.endswith('.arff')]
            print(f"     Archive contents: {csv_files}")

            extracted = False
            for name in zf.namelist():
                if name.endswith('.csv'):
                    with zf.open(name) as src, open(DATA_FILE, 'wb') as dst:
                        dst.write(src.read())
                    extracted = True
                    print(f"[OK] Extracted: {name} -> autism_screening.csv")
                    break

            if not extracted:
                # Try .arff file and convert
                for name in zf.namelist():
                    if name.endswith('.arff'):
                        print(f"     Found ARFF format: {name}")
                        with zf.open(name) as src:
                            content = src.read().decode('utf-8', errors='replace')

                        # Parse ARFF to CSV
                        lines = content.split('\n')
                        data_started = False
                        headers = []
                        rows = []
                        for line in lines:
                            line = line.strip()
                            if line.upper() == '@DATA':
                                data_started = True
                                continue
                            if not data_started:
                                if line.upper().startswith('@ATTRIBUTE'):
                                    parts = line.split()
                                    if len(parts) >= 2:
                                        headers.append(parts[1])
                            else:
                                if line and not line.startswith('%'):
                                    rows.append(line)

                        if headers and rows:
                            with open(DATA_FILE, 'w') as f:
                                f.write(','.join(headers) + '\n')
                                for row in rows:
                                    f.write(row + '\n')
                            extracted = True
                            print(f"[OK] Converted ARFF to CSV ({len(rows)} rows)")
                        break

            if not extracted:
                print("[ERROR] Could not find CSV in archive")
                raise Exception("No CSV found")

        # Clean up zip
        os.remove(zip_path)

    except Exception as e:
        print(f"\n[WARNING] Auto-download failed: {e}")
        print()
        print("  Please download manually:")
        print("  1. Go to: https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults")
        print("  2. Click 'Download' and extract the CSV")
        print(f"  3. Rename it to 'autism_screening.csv'")
        print(f"  4. Place it in: {ML_DIR}")
        print(f"  5. Re-run: python setup_and_train.py")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        sys.exit(1)

# ── Step 2: Load & preprocess ─────────────────────────────
print("\n" + "-" * 60)
print("  STEP 2: Loading & Preprocessing")
print("-" * 60)

df = pd.read_csv(DATA_FILE)
print(f"[OK] Loaded {len(df)} records, {len(df.columns)} columns")

# Normalize column names for different dataset versions
col_map = {}
for col in df.columns:
    cl = col.lower().strip()
    if cl in ('jundice', 'jaundice'):
        col_map[col] = 'jaundice'
    elif cl in ('austim', 'autism'):
        col_map[col] = 'family_autism'
    elif cl in ('class/asd', 'class', 'asd', 'class_asd'):
        col_map[col] = 'target'
    elif cl in ('contry_of_res', 'country_of_res'):
        col_map[col] = 'country'

df = df.rename(columns=col_map)

# Check required columns exist
aq_cols = sorted([c for c in df.columns if c.upper().startswith('A') and '_Score' in c],
                 key=lambda x: int(''.join(filter(str.isdigit, x.split('_')[0]))))

if len(aq_cols) < 10:
    # Try alternate naming: A1, A2, ... (without _Score)
    aq_cols = sorted([c for c in df.columns if c in [f'A{i}' for i in range(1, 11)]],
                     key=lambda x: int(x[1:]))

if len(aq_cols) < 10:
    print(f"[ERROR] Expected 10 AQ score columns, found {len(aq_cols)}: {aq_cols}")
    print(f"        Available columns: {list(df.columns)}")
    sys.exit(1)

print(f"     AQ-10 columns: {aq_cols}")

# Encode categoricals (handle both object and StringDtype)
for col, vals in [('gender', {'m': 1, 'male': 1, 'f': 0, 'female': 0}),
                  ('jaundice', {'yes': 1, 'no': 0}),
                  ('family_autism', {'yes': 1, 'no': 0}),
                  ('target', {'yes': 1, 'no': 0})]:
    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].apply(lambda x: vals.get(str(x).strip().lower(), 0))

df['age'] = pd.to_numeric(df.get('age', pd.Series([25]*len(df))), errors='coerce').fillna(25)

feature_cols = aq_cols + ['age', 'gender', 'jaundice', 'family_autism']

# Fill any remaining NaN
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X = df[feature_cols].values.astype(float)
y = df['target'].values.astype(int)

print(f"[OK] Features: {len(feature_cols)} -> {feature_cols}")
print(f"     Class distribution: NO={np.sum(y==0)}  YES={np.sum(y==1)}")

# Calculate class weight ratio for imbalanced handling
pos_weight = np.sum(y == 0) / max(np.sum(y == 1), 1)
print(f"     Class weight ratio (neg/pos): {pos_weight:.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Train: {len(X_train)}  Test: {len(X_test)}")

# Fit scaler (needed for SVM)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


def print_metrics(name, y_true, y_pred, cv_scores):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"[OK] {name} — Training complete")
    print(f"     Accuracy:  {acc:.4f}")
    print(f"     Precision: {prec:.4f}")
    print(f"     Recall:    {rec:.4f}")
    print(f"     F1 Score:  {f1:.4f}")
    print(f"     Cross-val: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    return {'accuracy': round(acc, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1': round(f1, 4),
            'cv_mean': round(float(cv_scores.mean()), 4),
            'cv_std': round(float(cv_scores.std()), 4)}


training_info = {
    'dataset': 'Autism Screening on Adults (Kaggle)',
    'dataset_url': 'https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults',
    'total_samples': len(X),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'n_features': len(feature_cols)
}


# ── Helper: Export sklearn tree ───────────────────────────
def export_tree(estimator):
    tree = estimator.tree_
    nodes = []
    for i in range(tree.node_count):
        nodes.append({
            'f': int(tree.feature[i]),
            't': round(float(tree.threshold[i]), 6),
            'l': int(tree.children_left[i]),
            'r': int(tree.children_right[i]),
            'v': [max(0, round(float(tree.value[i][0][0]))), max(0, round(float(tree.value[i][0][1])))]
        })
    return nodes


# ══════════════════════════════════════════════════════════
#  MODEL 1: Random Forest Classifier
# ══════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("  STEP 3: Training Model 1 — Random Forest Classifier")
print("-" * 60)

rf = RandomForestClassifier(
    n_estimators=50, max_depth=8, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
rf_metrics = print_metrics("Random Forest", y_test, rf_pred, rf_cv)

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print(f"\n     Top Features:")
for i in range(min(5, len(feature_cols))):
    idx = sorted_idx[i]
    bar = "#" * int(importances[idx] * 40)
    print(f"       {feature_cols[idx]:>15}: {importances[idx]:.4f} {bar}")

rf_json = {
    'model_type': 'RandomForestClassifier',
    'n_estimators': rf.n_estimators,
    'max_depth': rf.max_depth,
    'feature_names': feature_cols,
    'feature_importances': [round(float(x), 6) for x in importances],
    'trees': [export_tree(est) for est in rf.estimators_],
    'metrics': rf_metrics,
    'training_info': training_info
}

rf_path = os.path.join(ML_DIR, 'rf_model.json')
with open(rf_path, 'w') as f:
    json.dump(rf_json, f)
print(f"\n[OK] Saved: rf_model.json ({os.path.getsize(rf_path)/1024:.1f} KB)")


# ══════════════════════════════════════════════════════════
#  MODEL 2: XGBoost Classifier
# ══════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("  STEP 4: Training Model 2 — XGBoost Classifier")
print("-" * 60)

xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    scale_pos_weight=pos_weight, eval_metric='logloss',
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_cv = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')
xgb_metrics = print_metrics("XGBoost", y_test, xgb_pred, xgb_cv)

xgb_importances = xgb_model.feature_importances_
print(f"\n     Top Features:")
xgb_sorted = np.argsort(xgb_importances)[::-1]
for i in range(min(5, len(feature_cols))):
    idx = xgb_sorted[i]
    bar = "#" * int(xgb_importances[idx] * 40)
    print(f"       {feature_cols[idx]:>15}: {xgb_importances[idx]:.4f} {bar}")

# Export XGBoost as JSON trees
booster = xgb_model.get_booster()
xgb_trees_raw = booster.get_dump(dump_format='json')
xgb_trees = [json.loads(t) for t in xgb_trees_raw]

xgb_json = {
    'model_type': 'XGBClassifier',
    'n_estimators': int(xgb_model.n_estimators),
    'max_depth': int(xgb_model.max_depth),
    'learning_rate': float(xgb_model.learning_rate),
    'base_score': 0.5,
    'feature_names': feature_cols,
    'feature_importances': [round(float(x), 6) for x in xgb_importances],
    'trees': xgb_trees,
    'metrics': xgb_metrics,
    'training_info': training_info
}

xgb_path = os.path.join(ML_DIR, 'xgb_model.json')
with open(xgb_path, 'w') as f:
    json.dump(xgb_json, f)
print(f"\n[OK] Saved: xgb_model.json ({os.path.getsize(xgb_path)/1024:.1f} KB)")


# ══════════════════════════════════════════════════════════
#  MODEL 3: Support Vector Machine (SVM)
# ══════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("  STEP 5: Training Model 3 — SVM (RBF Kernel)")
print("-" * 60)

svm_model = SVC(
    kernel='rbf', probability=True, class_weight='balanced',
    C=1.0, gamma='scale', random_state=42
)
svm_model.fit(X_train_s, y_train)
svm_pred = svm_model.predict(X_test_s)
svm_cv = cross_val_score(svm_model, scaler.transform(X), y, cv=5, scoring='accuracy')
svm_metrics = print_metrics("SVM (RBF)", y_test, svm_pred, svm_cv)

# Export SVM: support vectors, dual coefficients, intercept, gamma, scaler
gamma_val = 1.0 / (X_train_s.shape[1] * X_train_s.var()) if svm_model.gamma == 'scale' else svm_model.gamma
# sklearn computes gamma='scale' as 1 / (n_features * X.var())
# But the actual fitted gamma can be retrieved:
gamma_val = float(svm_model._gamma)

svm_json = {
    'model_type': 'SVC_RBF',
    'kernel': 'rbf',
    'C': float(svm_model.C),
    'gamma': gamma_val,
    'n_support': [int(x) for x in svm_model.n_support_],
    'support_vectors': [[round(float(v), 8) for v in sv] for sv in svm_model.support_vectors_],
    'dual_coef': [[round(float(v), 8) for v in row] for row in svm_model.dual_coef_],
    'intercept': [round(float(v), 8) for v in svm_model.intercept_],
    'classes': [int(c) for c in svm_model.classes_],
    'scaler': {
        'mean': [round(float(x), 8) for x in scaler.mean_],
        'scale': [round(float(x), 8) for x in scaler.scale_]
    },
    'feature_names': feature_cols,
    'metrics': svm_metrics,
    'training_info': training_info
}

svm_path = os.path.join(ML_DIR, 'svm_model.json')
with open(svm_path, 'w') as f:
    json.dump(svm_json, f)
print(f"\n[OK] Saved: svm_model.json ({os.path.getsize(svm_path)/1024:.1f} KB)")


# ══════════════════════════════════════════════════════════
#  MODEL 4: LightGBM Classifier
# ══════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("  STEP 6: Training Model 4 — LightGBM Classifier")
print("-" * 60)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100, num_leaves=31, learning_rate=0.1,
    is_unbalance=True, random_state=42, n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_cv = cross_val_score(lgb_model, X, y, cv=5, scoring='accuracy')
lgb_metrics = print_metrics("LightGBM", y_test, lgb_pred, lgb_cv)

lgb_importances = lgb_model.feature_importances_.astype(float)
lgb_importances_norm = lgb_importances / max(lgb_importances.sum(), 1e-10)
print(f"\n     Top Features:")
lgb_sorted = np.argsort(lgb_importances)[::-1]
for i in range(min(5, len(feature_cols))):
    idx = lgb_sorted[i]
    bar = "#" * int(lgb_importances_norm[idx] * 40)
    print(f"       {feature_cols[idx]:>15}: {lgb_importances_norm[idx]:.4f} {bar}")

# Export LightGBM as JSON tree dump
lgb_dump = lgb_model.booster_.dump_model()
lgb_trees_data = lgb_dump['tree_info']

lgb_json = {
    'model_type': 'LGBMClassifier',
    'n_estimators': int(lgb_model.n_estimators),
    'num_leaves': int(lgb_model.num_leaves),
    'learning_rate': float(lgb_model.learning_rate),
    'feature_names': feature_cols,
    'feature_importances': [round(float(x), 6) for x in lgb_importances_norm],
    'trees': lgb_trees_data,
    'metrics': lgb_metrics,
    'training_info': training_info
}

lgb_path = os.path.join(ML_DIR, 'lgb_model.json')
with open(lgb_path, 'w') as f:
    json.dump(lgb_json, f)
print(f"\n[OK] Saved: lgb_model.json ({os.path.getsize(lgb_path)/1024:.1f} KB)")


# ── Done ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print(f"""
  Files created:
    {rf_path}
    {xgb_path}
    {svm_path}
    {lgb_path}

  Model 1 — Random Forest:  {rf_metrics['accuracy']*100:.1f}% accuracy, {rf.n_estimators} trees
  Model 2 — XGBoost:        {xgb_metrics['accuracy']*100:.1f}% accuracy, {xgb_model.n_estimators} trees
  Model 3 — SVM (RBF):      {svm_metrics['accuracy']*100:.1f}% accuracy, {len(svm_model.support_vectors_)} support vectors
  Model 4 — LightGBM:       {lgb_metrics['accuracy']*100:.1f}% accuracy, {lgb_model.n_estimators} trees

  Next step:
    cd ..
    node server.js

  The server will auto-detect and load all 4 models on startup.
""")
