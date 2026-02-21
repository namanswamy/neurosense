"""
============================================================
NeuroSense — One-Click ML Setup & Training Script
Run this single file to download dataset + train both models
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
print("  NeuroSense ML — Setup & Training")
print("=" * 60)

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Train: {len(X_train)}  Test: {len(X_test)}")

# ── Step 3: Train Random Forest ───────────────────────────
print("\n" + "-" * 60)
print("  STEP 3: Training Model 1 — Random Forest Classifier")
print("-" * 60)

rf = RandomForestClassifier(
    n_estimators=50, max_depth=8, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, zero_division=0)
rf_rec = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print(f"[OK] Training complete")
print(f"     Accuracy:  {rf_acc:.4f}")
print(f"     Precision: {rf_prec:.4f}")
print(f"     Recall:    {rf_rec:.4f}")
print(f"     F1 Score:  {rf_f1:.4f}")
print(f"     Cross-val: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

# Top features
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print(f"\n     Top Features:")
for i in range(min(5, len(feature_cols))):
    idx = sorted_idx[i]
    bar = "#" * int(importances[idx] * 40)
    print(f"       {feature_cols[idx]:>15}: {importances[idx]:.4f} {bar}")

# Export RF
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

rf_json = {
    'model_type': 'RandomForestClassifier',
    'n_estimators': rf.n_estimators,
    'max_depth': rf.max_depth,
    'feature_names': feature_cols,
    'feature_importances': [round(float(x), 6) for x in importances],
    'trees': [export_tree(est) for est in rf.estimators_],
    'metrics': {
        'accuracy': round(rf_acc, 4),
        'precision': round(rf_prec, 4),
        'recall': round(rf_rec, 4),
        'f1': round(rf_f1, 4),
        'cv_mean': round(float(rf_cv.mean()), 4),
        'cv_std': round(float(rf_cv.std()), 4)
    },
    'training_info': {
        'dataset': 'Autism Screening on Adults (Kaggle)',
        'dataset_url': 'https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults',
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_cols)
    }
}

rf_path = os.path.join(ML_DIR, 'rf_model.json')
with open(rf_path, 'w') as f:
    json.dump(rf_json, f)
print(f"\n[OK] Saved: rf_model.json ({os.path.getsize(rf_path)/1024:.1f} KB)")

# ── Step 4: Train Neural Network ──────────────────────────
print("\n" + "-" * 60)
print("  STEP 4: Training Model 2 — Neural Network MLP")
print("-" * 60)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(20, 12, 8), activation='relu', solver='adam',
    learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=2000, random_state=42, early_stopping=True,
    validation_fraction=0.15, n_iter_no_change=20, verbose=False
)
mlp.fit(X_train_s, y_train)

nn_pred = mlp.predict(X_test_s)
nn_acc = accuracy_score(y_test, nn_pred)
nn_prec = precision_score(y_test, nn_pred, zero_division=0)
nn_rec = recall_score(y_test, nn_pred, zero_division=0)
nn_f1 = f1_score(y_test, nn_pred, zero_division=0)

nn_cv = cross_val_score(
    MLPClassifier(hidden_layer_sizes=(20, 12, 8), activation='relu', solver='adam',
                  max_iter=2000, random_state=42, early_stopping=True),
    scaler.transform(X), y, cv=5, scoring='accuracy'
)

print(f"[OK] Training complete (converged in {mlp.n_iter_} epochs)")
print(f"     Accuracy:  {nn_acc:.4f}")
print(f"     Precision: {nn_prec:.4f}")
print(f"     Recall:    {nn_rec:.4f}")
print(f"     F1 Score:  {nn_f1:.4f}")
print(f"     Cross-val: {nn_cv.mean():.4f} (+/- {nn_cv.std():.4f})")

arch = [X.shape[1]] + list(mlp.hidden_layer_sizes) + [1]
print(f"\n     Architecture: {' -> '.join(map(str, arch))}")
for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
    act = 'ReLU' if i < len(mlp.coefs_) - 1 else 'Sigmoid'
    print(f"       Layer {i+1}: ({w.shape[0]}, {w.shape[1]}) + ({b.shape[0]},)  [{act}]")

# Export NN
layers = []
for i in range(len(mlp.coefs_)):
    W = mlp.coefs_[i].T  # transpose to (output, input) for Node.js
    b = mlp.intercepts_[i]
    layers.append({
        'W': [[round(float(w), 8) for w in row] for row in W],
        'b': [round(float(x), 8) for x in b]
    })

nn_json = {
    'model_type': 'MLPClassifier',
    'architecture': [int(x) for x in arch],
    'activation': 'relu',
    'output_activation': mlp.out_activation_,
    'scaler': {
        'mean': [round(float(x), 8) for x in scaler.mean_],
        'scale': [round(float(x), 8) for x in scaler.scale_]
    },
    'layers': layers,
    'metrics': {
        'accuracy': round(nn_acc, 4),
        'precision': round(nn_prec, 4),
        'recall': round(nn_rec, 4),
        'f1': round(nn_f1, 4),
        'cv_mean': round(float(nn_cv.mean()), 4),
        'cv_std': round(float(nn_cv.std()), 4)
    },
    'training_info': {
        'dataset': 'Autism Screening on Adults (Kaggle)',
        'dataset_url': 'https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults',
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'epochs': int(mlp.n_iter_),
        'best_val_score': round(float(mlp.best_validation_score_), 4)
    }
}

nn_path = os.path.join(ML_DIR, 'nn_model.json')
with open(nn_path, 'w') as f:
    json.dump(nn_json, f)
print(f"\n[OK] Saved: nn_model.json ({os.path.getsize(nn_path)/1024:.1f} KB)")

# ── Done ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print(f"""
  Files created:
    {rf_path}
    {nn_path}

  Model 1 — Random Forest:  {rf_acc*100:.1f}% accuracy, {rf.n_estimators} trees
  Model 2 — Neural Network: {nn_acc*100:.1f}% accuracy, {' -> '.join(map(str, arch))}

  Next step:
    cd ..
    node server.js

  The server will auto-detect and load both models on startup.
""")
