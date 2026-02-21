"""
============================================================
NeuroSense — Model 1: Random Forest Classifier
Trains on Kaggle 'Autism Screening on Adults' dataset (AQ-10)
Exports decision trees as JSON for Node.js inference
============================================================
Dataset: https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults
Place the CSV file in this directory as 'autism_screening.csv'
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import json
import os
import sys

# ── Load Dataset ──────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), 'autism_screening.csv')

if not os.path.exists(DATA_FILE):
    print("ERROR: Dataset not found!")
    print(f"Expected: {DATA_FILE}")
    print()
    print("Download from: https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults")
    print("Rename the CSV to 'autism_screening.csv' and place it in the ml/ folder.")
    sys.exit(1)

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
print(f"  Loaded {len(df)} records, {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")

# ── Preprocessing ─────────────────────────────────────────
# Handle column name variations across different Kaggle versions
col_map = {}
for col in df.columns:
    lower = col.lower().strip()
    if lower in ('jundice', 'jaundice'):
        col_map[col] = 'jaundice'
    elif lower in ('austim', 'autism'):
        col_map[col] = 'family_autism'
    elif lower in ('class/asd', 'class', 'asd'):
        col_map[col] = 'target'
    elif lower in ('contry_of_res', 'country_of_res'):
        col_map[col] = 'country'

df = df.rename(columns=col_map)

# Encode binary categoricals
if df['gender'].dtype == object:
    df['gender'] = df['gender'].map(lambda x: 1 if str(x).strip().lower() in ('m', 'male') else 0)

if df['jaundice'].dtype == object:
    df['jaundice'] = df['jaundice'].map(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

if df['family_autism'].dtype == object:
    df['family_autism'] = df['family_autism'].map(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# Target
if df['target'].dtype == object:
    df['target'] = df['target'].map(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)

# Handle missing age values
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median())

# Select features: 10 AQ-10 scores + 4 demographic features = 14
aq_cols = [c for c in df.columns if c.startswith('A') and c.endswith('_Score')]
aq_cols = sorted(aq_cols, key=lambda x: int(x.split('_')[0][1:]))  # A1..A10

feature_cols = aq_cols + ['age', 'gender', 'jaundice', 'family_autism']
print(f"\n  Features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols].values.astype(float)
y = df['target'].values.astype(int)

print(f"  Class distribution: NO={np.sum(y==0)}, YES={np.sum(y==1)}")

# ── Train/Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")

# ── Train Random Forest ──────────────────────────────────
print("\nTraining Random Forest (50 trees, max_depth=8)...")
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print(f"\n{'='*50}")
print(f"  RESULTS — Random Forest Classifier")
print(f"{'='*50}")
print(f"  Accuracy:   {accuracy:.4f}")
print(f"  Precision:  {precision:.4f}")
print(f"  Recall:     {recall:.4f}")
print(f"  F1 Score:   {f1:.4f}")
print(f"  Cross-val:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"\n  Feature Importance (top 5):")

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for i in range(min(5, len(feature_cols))):
    idx = sorted_idx[i]
    print(f"    {feature_cols[idx]:>15}: {importances[idx]:.4f}")

print(f"\n{classification_report(y_test, y_pred, target_names=['NO (Neurotypical)', 'YES (ASD Traits)'])}")

# ── Export Model to JSON ─────────────────────────────────
print("Exporting model to rf_model.json...")

def export_tree(estimator):
    """Convert a single sklearn decision tree to a flat node array."""
    tree = estimator.tree_
    nodes = []
    for i in range(tree.node_count):
        nodes.append({
            'f': int(tree.feature[i]),             # feature index (-2 = leaf)
            't': round(float(tree.threshold[i]), 6), # split threshold
            'l': int(tree.children_left[i]),        # left child index
            'r': int(tree.children_right[i]),       # right child index
            'v': [max(0, round(float(tree.value[i][0][0]))), max(0, round(float(tree.value[i][0][1])))]  # [n_class0, n_class1]
        })
    return nodes

model_json = {
    'model_type': 'RandomForestClassifier',
    'n_estimators': rf.n_estimators,
    'max_depth': rf.max_depth,
    'feature_names': feature_cols,
    'feature_importances': [round(float(x), 6) for x in importances],
    'trees': [export_tree(est) for est in rf.estimators_],
    'metrics': {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'cv_mean': round(float(cv_scores.mean()), 4),
        'cv_std': round(float(cv_scores.std()), 4)
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

out_path = os.path.join(os.path.dirname(__file__), 'rf_model.json')
with open(out_path, 'w') as f:
    json.dump(model_json, f)

file_size = os.path.getsize(out_path) / 1024
print(f"\n  Saved: {out_path}")
print(f"  Size:  {file_size:.1f} KB")
print(f"  Trees: {len(model_json['trees'])}")
print(f"\nDone! Model ready for Node.js inference.")
