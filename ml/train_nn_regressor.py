"""
============================================================
NeuroSense — Model 2: Neural Network MLP Classifier
Trains on Kaggle 'Autism Screening on Adults' dataset (AQ-10)
Exports weights + biases as JSON for Node.js inference
============================================================
Dataset: https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults
Place the CSV file in this directory as 'autism_screening.csv'
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
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

# ── Preprocessing ─────────────────────────────────────────
# Handle column name variations
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

if df['target'].dtype == object:
    df['target'] = df['target'].map(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median())

# Select features: 10 AQ-10 scores + 4 demographic features = 14
aq_cols = [c for c in df.columns if c.startswith('A') and c.endswith('_Score')]
aq_cols = sorted(aq_cols, key=lambda x: int(x.split('_')[0][1:]))

feature_cols = aq_cols + ['age', 'gender', 'jaundice', 'family_autism']
print(f"  Features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols].values.astype(float)
y = df['target'].values.astype(int)

print(f"  Class distribution: NO={np.sum(y==0)}, YES={np.sum(y==1)}")

# ── Train/Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Feature Scaling (required for neural networks) ───────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Scaler: mean={[round(x, 3) for x in scaler.mean_[:3]]}... scale={[round(x, 3) for x in scaler.scale_[:3]]}...")

# ── Train MLP Neural Network ─────────────────────────────
# Architecture: 14 → 20 → 12 → 8 → 1 (matches NeuroSense dashboard)
print("\nTraining Neural Network MLP (14→20→12→8→1)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(20, 12, 8),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    verbose=False
)
mlp.fit(X_train_scaled, y_train)

print(f"  Converged in {mlp.n_iter_} iterations")
print(f"  Best validation score: {mlp.best_validation_score_:.4f}")

# ── Evaluate ─────────────────────────────────────────────
y_pred = mlp.predict(X_test_scaled)
y_prob = mlp.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cv_scores = cross_val_score(
    MLPClassifier(hidden_layer_sizes=(20, 12, 8), activation='relu', solver='adam',
                  max_iter=2000, random_state=42, early_stopping=True),
    scaler.transform(X), y, cv=5, scoring='accuracy'
)

print(f"\n{'='*50}")
print(f"  RESULTS — Neural Network MLP Classifier")
print(f"{'='*50}")
print(f"  Accuracy:   {accuracy:.4f}")
print(f"  Precision:  {precision:.4f}")
print(f"  Recall:     {recall:.4f}")
print(f"  F1 Score:   {f1:.4f}")
print(f"  Cross-val:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Show layer info
print(f"\n  Network Architecture:")
layer_sizes = [X.shape[1]] + list(mlp.hidden_layer_sizes) + [1]
for i, (inp, out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    act = 'ReLU' if i < len(mlp.hidden_layer_sizes) else 'Logistic (Sigmoid)'
    print(f"    Layer {i+1}: {inp} → {out} ({act})")
    print(f"      Weights shape: ({out}, {inp}), Bias shape: ({out},)")

print(f"\n{classification_report(y_test, y_pred, target_names=['NO (Neurotypical)', 'YES (ASD Traits)'])}")

# ── Export Model to JSON ─────────────────────────────────
print("Exporting model to nn_model.json...")

# sklearn stores coefs_ as (input_dim, output_dim)
# We transpose to (output_dim, input_dim) for Node.js matVecMul
layers = []
for i in range(len(mlp.coefs_)):
    W = mlp.coefs_[i].T  # transpose: (output, input)
    b = mlp.intercepts_[i]
    layers.append({
        'W': [[round(float(w), 8) for w in row] for row in W],
        'b': [round(float(x), 8) for x in b]
    })

model_json = {
    'model_type': 'MLPClassifier',
    'architecture': [int(X.shape[1])] + list(mlp.hidden_layer_sizes) + [1],
    'activation': 'relu',
    'output_activation': mlp.out_activation_,
    'scaler': {
        'mean': [round(float(x), 8) for x in scaler.mean_],
        'scale': [round(float(x), 8) for x in scaler.scale_]
    },
    'layers': layers,
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
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'epochs': int(mlp.n_iter_),
        'best_val_score': round(float(mlp.best_validation_score_), 4)
    }
}

out_path = os.path.join(os.path.dirname(__file__), 'nn_model.json')
with open(out_path, 'w') as f:
    json.dump(model_json, f)

file_size = os.path.getsize(out_path) / 1024
print(f"\n  Saved: {out_path}")
print(f"  Size:  {file_size:.1f} KB")
print(f"  Layers: {len(layers)}")
print(f"\nDone! Model ready for Node.js inference.")
