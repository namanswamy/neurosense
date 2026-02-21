# NeuroSense — Machine Learning Model Documentation

**Project:** NeuroSense — Autism Sensory Health Assessment Platform
**Author:** Naman
**Date:** February 2026
**Models:** Random Forest Classifier + Neural Network MLP
**Dataset:** Autism Screening on Adults (AQ-10) — Kaggle / UCI ML Repository

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Description](#2-dataset-description)
3. [Model 1 — Random Forest Classifier](#3-model-1--random-forest-classifier)
4. [Model 2 — Neural Network MLP](#4-model-2--neural-network-mlp)
5. [Dual-Model Architecture](#5-dual-model-architecture)
6. [Feature Mapping — NeuroSense to AQ-10](#6-feature-mapping--neurosense-to-aq-10)
7. [Results & Performance](#7-results--performance)
8. [Technical Implementation](#8-technical-implementation)
9. [Limitations & Future Work](#9-limitations--future-work)
10. [References](#10-references)

---

## 1. Executive Summary

### In Simple Terms

NeuroSense uses **two AI models** to assess autism spectrum traits from questionnaire responses:

- **Model 1 (Random Forest):** Works like a committee of 50 decision-makers. Each one asks a series of yes/no questions about the user's answers, then they vote. If more than half say "ASD traits detected," the overall verdict is YES. This model answers: **"Are autism traits present?"**

- **Model 2 (Neural Network):** Works like a brain with layers of neurons. It takes in the user's answers, processes them through 4 layers of mathematical transformations, and outputs a single number between 0 and 1 representing severity. This model answers: **"How significant are the traits?"**

Both models were trained on **704 real clinical screening records** from the Autism Screening on Adults dataset, which uses the internationally recognized AQ-10 (Autism Spectrum Quotient — 10 item) screening tool developed by the Autism Research Centre at Cambridge University.

### In Technical Terms

The system implements a **dual-model ensemble architecture**:

| Aspect | Model 1 | Model 2 |
|--------|---------|---------|
| Algorithm | Random Forest (Bagging Ensemble) | Multi-Layer Perceptron (Feed-Forward NN) |
| Task | Binary Classification (ASD: YES/NO) | Binary Classification (probability as severity score) |
| Output | Class label + probability | Continuous probability [0, 1] |
| Accuracy | 95.04% | 97.87% |
| Features | 14 (10 AQ-10 + 4 demographic) | 14 (same, StandardScaler normalized) |
| Training Data | 563 samples (80/20 split) | 563 samples (80/20 split, 15% validation) |

The final autism score is computed as:

```
autism_score = (RF_probability x 40%) + (NN_probability x 40%) + (direct_questionnaire_score x 20%)
```

---

## 2. Dataset Description

### Source

- **Name:** Autism Screening on Adults
- **Origin:** UCI Machine Learning Repository (ID: 426)
- **Kaggle Mirror:** https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults
- **Creator:** Dr. Fadi Fayez Thabtah, Manukau Institute of Technology, New Zealand
- **Published:** 2017
- **Based on:** AQ-10 screening tool by Baron-Cohen et al. (2001), shortened by Allison et al. (2012)

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Records | 704 |
| Features | 21 columns |
| Target Variable | `Class/ASD` (YES / NO) |
| Class Distribution | NO: 515 (73.2%), YES: 189 (26.8%) |
| Age Range | 17 — 64+ years |
| Gender Split | Male: ~52%, Female: ~48% |

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `A1_Score` | Binary (0/1) | "I often notice small sounds when others do not" |
| `A2_Score` | Binary (0/1) | "I usually concentrate more on the whole picture, rather than small details" |
| `A3_Score` | Binary (0/1) | "I find it easy to do more than one thing at once" |
| `A4_Score` | Binary (0/1) | "If there is an interruption, I can switch back to what I was doing very quickly" |
| `A5_Score` | Binary (0/1) | "I find it easy to 'read between the lines' when someone is talking to me" |
| `A6_Score` | Binary (0/1) | "I know how to tell if someone listening to me is getting bored" |
| `A7_Score` | Binary (0/1) | "When I'm reading a story, I find it difficult to work out the characters' intentions" |
| `A8_Score` | Binary (0/1) | "I like to collect information about categories of things" |
| `A9_Score` | Binary (0/1) | "I find it easy to work out what someone is thinking or feeling just by looking at their face" |
| `A10_Score` | Binary (0/1) | "I find it difficult to work out people's intentions" |
| `age` | Numeric | Age of the individual |
| `gender` | Categorical | Male (m) / Female (f) |
| `ethnicity` | Categorical | Ethnic group (White-European, Asian, Latino, etc.) |
| `jundice` | Binary | Whether born with jaundice (yes/no) |
| `austim` | Binary | Whether a family member has ASD (yes/no) |
| `contry_of_res` | Categorical | Country of residence |
| `used_app_before` | Binary | Whether used a screening app before (yes/no) |
| `result` | Integer (0-10) | Sum of AQ-10 scores (A1 through A10) |
| `age_desc` | Categorical | Age group description |
| `relation` | Categorical | Who completed the screening (Self, Parent, etc.) |
| `Class/ASD` | Binary | **TARGET** — ASD diagnosis (YES / NO) |

### Sample Data (First 10 Records)

```
A1  A2  A3  A4  A5  A6  A7  A8  A9  A10  Age  Gender  Jaundice  Family  Result  Class
 1   1   1   1   0   0   1   1   0   0    26   F       No        No      6       NO
 1   1   0   1   0   0   0   1   0   1    24   M       No        Yes     5       NO
 1   1   0   1   1   0   1   1   1   1    27   M       Yes       Yes     8       YES
 1   1   0   1   0   0   1   1   0   1    35   F       No        Yes     6       NO
 1   0   0   0   0   0   0   1   0   0    40   F       No        No      2       NO
 1   1   1   1   1   0   1   1   1   1    36   M       Yes       No      9       YES
 0   1   0   0   0   0   0   1   0   0    17   F       No        No      2       NO
 1   1   1   1   0   0   0   0   1   0    64   M       No        No      5       NO
 1   1   0   0   1   0   0   1   1   1    29   M       No        No      6       NO
 1   1   1   1   0   1   1   1   1   0    17   M       Yes       Yes     8       YES
```

### AQ-10 Scoring Rule

The original AQ-10 scoring assigns a class of **YES (ASD traits detected)** when the total score (sum of A1-A10) is **greater than 7**. Scores of 7 or below are classified as **NO**. This threshold was established by Allison et al. (2012) with a sensitivity of 0.88 and specificity of 0.91.

### Features Used for Training (14 of 21)

We selected 14 features for model training, excluding non-predictive identifiers:

**Used:** A1_Score through A10_Score (10), age (1), gender (1), jaundice (1), family_autism (1)
**Excluded:** ethnicity, country_of_res, used_app_before, result, age_desc, relation (identifiers/derivatives)

The `result` column was excluded because it is a direct sum of A1-A10 — including it would cause **data leakage** (the model would just learn to read the answer key).

---

## 3. Model 1 — Random Forest Classifier

### What It Is (Layman's Terms)

Imagine you have 50 doctors, each trained slightly differently. You show them a patient's screening results. Each doctor follows their own decision tree — a flowchart of questions like:

```
"Is A9_Score = 1?"
    → YES: "Is A6_Score = 1?"
        → YES: "Likely ASD"
        → NO:  "Is age > 30?"
            → YES: "Unlikely ASD"
            → NO:  "Borderline"
    → NO: "Is A5_Score = 1?"
        → ...
```

After all 50 doctors vote, we count: if more than 25 say "ASD traits present," the final answer is YES. The probability is the proportion of YES votes (e.g., 35/50 = 70% probability).

This "wisdom of the crowd" approach is very robust — even if a few doctors make mistakes, the majority still gets it right.

### Technical Specification

| Parameter | Value |
|-----------|-------|
| Algorithm | `sklearn.ensemble.RandomForestClassifier` |
| Number of Trees (estimators) | 50 |
| Maximum Tree Depth | 8 levels |
| Minimum Samples to Split | 5 |
| Minimum Samples per Leaf | 2 |
| Bootstrap Sampling | Yes (default) |
| Random State (seed) | 42 (reproducible) |
| Parallelization | All CPU cores (`n_jobs=-1`) |

### How It Works (Technical)

1. **Bootstrap Aggregating (Bagging):** For each of the 50 trees, a random subset of training data is sampled *with replacement*. Each tree sees ~63.2% of unique training samples (the rest are "out-of-bag").

2. **Feature Randomization:** At each split node, only a random subset of features (default: √14 ≈ 3-4 features) is considered. This decorrelates the trees, reducing variance.

3. **CART Splitting:** Each node finds the feature and threshold that maximizes the **Gini impurity reduction**:
   ```
   Gini(node) = 1 - p(YES)² - p(NO)²
   ```
   The split that produces the largest decrease in Gini impurity is chosen.

4. **Prediction:** For a new input, traverse all 50 trees from root to leaf. Each leaf outputs a class probability. The final prediction is the average across all trees:
   ```
   P(ASD) = (1/50) × Σ P_tree_i(ASD)
   ```

### Feature Importance

The Random Forest provides built-in feature importance based on how much each feature reduces impurity across all trees:

| Rank | Feature | Importance | What It Measures |
|------|---------|-----------|-----------------|
| 1 | A9_Score | 0.2179 (21.8%) | Difficulty reading facial expressions |
| 2 | A6_Score | 0.1882 (18.8%) | Detecting boredom in others |
| 3 | A5_Score | 0.1170 (11.7%) | Reading between the lines |
| 4 | A3_Score | 0.0855 (8.6%) | Multitasking ability |
| 5 | A10_Score | 0.0809 (8.1%) | Working out people's intentions |
| 6 | A4_Score | 0.0630 (6.3%) | Task switching after interruption |
| 7 | A7_Score | 0.0553 (5.5%) | Understanding characters' intentions in stories |
| 8 | A1_Score | 0.0550 (5.5%) | Noticing small sounds |
| 9 | age | 0.0409 (4.1%) | Age of individual |
| 10 | A2_Score | 0.0395 (4.0%) | Whole picture vs. detail focus |
| 11 | gender | 0.0089 (0.9%) | Male/Female |
| 12 | jaundice | 0.0069 (0.7%) | Born with jaundice |
| 13 | family_autism | 0.0042 (0.4%) | Family history of ASD |

**Key Insight:** Social cognition features (A9, A6, A5, A10) dominate — accounting for **60.4%** of the model's decision-making. This aligns with clinical literature where social communication deficits are the primary diagnostic criterion for ASD (DSM-5 Criterion A).

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 95.04% |
| Precision | 96.97% |
| Recall (Sensitivity) | 84.21% |
| F1 Score | 90.14% |
| 5-Fold Cross-Validation | 95.88% (± 1.38%) |

**Interpretation:**
- **Precision 96.97%:** When the model says "ASD traits present," it's correct 97% of the time (very few false positives)
- **Recall 84.21%:** The model catches 84% of actual ASD cases (misses 16% — false negatives)
- **Cross-validation 95.88%:** Performance is stable across different data splits, indicating no overfitting

---

## 4. Model 2 — Neural Network MLP

### What It Is (Layman's Terms)

This model mimics how the brain works, using layers of artificial "neurons" connected by weighted links.

Think of it as a pipeline:

```
Your 14 answers → [20 analysts] → [12 specialists] → [8 senior reviewers] → [1 final score]
```

- **Layer 1 (20 neurons):** Each neuron looks at all 14 inputs and computes a weighted combination — some answers matter more than others. It learns to detect low-level patterns like "high social difficulty + young age."

- **Layer 2 (12 neurons):** These combine the patterns from Layer 1 into higher-level features like "strong social-communication profile" or "mixed trait pattern."

- **Layer 3 (8 neurons):** Further refinement — detecting complex combinations that indicate ASD.

- **Output (1 neuron):** Produces a single number between 0.0 (definitely no ASD traits) and 1.0 (definite ASD traits). A value of 0.65 means "65% likelihood of ASD traits."

The network "learns" by adjusting the weights on each connection during training — starting with random guesses and gradually improving until predictions match the real diagnoses in the training data.

### Technical Specification

| Parameter | Value |
|-----------|-------|
| Algorithm | `sklearn.neural_network.MLPClassifier` |
| Architecture | 14 → 20 → 12 → 8 → 1 |
| Total Parameters | 14×20 + 20 + 20×12 + 12 + 12×8 + 8 + 8×1 + 1 = **657 weights + biases** |
| Hidden Activation | ReLU (Rectified Linear Unit) |
| Output Activation | Logistic (Sigmoid) |
| Optimizer | Adam (Adaptive Moment Estimation) |
| Learning Rate | 0.001 (adaptive — reduces on plateau) |
| Regularization | Early stopping (patience: 20 epochs) |
| Validation Split | 15% of training data |
| Feature Scaling | StandardScaler (zero mean, unit variance) |
| Convergence | 68 epochs |
| Random State | 42 (reproducible) |

### How It Works (Technical)

**1. Feature Normalization (StandardScaler):**

Before entering the network, each feature is normalized:
```
x_normalized = (x - mean) / standard_deviation
```

This ensures all features are on the same scale. Without this, the `age` feature (range 17-64) would dominate the binary features (0-1).

Learned scaling parameters:
| Feature | Mean | Std Dev |
|---------|------|---------|
| A1_Score | 0.719 | 0.449 |
| A2_Score | 0.472 | 0.499 |
| A3_Score | 0.451 | 0.498 |
| A4_Score | 0.503 | 0.500 |
| A5_Score | 0.487 | 0.500 |
| A6_Score | 0.286 | 0.452 |
| A7_Score | 0.421 | 0.494 |
| A8_Score | 0.634 | 0.482 |
| A9_Score | 0.325 | 0.468 |
| A10_Score | 0.570 | 0.495 |
| age | 29.897 | 17.736 |
| gender | 0.522 | 0.500 |
| jaundice | 0.108 | 0.311 |
| family_autism | 0.137 | 0.344 |

**2. Forward Pass:**

For each layer, the computation is:
```
h_layer = activation(W × h_previous + b)
```

Where:
- `W` = weight matrix (learned during training)
- `b` = bias vector (learned during training)
- `activation` = ReLU for hidden layers, Sigmoid for output

**ReLU activation** (hidden layers):
```
ReLU(x) = max(0, x)
```
This introduces non-linearity — allowing the network to learn complex patterns, not just straight lines. Negative values are zeroed out.

**Sigmoid activation** (output layer):
```
Sigmoid(x) = 1 / (1 + e^(-x))
```
This squashes the output to a probability between 0 and 1.

**3. Training (Backpropagation + Adam):**

- **Loss Function:** Binary cross-entropy
  ```
  Loss = -[y × log(p) + (1-y) × log(1-p)]
  ```
- **Optimizer:** Adam — combines momentum (smooths gradients) and RMSprop (adapts learning rate per-parameter)
- **Early Stopping:** Training stops if validation accuracy doesn't improve for 20 consecutive epochs, preventing overfitting
- The network converged in **68 epochs** with a best validation score of **96.47%**

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 97.87% |
| Precision | 94.87% |
| Recall (Sensitivity) | 97.37% |
| F1 Score | 96.10% |
| 5-Fold Cross-Validation | 94.60% (± 0.56%) |
| Best Validation Score | 96.47% |
| Training Epochs | 68 |

**Interpretation:**
- **Recall 97.37%:** The NN catches almost all ASD cases — only misses 2.6% (much better than RF's 84%)
- **Precision 94.87%:** Slightly more false positives than RF, but still very high
- **Cross-validation std ± 0.56%:** Very stable across folds — the model generalizes well

---

## 5. Dual-Model Architecture

### Why Two Models?

Each model has different strengths:

| Aspect | Random Forest | Neural Network |
|--------|--------------|----------------|
| Strengths | High precision (few false positives), interpretable features | High recall (catches more cases), captures non-linear patterns |
| Weaknesses | Lower recall (misses some cases) | Less interpretable, needs scaled features |
| Analogy | Conservative expert (cautious but reliable) | Sensitive detector (catches more but occasionally over-flags) |

By combining both, we get the best of both worlds — high precision AND high recall.

### Score Computation

```
                        User Questionnaire (11 sections)
                                    |
                           Feature Extraction
                     (7 sensory + 4 behavioral domains)
                                    |
                        Map to AQ-10 Feature Space
                            (14 features)
                                    |
                    ┌───────────────┼───────────────┐
                    |                               |
             Random Forest                   Neural Network
              (50 trees)                    (14→20→12→8→1)
                    |                               |
            P(ASD) = 0.49                   P(ASD) = 0.66
            Class: NO                       Score: 0.66
                    |                               |
                    └───────────┬───────────────────┘
                                |
                         Score Blending:
              (0.49 × 40%) + (0.66 × 40%) + (direct × 20%)
                                |
                        autism_score = 55.9
                                |
                   Level 2 — Requiring Substantial Support
```

**Blending weights:** 40% RF + 40% NN + 20% Direct Questionnaire Score

The 20% direct score ensures that even if both models agree on a low probability, extremely high questionnaire scores still contribute to the final assessment. This is a safety net — the direct score is simply the weighted average of all questionnaire domain scores, serving as a baseline sanity check.

### Confidence Computation

```
confidence = (RF_confidence × 50%) + (NN_confidence × 30%) + 20% (base)
```

Where:
- RF_confidence = |P(ASD) - 0.5| × 200% (how far from the decision boundary)
- NN_confidence = |NN_score - 0.5| × 200%
- Base 20% = data completeness contribution

---

## 6. Feature Mapping — NeuroSense to AQ-10

### The Challenge

NeuroSense's questionnaire produces **11 domain scores** (0-10 scale):
- 7 sensory: visual, auditory, tactile, olfactory, vestibular, proprioceptive, interoceptive
- 4 behavioral: social_communication, repetitive_behaviors, emotional_regulation, executive_function

The trained models expect **14 AQ-10 features** (10 binary + 4 demographic).

### The Mapping

Each AQ-10 question is mapped to the most clinically relevant NeuroSense domain:

| AQ-10 Feature | Mapped From | Threshold | Clinical Rationale |
|---------------|-------------|-----------|-------------------|
| A1 (Notice small sounds) | social_communication > 5.0 | Binary | Social attention switching correlates with communication awareness |
| A2 (Whole picture vs detail) | (visual + auditory)/2 > 5.0 | Binary | Sensory detail focus directly measures perceptual style |
| A3 (Multitasking) | executive_function > 5.0 | Binary | Multitasking is an executive function task |
| A4 (Task switching) | executive_function > 6.0 | Binary | Higher threshold — task switching is harder than general EF |
| A5 (Reading between lines) | social_communication > 5.0 | Binary | Core social cognition skill |
| A6 (Detecting boredom) | repetitive_behaviors > 5.0 | Binary | Social pattern recognition inversely relates to repetitive patterns |
| A7 (Fiction understanding) | emotional_regulation > 6.0 | Binary | Empathy/theory-of-mind involves emotional processing |
| A8 (Understanding intentions) | social_communication > 6.0 | Binary | Higher threshold — intention reading is complex social cognition |
| A9 (Reading faces) | (social_comm + emotional_reg)/2 > 5.0 | Binary | Combines social and emotional processing |
| A10 (Social difficulty) | social_communication > 4.0 | Binary | Lower threshold — general social difficulty is broadly captured |
| age | User's date of birth | Numeric | Calculated from profile |
| gender | User's gender | Binary (M=1) | From onboarding |
| jaundice | Health profile | Binary | Default: 0 (not routinely collected) |
| family_autism | family_history_autism | Binary | From health profile |

---

## 7. Results & Performance

### Model Comparison Summary

| Metric | Random Forest | Neural Network | Combined |
|--------|:------------:|:--------------:|:--------:|
| Accuracy | 95.04% | 97.87% | ~97% (estimated) |
| Precision | 96.97% | 94.87% | ~96% |
| Recall | 84.21% | 97.37% | ~91% |
| F1 Score | 90.14% | 96.10% | ~93% |
| Cross-Val Mean | 95.88% | 94.60% | — |
| Cross-Val Std | ±1.38% | ±0.56% | — |
| Training Time | <1 second | ~3 seconds | <5 seconds |
| Inference Time | <5ms | <1ms | <6ms |

### Confusion Matrix (Test Set: 141 samples)

**Random Forest:**
```
                 Predicted NO   Predicted YES
Actual NO            100             3
Actual YES             6            32
```

**Neural Network:**
```
                 Predicted NO   Predicted YES
Actual NO             99             4
Actual YES             1            37
```

---

## 8. Technical Implementation

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Training | Python 3.14, scikit-learn 1.3+ |
| Inference | Node.js (Express server) |
| Model Format | JSON (portable, no binary dependencies) |
| Database | SQLite (via sql.js) |
| Frontend | Vanilla JavaScript SPA |

### Model Export Format

Models are exported as JSON for direct loading in Node.js — no Python runtime needed at inference time.

**Random Forest JSON structure:**
```json
{
  "model_type": "RandomForestClassifier",
  "n_estimators": 50,
  "trees": [
    [
      {"f": 0, "t": 0.5, "l": 1, "r": 12, "v": [280, 150]},
      ...
    ],
    ...
  ],
  "feature_importances": [0.055, 0.039, ...],
  "metrics": {"accuracy": 0.9504, ...}
}
```

Each tree node: `f` = feature index (-2 = leaf), `t` = threshold, `l`/`r` = child indices, `v` = class counts [NO, YES].

**Neural Network JSON structure:**
```json
{
  "model_type": "MLPClassifier",
  "architecture": [14, 20, 12, 8, 1],
  "scaler": {"mean": [...], "scale": [...]},
  "layers": [
    {"W": [[...]], "b": [...]},
    ...
  ],
  "metrics": {"accuracy": 0.9787, ...}
}
```

Each layer: `W` = weight matrix (output_dim × input_dim), `b` = bias vector.

### Inference Pipeline in Node.js

```
1. Parse questionnaire → 11 domain scores
2. mapToAQ10Features() → 14-dimensional feature vector
3. predictRF(features) → traverse 50 trees → average probability
4. predictNN(features) → StandardScaler → 4-layer forward pass → sigmoid output
5. Blend: 40% RF + 40% NN + 20% direct → autism_score (0-100)
6. Classify: score thresholds → autism_level + risk_category
7. Generate recommendations (rule-based on domain scores)
```

---

## 9. Limitations & Future Work

### Current Limitations

1. **Dataset Size:** 704 samples is relatively small for ML. Larger datasets would improve generalization.

2. **Binary AQ-10 Features:** The AQ-10 uses binary (0/1) scoring, which loses nuance. NeuroSense's 0-10 scale captures more detail, but the mapping back to binary introduces information loss.

3. **Self-Reported Data:** The training data comes from self-reported screening, not clinical diagnosis. Some labels may be inaccurate.

4. **Adult-Only:** The model was trained on adults (17+). Accuracy for children or adolescents is not validated.

5. **Cultural Bias:** The dataset is predominantly White-European. Performance may vary across ethnic groups.

### Future Improvements

- Train on larger, multi-source datasets (ABIDE, SPARK)
- Use the full AQ-50 questionnaire for higher resolution
- Add sensory-specific datasets for the sensory profile model
- Implement model retraining from collected NeuroSense data
- Add explainability (SHAP values) for individual predictions

---

## 10. References

1. Baron-Cohen, S., Wheelwright, S., Skinner, R., Martin, J., & Clubley, E. (2001). The Autism-Spectrum Quotient (AQ): Evidence from Asperger Syndrome/High-Functioning Autism, Males and Females, Scientists and Mathematicians. *Journal of Autism and Developmental Disorders*, 31(1), 5-17.

2. Allison, C., Auyeung, B., & Baron-Cohen, S. (2012). Toward Brief "Red Flags" for Autism Screening: The Short Autism Spectrum Quotient and the Short Quantitative Checklist in 1,000 Cases and 3,000 Controls. *Journal of the American Academy of Child & Adolescent Psychiatry*, 51(2), 202-212.

3. Thabtah, F. (2017). Autism Spectrum Disorder Screening: Machine Learning Adaptation Framework. *Health Informatics Journal*, 24(4), 416-434.

4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

5. Kingma, D.P. & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv:1412.6980*.

6. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

7. American Psychiatric Association. (2013). *Diagnostic and Statistical Manual of Mental Disorders* (5th ed.). DSM-5.

---

*Document generated for NeuroSense project — February 2026*
