# NeuroSense — Training Dataset Documentation

**Project:** NeuroSense — Autism Sensory Health Assessment Platform
**Author:** Naman
**Date:** February 2026
**Dataset:** Autism Screening on Adults (AQ-10)
**Source:** UCI Machine Learning Repository / Kaggle

---

## 1. What Is This Dataset?

This dataset contains **704 real screening records** collected from adults who took the **AQ-10 test** — a quick, 10-question checklist used by clinicians worldwide to spot early signs of Autism Spectrum Disorder (ASD).

Each record captures three things about a person:

- **10 behavioral answers** — scored as 0 (no trait) or 1 (trait present)
- **Basic demographics** — age, gender, ethnicity, country
- **Screening outcome** — YES (ASD traits detected) or NO (neurotypical)

We used this dataset to train both ML models (Random Forest + Neural Network) that power NeuroSense's assessment engine.

**Technically speaking:** It is a labeled binary classification dataset — 704 instances, 21 attributes, with a 73:27 class split (NO:YES). The features are a mix of binary, categorical, and continuous variables.

---

## 2. Where Does It Come From?

| Property | Details |
|----------|---------|
| **Name** | Autism Screening Adult Data Set |
| **Repository** | UCI Machine Learning Repository (ID: 426) |
| **Kaggle Mirror** | kaggle.com/datasets/andrewmvd/autism-screening-on-adults |
| **Created By** | Dr. Fadi Fayez Thabtah, Manukau Institute of Technology, New Zealand |
| **Year** | 2017 |
| **Based On** | AQ-10 screening tool — developed at Cambridge University by Baron-Cohen et al. |
| **License** | Creative Commons Attribution 4.0 (free to use) |

The AQ-10 (Autism Spectrum Quotient, 10-item version) was originally developed as a shortened "red flag" screening from the full 50-question AQ test. It was validated on 1,000 ASD cases and 3,000 controls with 88% sensitivity and 91% specificity (Allison et al., 2012).

---

## 3. Dataset at a Glance

| Property | Value |
|----------|-------|
| Total Records | 704 people |
| Total Columns | 21 |
| Behavioral Questions | 10 (A1 through A10) |
| Demographic Columns | 10 (age, gender, etc.) |
| Target Column | Class/ASD (YES or NO) |
| File Size | ~68 KB |

### Who's In the Dataset?

| Group | Breakdown |
|-------|-----------|
| **Gender** | 52% male, 48% female |
| **Age** | 17 to 64 years (average: 30) |
| **Top Ethnicities** | White-European (33%), Asian (18%), Middle Eastern (13%) |
| **Countries** | 60+ countries — US (16%), UAE (12%), NZ (12%), India (12%), UK (11%) |
| **Jaundice at birth** | 10% yes, 90% no |
| **Family autism history** | 13% yes, 87% no |
| **Filled by** | 74% self-reported, 7% by parent, 4% by relative |

### How Many Have ASD Traits?

| Class | Count | Percentage |
|-------|-------|------------|
| NO (Neurotypical) | 515 | 73% |
| YES (ASD Traits) | 189 | 27% |

About 1 in 4 people in this dataset screened positive for ASD traits. In the real world, ASD prevalence is much lower (~1-2%), so this dataset intentionally includes more positive cases to help the ML models learn the difference.

---

## 4. The 10 Questions (AQ-10)

Each question targets a specific behavioral area. A score of **1** means the person's answer is associated with autism traits.

| # | Question (simplified) | What It Tests |
|---|----------------------|---------------|
| A1 | "I notice small sounds others miss" | Sensory sensitivity |
| A2 | "I focus on details, not the big picture" | Detail orientation |
| A3 | "I find multitasking difficult" | Executive function |
| A4 | "After an interruption, I struggle to refocus" | Attention switching |
| A5 | "I struggle to read between the lines" | Social understanding |
| A6 | "I can't tell when someone is bored" | Social awareness |
| A7 | "I struggle to understand characters in stories" | Empathy / theory of mind |
| A8 | "I like collecting and categorizing information" | Pattern-seeking behavior |
| A9 | "I struggle to read facial expressions" | Emotional recognition |
| A10 | "I find it hard to understand people's intentions" | Social cognition |

**Scoring rule:** If the total (A1 + A2 + ... + A10) is **greater than 7**, the AQ-10 flags the person as likely having ASD traits.

### Other Columns

| Column | What It Is | Used in Training? |
|--------|-----------|-------------------|
| age | Person's age in years | Yes |
| gender | Male or Female | Yes |
| jaundice | Born with jaundice? | Yes — it's a known ASD risk factor |
| family_autism | Family member with ASD? | Yes — genetics play a role |
| ethnicity | Ethnic background | No — 13.5% missing, not a direct predictor |
| country | Country of residence | No — too many categories (60+) |
| result | Sum of A1-A10 | No — this would be "cheating" (data leakage) |
| used_app_before | Used screening app before? | No — 98% said no, so it's useless |
| age_desc | Age group label | No — redundant with age |
| relation | Who filled the form | No — metadata, not a trait |

**In total, we use 14 features for training:** 10 AQ scores + age + gender + jaundice + family history.

---

## 5. Which Questions Matter Most?

We measured how strongly each question correlates with an ASD diagnosis:

| Rank | Question | Correlation | In Plain English |
|------|----------|-------------|-----------------|
| 1 | A9 — Reading faces | 0.64 (Strong) | People who can't read faces are most likely to have ASD traits |
| 2 | A6 — Detecting boredom | 0.59 (Strong) | Not noticing social cues is a strong indicator |
| 3 | A5 — Reading between lines | 0.54 (Moderate) | Difficulty with implied meaning is common in ASD |
| 4 | A4 — Task switching | 0.47 (Moderate) | Inflexibility with change is a core ASD trait |
| 5 | A3 — Multitasking | 0.44 (Moderate) | Executive function difficulties are typical |
| 6 | A10 — Understanding intentions | 0.39 (Moderate) | Social cognition challenges |
| 7 | A7 — Story character intentions | 0.35 (Moderate) | Theory of mind difficulties |
| 8 | A2 — Detail vs big picture | 0.31 (Weak) | Common in general population too |
| 9 | A1 — Small sounds | 0.30 (Weak) | Many people notice small sounds |
| 10 | A8 — Collecting info | 0.24 (Weak) | Pattern-seeking is widespread |

**The takeaway:** Social skills questions (A9, A6, A5, A10) are the strongest predictors — which aligns with DSM-5, where social communication deficits are the #1 diagnostic criterion for ASD.

Our Random Forest model independently learned the same ranking (A9 > A6 > A5), confirming it captured clinically meaningful patterns from the data.

---

## 6. Score Comparison: ASD vs Non-ASD

| Measure | NO (Neurotypical) | YES (ASD Traits) |
|---------|-------------------|-------------------|
| Average total score | 3.6 out of 10 | 8.3 out of 10 |
| Median score | 4 | 8 |
| Score range | Mostly 1-6 | Mostly 7-10 |

The difference is clear: people with ASD traits score **4.7 points higher** on average. The clinical threshold of 7 cleanly separates the two groups, which is why the AQ-10 is such an effective screening tool.

---

## 7. Sample Records

Here are a few real records from the dataset to show what the data looks like:

### A Clear "NO" Case — Row 5
```
A1=1, A2=0, A3=0, A4=0, A5=0, A6=0, A7=0, A8=1, A9=0, A10=0
Age: 40, Gender: F, Jaundice: No, Family: No
Total Score: 2/10 → Class: NO
```
Only scored on A1 (sounds) and A8 (collecting info) — these are common traits even in neurotypical people. Score of 2 is well below the threshold of 7.

### A Clear "YES" Case — Row 11
```
A1=1, A2=1, A3=1, A4=1, A5=1, A6=1, A7=1, A8=1, A9=1, A10=1
Age: 33, Gender: M, Jaundice: No, Family: No
Total Score: 10/10 → Class: YES
```
Scored 1 on every single question — maximum possible score. Difficulty with social cues, multitasking, empathy, and more.

### A Borderline Case — Row 1
```
A1=1, A2=1, A3=1, A4=1, A5=0, A6=0, A7=1, A8=1, A9=0, A10=0
Age: 26, Gender: F, Jaundice: No, Family: No
Total Score: 6/10 → Class: NO
```
Scored 6 — just one point below the clinical threshold. Classified as NO, but shows attention-to-detail and empathy traits. These borderline cases are exactly where ML models add value over simple threshold rules.

---

## 8. Data Cleaning Steps

The raw dataset had a few issues that we fixed before training:

| Issue | What We Did |
|-------|-------------|
| Column name typos ("jundice", "austim", "contry_of_res") | Renamed to correct spellings |
| Text values ("yes"/"no", "m"/"f") | Converted to numbers (1/0) |
| One person listed as age 383 | Replaced with the median age (27) |
| 13.5% missing ethnicity values | Excluded ethnicity from model features |
| "result" column = sum of A1-A10 | Excluded — using it would be data leakage |

**Data leakage explained simply:** The "result" column is just A1+A2+...+A10 added up. If we gave this to the model, it would learn to just read the answer instead of learning the patterns. That's cheating — so we removed it.

---

## 9. How NeuroSense Uses This Data

### Training (One-Time)

```
704 screening records
        |
  Clean & encode (14 features selected)
        |
  Split: 80% training (563) / 20% testing (141)
        |
  +------------------+------------------+
  |                                     |
  Random Forest                  Neural Network
  (50 decision trees)            (4-layer MLP)
  |                                     |
  rf_model.json                  nn_model.json
```

### At Runtime (Every Assessment)

When a user completes NeuroSense's questionnaire:

1. Their 11 domain scores (visual, auditory, social, etc.) are **mapped** to the 14 AQ-10 features using clinical threshold rules
2. Both models process the features independently
3. Results are blended: **40% Random Forest + 40% Neural Network + 20% direct questionnaire score**
4. Final autism score (0-100) is displayed with confidence level

---

## 10. Important Notes

- **This is a screening tool, not a diagnosis.** A positive result means "consider a clinical evaluation" — not "you have autism."
- **Self-reported data** — accuracy depends on honest, self-aware responses.
- **Adults only** (17+) — not validated for children.
- **Cultural considerations** — the AQ-10 was developed in the UK; social norms vary across cultures.
- **No personal data** — all records are fully anonymous.

---

## 11. References

1. Baron-Cohen, S. et al. (2001). "The Autism-Spectrum Quotient (AQ)." *Journal of Autism and Developmental Disorders*, 31(1), 5-17.

2. Allison, C. et al. (2012). "Toward Brief Red Flags for Autism Screening." *JAACAP*, 51(2), 202-212.

3. Thabtah, F. (2017). "ASD Screening: ML Adaptation Framework." *Health Informatics Journal*, 24(4), 416-434.

4. American Psychiatric Association. (2013). *DSM-5.*

5. Dua, D. & Graff, C. (2019). UCI Machine Learning Repository.

---

*Document generated for NeuroSense project — February 2026*
