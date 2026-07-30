# Comment Category Prediction Challenge

Multiclass classification of user-generated comments using a hybrid TF-IDF + engineered-metadata pipeline, tuned Logistic Regression / LightGBM models, and a soft-voting ensemble.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## Overview

An online discussion platform automatically assigns each comment to one of four internal handling categories based on its content, engagement, and metadata. This project builds a model that reproduces that decision — treating it as a **4-class classification problem** over a dataset that mixes raw text, engagement counts, timestamps, emoticon flags, and (heavily missing) identity attributes.

The dataset is intentionally messy and heterogeneous, which makes this less of a "clean TF-IDF classifier" problem and more of a **feature-engineering + pipeline-design** problem: getting text, categorical, numeric, and binary signals to play well together inside a single `scikit-learn` `Pipeline`.

## Problem Statement

- **Task:** Predict `label` (classes `0, 1, 2, 3`) for each comment.
- **Metric:** Macro F1-score — the target is heavily imbalanced, so accuracy alone is misleading.
- **Challenge:** Combine unstructured text, engagement signals, and sparse identity metadata into one reliable prediction pipeline.

## Dataset

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 198,000 | 15 | Includes target `label` |
| `test.csv` | 102,000 | 14 | No target |
| `Sample.csv` | 102,000 | 2 | Submission format |

**Columns**

| Column | Description |
|---|---|
| `comment` | Raw text of the comment |
| `created_date` | Timestamp the comment was posted |
| `post_id` | Thread/parent post identifier |
| `emoticon_1/2/3` | Presence flags for internal emoticon groups |
| `upvote`, `downvote` | Engagement counts |
| `if_1`, `if_2` | Hidden internal platform features |
| `race`, `religion`, `gender` | System-detected identity references (~73% missing) |
| `disability` | System-detected ability reference (binary) |
| `label` | Target — final assigned category |

## Exploratory Data Analysis — key findings

- **Class imbalance:** class `0` is 57.7% of the data, class `2` is 31.5%, while classes `1` (8.0%) and `3` (2.8%) are minority classes → Macro F1 chosen as the evaluation metric over accuracy.
- **Missingness:** `race`, `religion`, `gender` are ~73% missing and too sparse to use directly — converted into presence flags instead of raw categories.
- **No duplicate rows** in either train or test.
- **Comment length/word count** vary widely across the dataset, supporting text-meta features (length, punctuation density, uppercase ratio).
- **Engagement outliers** (very high upvotes/downvotes) reflect real viral/controversial comments rather than noise — handled via log transforms instead of removal.
- **`post_id` overlap between train and test is 100%** (45 of 45 test post IDs also appear in train) — confirmed safe to use post-level features without leakage risk.
- **Temporal and identity-flag signals** show visible variation across labels, motivating their inclusion as engineered features.

## Feature Engineering

**Text cleaning** — HTML/entity stripping, URL/email/mention/hashtag normalization (with CamelCase hashtag splitting), number tokenization, case normalization, and negation-scope tagging before vectorization.

**Engineered feature groups:**
- **Text-meta:** length, word count, unique words, lexical diversity, punctuation/uppercase ratios, repeated characters/words, emphasis and sentiment-signal counts
- **Temporal:** year, month, day, hour, minute, weekday, `is_weekend`, `days_since_start`
- **Engagement:** `net_votes`, `total_reactions`, `upvote_ratio`, `downvote_ratio`, `controversy`, log-transformed vote/internal-feature counts
- **Identity:** `race_flag`, `religion_flag`, `gender_flag`, `identity_count`, plus interaction terms (`identity_x_upvote`, `identity_x_downvote`)

This expanded the raw 15 columns into **70 modeling features**.

## Modeling Pipeline

A single `ColumnTransformer` routes each feature group to the right preprocessing:

```
text        → hybrid TF-IDF (word 1-2gram, 50K features  +  char_wb 3-5gram, 30K features)
categorical → constant-impute + one-hot
numeric     → median-impute + Standard/MinMax scaling
binary      → passthrough
```

**Models compared (baseline, stratified 80/20 split):**

| Model | Accuracy | Macro F1 |
|---|---|---|
| **LightGBM** | **0.92** | **0.82** |
| Logistic Regression | — | close second |
| Ridge / SGD | — | mid-tier |
| Complement / Multinomial / Bernoulli NB | — | weakest |

LightGBM led the baseline sweep, closely followed by Logistic Regression — confirming that both sparse text signal and nonlinear metadata interactions matter here. Naive Bayes variants underperformed once sparse TF-IDF was mixed with dense engineered features.

**Hyperparameter tuning** (`GridSearchCV`, 3-fold, `scoring="f1_macro"`) was run on:
- Logistic Regression — `C`, `penalty`, `class_weight`
- LightGBM — `class_weight`

**Final model — Soft Voting Ensemble** (Tuned Logistic Regression + Tuned LightGBM, weights `1.0 / 2.1`):

| Metric | Score |
|---|---|
| Accuracy | 0.919 |
| **Macro F1** | **0.837** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 0.98 | 0.95 | 0.96 | 22,835 |
| 1 | 0.76 | 0.83 | 0.79 | 3,183 |
| 2 | 0.88 | 0.91 | 0.89 | 12,488 |
| 3 | 0.69 | 0.71 | 0.70 | 1,094 |

The ensemble raised recall on the two minority classes (1 and 3) relative to the LightGBM-only baseline, at a small cost to class-2 precision — a better trade-off under Macro F1.

## Error Analysis

- Class **3** remains the hardest category (lowest F1), most often confused with class **2**.
- Class **1** also overlaps with class **2**.
- Class **0** is the easiest to separate.
- 3,216 of 39,600 validation comments (~8.1%) were misclassified — reviewed individually to sanity-check whether errors were driven by genuinely ambiguous text or by feature gaps.

## Repository Structure

```
.
├── notebook.ipynb          # Full EDA → feature engineering → modeling → submission pipeline
├── train.csv                # Training data (not included — see Kaggle competition page)
├── test.csv                 # Test data
├── Sample.csv                # Submission format
├── submission.csv           # Final predictions
└── README.md
```

## How to Reproduce

```bash
git clone <repo-url>
cd comment-category-prediction
pip install pandas numpy scikit-learn lightgbm seaborn matplotlib regex
jupyter notebook notebook.ipynb
```

Update `TRAIN_PATH`, `TEST_PATH`, and `SUB_PATH` at the top of the notebook to point to your local copies of the competition data, then run all cells top to bottom. The final cell writes `submission.csv` in the Sample.csv format.

## Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` (Pipeline, ColumnTransformer, TF-IDF, GridSearchCV, VotingClassifier) · `LightGBM` · `matplotlib` / `seaborn` · `regex`

## Key Takeaways

- Hybrid word + character TF-IDF outperforms either alone on noisy, informal comment text.
- A single unified `Pipeline`/`ColumnTransformer` across text, numeric, categorical, and binary features avoids train/test preprocessing drift and keeps the whole workflow leak-safe and reproducible.
- On imbalanced multiclass problems, a linear model (Logistic Regression) and a tree ensemble (LightGBM) tend to make *different* mistakes — soft-voting between them is a cheap way to improve minority-class recall without building a heavier stacked model.
- Sparse, mostly-missing identity columns are more useful as **presence flags** than as raw categorical features.

## Future Improvements

- Try transformer-based embeddings (e.g., a lightweight sentence encoder) alongside TF-IDF for the text branch.
- Address class 3 specifically with targeted oversampling (SMOTE-variants for mixed sparse/dense features) or a focal-loss objective.
- Stack additional base learners (e.g., LinearSVC, CatBoost) instead of a two-model vote.
- Investigate post-level aggregate features further, given the confirmed 100% post_id overlap between train and test.

---

*Built as part of the Comment Category Prediction Challenge (Jan 17 – Mar 31, 2026).*
