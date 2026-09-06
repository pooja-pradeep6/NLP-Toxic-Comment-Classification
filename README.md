# NLP-Based Toxic Comment Classification with Explainability

A mini NLP project that classifies online comments as **toxic** or **clean**, and explains each prediction using **LIME** (Local Interpretable Model-agnostic Explanations) instead of treating the model as a black box.

## Problem Statement

Online platforms receive far more comments than humans can realistically moderate manually. An automated system can help by flagging potentially toxic content early, but for moderators to actually trust and act on a system's decisions, it needs to explain *why* it made a call rather than just spitting out a label. This project builds that combination: a working toxicity classifier paired with word-level explanations for every prediction.

## Objectives

- Build an NLP-based text classification model to distinguish toxic comments from clean ones
- Convert raw comment text into meaningful numerical features using TF-IDF
- Train a model that is effective and genuinely explainable, not just accurate
- Use LIME to make individual predictions transparent and verifiable
- Evaluate performance using metrics appropriate for an imbalanced classification problem
- Honestly identify the limitations of the approach

## Dataset

**Civil Comments** dataset, built by Jigsaw (a unit within Google/Alphabet) from real comments on independent news sites. It's one of the most widely used datasets in toxicity-detection research, and is hosted on Hugging Face, so it loads directly with no manual download or account needed.

**Dataset name:** Civil Comments

**Dataset source:** Jigsaw / Google (Conversation AI), hosted on Hugging Face at `google/civil_comments`

**Dataset size:** The full dataset contains approximately 1.8 million comments. For this project, a working sample of 30,000 comments was drawn for practical training and experimentation times in Google Colab.

**Important attributes/fields:**
- `text` — the raw comment text (renamed to `comment_text` in this project)
- `toxicity` — a continuous score between 0 and 1, representing the fraction of human raters who judged the comment as toxic
- The original dataset also includes more granular annotation fields (e.g. `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, `sexual_explicit`), though this project uses only the overall `toxicity` score

**Target variable:** A derived binary `label` column — `1` (Toxic) if `toxicity >= 0.5`, `0` (Clean) otherwise. This is engineered from the raw `toxicity` score rather than provided directly by the dataset.

**Number of classes/categories:** 2 (Toxic, Clean) after binarization. In its raw form, the dataset provides a continuous toxicity score rather than fixed categories.

**Limitations of the dataset:**
- Labels are derived from human annotator judgments, which are inherently subjective — annotators do not always agree on whether a comment is toxic, especially for sarcasm, political speech, or context-dependent insults
- The dataset is significantly imbalanced: only about 8% of the working sample is labeled toxic after binarization, which limits how many diverse toxic examples the model can learn from
- Comments are sourced primarily from Anglophone, Western news-site discussions, so the model may not generalize well to other cultural contexts, dialects, or languages
- Collapsing the continuous `toxicity` score into a binary label at a 0.5 threshold discards nuance present in the original annotations
- As documented in this project's Limitations section, many comments relying on coded language, dog-whistles, or implicit toxicity are inconsistently or incorrectly captured even in the original annotations, not just by this project's model

## Methodology

```
Raw Comment Text → Preprocessing → TF-IDF Vectorization → Logistic Regression → Prediction → LIME Explanation → Evaluation
```

**Preprocessing:** lowercasing, removing URLs and newlines, stripping punctuation/numbers, and removing English stopwords.

**Feature extraction:** TF-IDF with `ngram_range=(1,2)`, so the model looks at both single words and two-word phrases (e.g. "not good"), which helps capture patterns a single word alone would miss.

**Model:** Logistic Regression, trained with `class_weight='balanced'` to counter the class imbalance.

**Explainability:** LIME generates a local explanation for individual predictions by perturbing the input text and observing how the prediction shifts, then highlighting which words pushed it toward *Toxic* or *Clean*.

## Why Logistic Regression

Logistic Regression was chosen over more complex models for a few concrete reasons:

- It works well with TF-IDF's sparse, high-dimensional features without needing large amounts of data or a GPU
- It naturally outputs a probability for each class, which LIME requires to generate its explanations
- Its learned coefficients are directly interpretable, which matters for a project centered on explainability
- A simple, well-understood model that can be fully explained is preferable to a complex one that can't be justified confidently

## Results

The model achieves roughly **87% accuracy** overall. On its own this number is misleading, since about 92% of comments are clean, so a model that always predicted "Clean" would score higher on accuracy while catching zero toxic comments. Precision, Recall, and F1-score for the Toxic class are the metrics that actually matter here:

- **Clean:** Precision 0.96, Recall 0.90, F1 0.93
- **Toxic:** Precision 0.33, Recall 0.58, F1 0.42

This reflects a deliberate trade-off from using `class_weight='balanced'`: the model is more willing to flag a comment as toxic, which increases recall (catching more real toxic comments) at some cost to precision (more false alarms).

## Limitations

### 1. The model detects toxic *vocabulary*, not toxic *intent*

TF-IDF represents a comment purely as a bag of word frequencies — it has no understanding of meaning, tone, or context. It learns that certain words tend to co-occur with the "toxic" label, and flags future comments based on the presence of those same words. This works reasonably well when toxicity is expressed through explicit profanity or slurs, but it breaks down whenever toxicity is expressed *implicitly*.

**Example 1 — Indirect insult without profanity:** A political comment that harshly insulted public figures (implying cowardice and dishonesty) without using any profanity or slurs was classified as **Clean**. A human reader would likely call this toxic based on tone and intent alone, but since none of the specific words in the comment appeared frequently in the "toxic" training examples, the model had no signal to flag it.

**Example 2 — Coded hate speech:** A comment using dehumanizing language toward a religious/ethnic group, calling for discriminatory action, and referencing a well-documented antisemitic conspiracy theory was also classified as **Clean**. This is a more serious miss, since the comment is unambiguously hate speech. The likely reason is that it relied on coded phrasing and real-world references that require background knowledge to recognize as hateful — something a word-frequency model has no mechanism to capture. It can only respond to words it has already learned to associate with toxicity, not to ideology, dog-whistles, or implication.

This is the single most important limitation to understand: **the model is a pattern-matcher over vocabulary, not a reasoner about meaning.** This is a genuine, actively researched limitation of bag-of-words approaches in NLP generally, not a mistake specific to this implementation — it's part of why more advanced context-aware models (like BERT) exist.

### 2. The precision/recall trade-off from class balancing

The model was trained with `class_weight='balanced'` to counter the fact that only ~8% of comments in the dataset are toxic. This deliberately makes the model more willing to flag a comment as toxic, which increases **recall** (catching more real toxic comments — 58% of them) at the cost of **precision** (33%, meaning many flagged comments are actually clean false alarms). This is a conscious design choice, not an accident: for a moderation system, missing genuinely toxic content is generally considered worse than occasionally over-flagging a borderline comment for human review. Without this setting, the model would default toward predicting "Clean" more often, which would raise accuracy but make it far less useful at its actual job.

### 3. Binary labels lose the nuance of the original data

The Civil Comments dataset provides a continuous toxicity score between 0 and 1 (the fraction of human raters who flagged a comment). This project simplifies that into a single binary label using a 0.5 cutoff. A comment scored 0.51 and one scored 0.99 are treated identically as "toxic," even though the second is far more clearly toxic than the first. This simplification made the project more tractable, but it does discard information that the original annotators captured.

### 4. Limited training data relative to what's available

Only 30,000 of the roughly 1.8 million comments in the full Civil Comments dataset were used, to keep training and experimentation times reasonable in Google Colab. A larger training sample would likely expose the model to a wider variety of toxic language patterns and could improve its ability to generalize, though it would not fix the core vocabulary-vs-intent limitation described above.

### 5. Few toxic examples to learn from

Because toxic comments made up only ~8% of the working sample (roughly 2,300 out of 30,000), the model had comparatively few positive examples to learn the boundaries of toxic language from, compared to the much larger pool of clean examples. This likely limits how well it generalizes to less common or more subtle toxic phrasing patterns that weren't well represented in that smaller subset.

### Why this matters for evaluation and future work

None of these limitations mean the project failed — a simple, honestly-evaluated model that clearly understands and documents its own blind spots is exactly what the assignment brief asks for over a complex model that can't be explained. These limitations also directly motivate the Future Scope section below, particularly the move toward context-aware pretrained models, which are the standard research response to Limitation #1.

## Future Scope

- Train on a larger sample of the dataset
- Move to context-aware pretrained models such as BERT or DistilBERT, which are better equipped to catch coded language
- Extend to multiple languages
- Deploy as a real-time moderation aid with a human-in-the-loop review step for low-confidence predictions

## Tech Stack

- Python
- pandas, scikit-learn
- nltk (stopword removal)
- lime (explainability)
- matplotlib (visualization)
- Hugging Face `datasets` (data loading)

## How to Run

1. Open `source_code.py` (or the notebook) in Google Colab
2. Run all cells top to bottom — the dataset loads automatically from Hugging Face, no manual download needed
3. To test a custom comment, edit the `your_comment` variable in the last cell and re-run it

## Repository Structure

```
NLP-Project/
├── README.md
├── source_code.py
├── screenshots/
├── report/
└── requirements.txt
```

## Author

Pooja — BCA, Data Science
