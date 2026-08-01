# Detecting AI-Generated Text

**SEA 820 — NLP Final Project**
**Team:** Aditi & Victoria

A comparison of a classic machine-learning baseline (TF-IDF + linear classifiers) against a
fine-tuned Transformer (DistilBERT) for classifying English essays as **human-written (0)** or
**AI-generated (1)**, with error analysis and a discussion of the ethical risks of deploying such
a detector.

---

## Overview

The project follows four stages, one notebook each:

1. **EDA** — validate the dataset, measure class balance and text length, identify junk samples,
   and compare vocabulary between the two classes.
2. **Baseline** — TF-IDF features (1–2 grams) with Logistic Regression, Multinomial Naive Bayes,
   and a Linear SVM; the best of the three sets the score the Transformer must beat.
3. **Transformer** — fine-tune `distilbert-base-uncased` with the Hugging Face `Trainer` API,
   logging multiple hyperparameter configurations.
4. **Error analysis** — inspect actual misclassified examples from both models, and test whether
   errors correlate with text length or with the lexical cues the baseline relies on.

A key theme running through the project: both models score extremely highly, but much of that
performance appears to come from **dataset artifacts** rather than genuine stylistic
understanding. See [Key findings](#key-findings) and
[`reports/ethical_considerations.md`](reports/ethical_considerations.md).

---

## Dataset

[AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) (Kaggle) —
487,235 essays labelled `generated = 0.0` (human) or `1.0` (AI).

The CSV is **not committed to this repository** (it is ~1 GB uncompressed, and redistributing
Kaggle data is against the dataset's terms). Download it yourself:

1. Kaggle → Settings → API → create a token. This downloads `kaggle.json`.
2. In Colab, run the setup cell at the top of `notebooks/01_eda.ipynb` and upload `kaggle.json`
   when prompted. It will download and unzip `AI_Human.csv` into the session.

Alternatively, place `AI_Human.csv` in your Google Drive root and mount Drive instead.

> **Never commit `kaggle.json`.** It is listed in `.gitignore`, along with `*.csv` and `*.zip`.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_baseline_tfidf.ipynb         # TF-IDF + LR / NB / Linear SVM
│   ├── 03_transformer_distilbert.ipynb # DistilBERT fine-tuning
│   └── 04_error_analysis.ipynb         # Error analysis across both models
└── reports/
    ├── project_plan.md
    ├── ethical_considerations.md
    ├── final_report.pdf                # TODO
    └── presentation.pdf                # TODO
```

---

## Setup

Everything was developed and run in **Google Colab**. Notebook 03 requires a GPU
(*Runtime → Change runtime type → T4 GPU*); the others run fine on CPU.

To run locally instead:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Note that notebooks 02–04 read and write files under `/content/drive/MyDrive`; running locally
requires changing those paths.

---

## How to run

Run the notebooks **in order** — each depends on files written by the previous one.

| # | Notebook | Reads | Writes |
|---|----------|-------|--------|
| 01 | `01_eda.ipynb` | `AI_Human.csv` | figures only |
| 02 | `02_baseline_tfidf.ipynb` | `AI_Human.csv` | `baseline_results.json`, `baseline_test_predictions.csv`, `baseline_lr_tfidf.joblib` |
| 03 | `03_transformer_distilbert.ipynb` | `AI_Human.csv`, `baseline_results.json` | `distilbert_ai_detector/`, `distilbert_test_predictions.csv` |
| 04 | `04_error_analysis.ipynb` | both prediction CSVs | figures only |

All notebooks apply the **same filters and the same seed** (`random_state = 42`, texts under 20
words removed, duplicates dropped), so the held-out test set is identical across notebooks and
the two models are compared on exactly the same examples.

Approximate runtime on a Colab T4: notebook 02 ≈ 10 min, notebook 03 ≈ 40 min for two
fine-tuning runs.

---

## Results

Held-out test set: 97,440 essays (20% stratified split).

| Model | Accuracy | Precision (AI) | Recall (AI) | F1 (AI) | Macro-F1 |
|---|---|---|---|---|---|
| **Linear SVM + TF-IDF** | 0.9997 | 0.9995 | 0.9996 | 0.9996 | **0.9997** |
| Logistic Regression + TF-IDF | 0.9963 | 0.9965 | 0.9935 | 0.9950 | 0.9960 |
| Multinomial Naive Bayes | 0.9727 | 0.9798 | 0.9461 | 0.9626 | 0.9705 |
| DistilBERT (fine-tuned) | 0.9947 | 0.9900 | 0.9958 | 0.9929 | 0.9943 |


Because the classes are imbalanced (~63% human / 37% AI), accuracy alone is misleading — a model
that always predicted "human" would score ~63%. Precision, recall, and F1 are reported throughout
for this reason.

**The classic baseline outperformed the fine-tuned Transformer.** This is an unusual result and we
treat it as a finding rather than a failure: the dataset appears to be close to linearly separable
on surface lexical cues, which TF-IDF captures directly, while DistilBERT was trained on a 40k-row
subsample with a 256-token truncation limit.

---

## Key findings

- **Class imbalance (63% / 37%)** makes accuracy a poor headline metric; F1 is reported per class
  and macro-averaged.
- **Anonymisation placeholders leak the label.** Tokens like `teacher_name` and `school_name`
  appear only in human essays, because that half of the dataset was scrubbed before release.
  A model can learn "contains a placeholder → human" without learning anything about writing style.
- **Typos mark human text.** Misspellings and informal forms (`driveless`, `wont`) are strong human
  indicators, while AI output is nearly typo-free — meaning a careful proofreader looks more
  "AI-like" to these models than a careless writer.
- **Topic leakage.** Words like `nasa`, `venus`, and `car` act as strong "human" signals, reflecting
  which prompts happened to be more common in each class rather than any stylistic difference.
- **Truncation hurts the Transformer on long texts.** 83.8% of DistilBERT's errors fall on texts
  longer than its 256-token limit, versus 67.7% for the baseline.
- **Both models share a blind spot.** AI text that evades detection contains formulaic phrases
  ("in conclusion", "additionally") far less often than AI text that is caught — so lightly edited
  AI writing is likely to defeat this whole approach, not just one model.


---

## Ethical considerations

A full discussion is in [`reports/ethical_considerations.md`](reports/ethical_considerations.md),
covering disparate impact on non-native English writers, dataset artifacts and topic leakage,
the asymmetry between false-positive and false-negative costs in an academic-integrity setting,
and why this model should never be the sole basis for an accusation.

**In short: high benchmark accuracy here does not imply real-world reliability.** Much of the
measured performance rests on properties of this particular dataset.

---

## Team contributions

| Area | Owner |
|---|---|
| Repository setup, Colab environment, dataset access | Aditi |
| `01_eda.ipynb` — exploratory data analysis | Aditi |
| `02_baseline_tfidf.ipynb` — classic baseline | Victoria |
| `03_transformer_distilbert.ipynb` — DistilBERT fine-tuning | Victoria |
| `04_error_analysis.ipynb` — error analysis | Both |
| `reports/ethical_considerations.md` | Victoria |
| README, project plan | Aditi |
| Final report and presentation | Both |

---

## References

- Dataset: Gerami, S. *AI vs Human Text.* Kaggle.
  https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT.*
  arXiv:1910.01108
- Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). *GPT detectors are biased against
  non-native English writers.* Patterns, 4(7). — *verify citation details before submission*
- Hugging Face Transformers documentation. https://huggingface.co/docs/transformers
