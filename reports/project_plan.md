# Project Plan — Detecting AI-Generated Text

**Team:** Aditi & Victoria
**Course:** SEA 820 — NLP Final Project

## Goal

Build, evaluate, and compare a classic ML baseline and a fine-tuned Transformer
for classifying text as human-written (0) vs. AI-generated (1) on the Kaggle
*AI vs. Human Text* dataset.

## Timeline & ownership

### Week 1 — Foundations & baseline
- [x] Repo + Colab setup, shared dataset access — *Aditi*
- [x] Exploratory data analysis (`01_eda.ipynb`): class balance, length, quality
      filters, vocabulary — *Aditi*
- [x] Classic baseline (`02_baseline_tfidf.ipynb`): TF-IDF + Logistic Regression,
      Naive Bayes, Linear SVM; full metrics — *Victoria*
- [x] This project plan — *both*

### Week 2 — Transformer
- [x] Tokenization + `datasets` pipeline on the shared split — *Victoria*
- [x] Fine-tune DistilBERT with `Trainer`; log ≥2 hyperparameter configs — *Victoria*
- [x] Baseline-vs-Transformer comparison table — *both*

### Week 3 — Analysis, report, presentation
- [x] Error analysis (`04_error_analysis.ipynb`): inspect false positives /
      negatives, form hypotheses — *both*
- [x] Ethical discussion (bias against non-native writers, dataset artifacts) — *Aditi*
- [x] README and repository documentation — *Aditi*
- [x] Final report (PDF) — *both*
- [x] Presentation slides (5–7 min, PDF) — *both*
- [ ] Reproducibility fixes — *outstanding*: consistent data filenames across
      notebooks 02 and 03, notebook 02 to write `baseline_test_predictions.csv`,
      and the near-duplicate check described in the report

## Design decisions (from EDA)

- Report precision / recall / F1 (not just accuracy) due to the ~63/37 imbalance.
- Stratified 80/20 split, `random_state=42`, reused across all notebooks so both
  model families are evaluated on identical held-out data.
- Remove texts under 20 words (36 found, all AI-labelled) and exact duplicates
  (0 found).
- Watch for dataset artifacts — anonymisation tokens, topic words, and typos —
  that leak the label without reflecting writing style.

## Known limitations

- DistilBERT was fine-tuned on a 40,000-row subsample with a 256-token limit,
  both compute constraints on a free Colab GPU.
- Notebook 03 selects between hyperparameter configurations using the test set;
  a separate validation split would be correct.
- Reported scores are provisional pending the near-duplicate check.

## Contributions summary

**Aditi** — repository and environment setup, exploratory data analysis, ethical
analysis, README, final report, presentation.
**Victoria** — TF-IDF baseline, DistilBERT fine-tuning and hyperparameter
experiments.
**Both** — error analysis, results comparison.

## Deliverables

Source code (GitHub) · Final report (PDF) · Presentation slides (PDF).
