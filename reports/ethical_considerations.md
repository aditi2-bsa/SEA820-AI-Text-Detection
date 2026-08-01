# Ethical Considerations

**SEA 820 Final Project: Detecting AI-Generated Text**
Team: Aditi & Victoria

This section discusses the societal impact, potential biases, and ethical
risks of building an AI-text detector — grounded specifically in what our own
EDA, baseline model, and error analysis found, not in generic claims about
detectors in the abstract.

## 1. Who benefits, and who could be harmed

**Potential beneficiaries** include educators and academic-integrity offices
looking for a first-pass screening signal, platforms doing large-scale content
moderation, and researchers/policymakers studying the spread of
AI-generated misinformation. A tool like this can flag content for human
review far faster than manual inspection.

**Who could be harmed**, based specifically on what we found:

- **Non-native English speakers and ESL writers.** Our baseline's most
  predictive features for "AI" were formal transitional phrases — *in
  conclusion, additionally, however, essay, this essay* — which are exactly
  the kind of structured, formal phrasing taught explicitly in ESL writing
  curricula. A careful non-native speaker who follows the transitional
  structure they were taught could be flagged as AI for writing "correctly."
  This is not a hypothetical concern: Liang et al. (2023) found that widely
  used GPT detectors misclassified a majority of TOEFL essays written by
  non-native English speakers as AI-generated, while classifying essays by
  native speakers correctly.
- **Careful, "polished" writers generally.** Our EDA (`01_eda.ipynb`) found
  that typos, informal spelling, and irregular phrasing were markers the
  models associate with "human," while AI text is nearly typo-free. This means
  a *meticulous* human writer — someone who proofreads carefully — looks more
  "AI-like" to our models than a careless one. The detector risks penalizing
  good writing.
- **Writers of unusually long, detailed work.** Our error analysis
  (`04_error_analysis.ipynb`) found the transformer's errors skew toward
  longer texts: 83.8% of its misclassifications occurred on texts over 256
  words, versus 67.7% for the baseline. DistilBERT's limit is 256 *tokens*,
  which corresponds to roughly 190 words after sub-word tokenization, so the
  true share of its errors on truncated text is higher still. Students who
  write longer, more thorough responses are disproportionately exposed to
  truncation-driven errors.
- **Anyone accused based on a false positive.** In an academic-integrity
  context, a false accusation of AI use can carry serious consequences
  (failing grades, disciplinary hearings, reputational harm) that are not
  symmetric with the cost of a false negative (an AI-written submission going
  undetected). This asymmetry means precision on human-authored text should be
  weighted more heavily than raw accuracy suggests.

## 2. Biases we found in the dataset and models

- **Dataset artifact, not writing style: anonymization placeholders.** Our
  EDA found tokens like `teacher_name`, `school_name`, and `proper_name`
  appear *only* in the human-written class, because those essays were
  anonymized before being released and the AI-generated samples were not.
  This gives a model a shortcut — "contains a placeholder → human" — that has
  nothing to do with how humans actually write. A model exploiting this
  shortcut would perform far worse on real-world text that doesn't share this
  dataset's specific anonymization convention (e.g., real student submissions
  that still contain real names).
- **Topic leakage.** Both the EDA and our baseline's feature importances
  turned up topic-specific words (e.g., `nasa`, `venus`, `driving`, `car`) as
  strong "human" indicators. This reflects which essay prompts happened to be
  more common in one class than the other in this specific dataset — not a
  genuine stylistic difference between human and AI writing. A detector
  trained on this signal will not generalize to different topics or writing
  contexts.
- **Truncation bias against longer human writing.** The EDA flagged that
  DistilBERT's context limit disproportionately affects human texts, since
  human writing in this dataset has higher length variance (std 187 words)
  than AI writing (std 117 words). Our error analysis confirmed this
  materializes as a real, measurable effect on the transformer's errors.
- **A shared reliance on surface phrasing.** AI text that DistilBERT
  classified correctly contains formulaic "AI-tell" phrases 88.7% of the time,
  but AI text it missed contains them only 64.5% of the time — the failures
  cluster on generated text that does not read like templated AI writing. The
  baseline shows the same direction, though its 14 false negatives are too few
  a sample to support a percentage. Since both model families lean on the same
  lexical signal identified in our feature-importance analysis, they are not
  independently cross-checking each other. AI-generated text that avoids
  formulaic phrasing — through better prompting or light editing — is likely
  to evade this entire approach rather than just one model.

## 3. Data provenance and label quality

Two further concerns arise from the corpus itself rather than from the models.

The human half consists of real student essays. Whether those writers consented
to their work being redistributed as machine-learning training data is not
documented in the dataset description, and the anonymization applied is
shallow — placeholder substitution rather than any formal privacy guarantee.

Our error analysis also surfaced systematic character corruption in the source
text ("TLE work" for "the work", "iy Roiert Duffer" for "by Robert Duffer"),
which appears to be an encoding or scraping artifact upstream of dataset
assembly. Some of what we labelled and measured is therefore not text any human
or model actually produced in that form, which places a floor on how much of
our reported performance reflects genuine detection skill.

## 4. Recommendations for responsible use

- **Do not use as the sole basis for a punitive decision.** Given the
  precision/recall trade-offs and the specific artifact-driven shortcuts
  identified above, any deployment should treat the model's output as a
  screening signal that triggers human review, not as a determination on its
  own.
- **Audit before deploying outside this dataset's distribution.** Because
  several of the strongest features we found are dataset-specific artifacts
  (anonymization tokens) or dataset-specific topic correlations, performance
  on this benchmark should not be assumed to transfer to other genres,
  languages, or writing contexts.
- **Be aware of disparate impact on ESL writers.** Institutions considering
  this kind of tool should weigh the documented risk of flagging
  non-native-English writing styles, and should have an appeals or review
  process that accounts for this.
- **Address the shared blind spot, don't just add more models.** Because
  our two very different model families failed on the same kind of text,
  simply ensembling more classifiers is unlikely to close this gap on its
  own; addressing it would require training signal that doesn't rely on
  surface phrasing (e.g., more diverse AI-generation styles/prompts in
  training data).

## Reference

Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). GPT detectors
are biased against non-native English writers. *Patterns*, 4(7), 100779.
