# Words in Power, Words in Opposition

NLP analysis of emotional tone in French legislative campaign leaflets (1981 & 1993).  
Do ideology, incumbency, or local economic context shape how candidates write?

## Research Questions

- Do candidates further from the political centre use a more emotionally intense 
  register in their *professions de foi*? (U-shape hypothesis)
- Do incumbents use more positive language than opposition candidates?
- Does local unemployment affect the emotional tone of campaign messages?

## Repository Structure

| Notebook | Description |
|---|---|
| `dataset_construction.ipynb` | Corpus construction, political classification, contextual variables |
| `emotion_scoring.ipynb` | Emotion measurement, model comparison, validation |
| `RQ1_ideology_emotion.ipynb` | U-shape hypothesis, OLS regressions, sensitivity analysis |
| `RQ2_incumbency_governing_party.ipynb` | Incumbency, governing party status, and positive rhetoric |



## Dataset Construction

**1. Corpus construction** — OCR-scanned *professions de foi* from the CEVIPOF / 
Sciences Po Archelec archive are loaded and matched to candidate metadata (name, 
party affiliation, constituency). 98.25% of leaflets are successfully matched; 
unmatched documents are dropped, leaving a corpus of 6,446 individual candidates.

**2. Political classification** — Candidates are classified ideologically from 
Ministry of the Interior party labels. The 1981 corpus contains 270 distinct party 
labels and the 1993 corpus 307, making direct classification impossible. A 
dictionary-based approach maps each label to a standardised *nuance* code, followed 
by manual corrections for residual unmatched candidates. Nuance codes are then 
converted into five ideological positions (far-left, left, ecologist, right, 
far-right) and a continuous distance-from-centre score.

**3. Contextual variables** — Departmental unemployment rates are merged from INSEE 
data to enable local economic context analysis. Coverage is 98.4% of the core sample.

**4. Analytical samples** — Two samples are derived:
- `df_full` — all five ideological positions, used for full-spectrum and 
  distance-from-centre analysis
- `df_model` — left and right only, used for incumbency analysis

## Emotion Measurement

Emotional intensity is measured using three independent methods:

- **[pyFeel](https://github.com/AdilZouitine/pyFeel)** — French lexicon-based scoring 
  (bag-of-words) across six dimensions: anger, fear, sadness, surprise, joy, disgust
- **[thomasrenault/emotion](https://huggingface.co/thomasrenault/emotion)** — DistilBERT 
  fine-tuned on ~200k US political texts, applied after French→English translation via 
  [facebook/nllb-200-1.3B](https://huggingface.co/facebook/nllb-200-1.3B) with 
  chunk-based translation (400 tokens/chunk, 20-token overlap)
- **[mDeBERTa zero-shot](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)** 
  — multilingual NLI model applied directly to the original French text

Three methods are used because no validated benchmark exists for this corpus. 
Convergence across methods with different failure modes is treated as a robustness 
criterion. Following a manual validation exercise, **thomasrenault** is used as the 
primary measure for RQ2 and RQ3, while all three intensity scores are used for RQ1.

## RQ1 — Ideology and Emotional Register

Candidates further from the political centre use a significantly more intense 
emotional register. The U-shape holds across all three models and both electoral 
years (1981 and 1993). On thomasrenault, far-left and far-right score 12–14 points 
higher than the mainstream left (β = +0.124 and β = +0.137, p < 0.001, R² = 0.36).

The two extremes differ in emotional profile: far-right intensity is driven more by 
fear and sadness alongside anger; far-left intensity by anger and disgust. Joy 
follows an inverted pattern — ecologists and the mainstream left score highest, 
confirming that positive affect is a feature of the centre, not the extremes.

## RQ2 — Incumbency, Governing Party, and Positive Rhetoric

Both dimensions of political power predict positivity, but the governing party 
effect is substantially stronger than the individual incumbency effect.

**RQ2a — Individual mandate**: candidates with an active mandate score 2 points 
higher on joy than challengers (β = +0.019, p < 0.001), controlling for bloc and 
year. The effect is concentrated in the far-left, where incumbents are markedly 
more positive than challengers (β = +0.068, p < 0.001) — holding office appears 
to moderate far-left rhetoric specifically. The effect is driven by 1993 and is 
not significant in 1981.

**RQ2b — Governing party**: candidates from the party in power score nearly 10 
points higher on joy than opposition candidates (β = +0.095, p < 0.001) — five 
times larger than the individual incumbency effect. The effect holds in both years 
but is stronger in 1981 (β = +0.127), when the contrast between a newly elected 
Socialist government and a frustrated opposition was at its sharpest.
