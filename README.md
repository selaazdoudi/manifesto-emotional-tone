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
