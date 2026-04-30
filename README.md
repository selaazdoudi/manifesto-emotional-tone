# Party Above All: The Determinants of Emotional Tone in French Campaign Manifestos

## Overview

This project studies whether ideology, incumbency, or local economic context shape the emotional tone of French legislative campaign manifestos. Using a corpus of 6,446 *professions de foi* from two elections (1981 and 1993), I build an NLP pipeline that:

- Classifies candidates ideologically from Ministry of the Interior party labels
- Measures discrete emotions via three independent NLP methods 
- Analyses emotional tone by political bloc, incumbency status, governing-party membership, and local unemployment
- Compares textual and facial emotional signals using a face detection model

**Core finding:** Party affiliation is the dominant driver of emotional tone. Candidates at the ideological extremes use more intense, negative language, while governing parties adopt a more positive register. Local unemployment and individual incumbency have limited effects.

---

## Repository Structure

```
manifesto-emotional-tone/
│
├── notebooks/
│   ├── 0_dataset_construction.ipynb          # Corpus construction, political classification, contextual variables
│   ├── 1_emotion_measurement.ipynb           # Three-model emotion pipeline, validation, model selection
│   ├── 2_ideology_emotion.ipynb              # U-shape hypothesis, descriptive patterns by bloc
│   ├── 3_economic_context.ipynb              # LDA topic modelling, unemployment × emotion analysis
│   └── 4_incumbency_and_emotionvstext.ipynb  # Incumbency, governing party, face vs. text
│
└── report
```

---

## Pipeline

```
Archelec .txt files + candidate metadata
        │
        ▼
0_dataset_construction.ipynb
        │  df_full (6,446 manifestos × political + contextual variables)
        ▼
1_emotion_measurement.ipynb
        │  df_final_emotions (+ pyFeel · DistilBERT · mDeBERTa scores)
        ▼
2_ideology_emotion.ipynb              →  RQ1: ideology × emotional intensity
3_economic_context.ipynb              →  RQ3: local unemployment × emotional tone
4_incumbency_and_emotionvstext.ipynb  →  RQ2 + RQ4: incumbency · governing party · face vs. text
```

---

## Data Sources

| Source | Description |
|--------|-------------|
| Archelec archive (CEVIPOF / Sciences Po) | Raw OCR-scanned *professions de foi*, 1981 and 1993 |
| CDSP nuance codes | Official party-label harmonisation dataset |
| INSEE unemployment data | Departmental rates, lagged one year (1992 for the 1993 election) |

**Coverage** — 98.25% of leaflets matched to candidate metadata (6,446 retained). Unemployment data covers 98.4% of the core sample. 54.6% of candidates held an active mandate at the time of the election.

> **Note:** Raw manifesto texts are not included in this repository. They can be accessed via the [Archelec archive](https://archive.org) maintained by CEVIPOF / Sciences Po.

---

## Notebooks

| Notebook | What it does |
|----------|--------------|
| `0_dataset_construction.ipynb` | Loads Archelec texts, matches metadata, harmonises 270/307 party labels via CDSP dictionary + 639 manual corrections, codes incumbency, merges INSEE unemployment |
| `1_emotion_measurement.ipynb` | Runs pyFeel, translates with `facebook/nllb-200-1.3B` then scores with DistilBERT, runs mDeBERTa zero-shot; validates with convergent validity, face validity (50 manifestos), BERTopic corpus test |
| `2_ideology_emotion.ipynb` | Descriptive patterns by bloc, per-emotion lollipop charts, robustness checks across all three models |
| `3_economic_context.ipynb` | LDA K-selection (coherence + Jaccard stability), topic labelling, emotion scores vs. local unemployment rate, bloc interaction |
| `4_incumbency_and_emotionvstext.ipynb` | Incumbency and governing-party positivity comparisons, bloc × incumbency interaction plots, face extraction, face-vs-text emotion comparison |

---

## Emotion Measurement

Three methods are used because no validated benchmark exists for this corpus. Convergence across methods with different failure modes is treated as a robustness criterion.

| Method | Model | Language | Approach |
|--------|-------|----------|----------|
| pyFeel | Abdaoui et al. (2017) | French (native) | Lexicon — 14,000-word NRC-FR, bag-of-words |
| DistilBERT | thomasrenault/emotion | English (via translation) | Fine-tuned transformer, ~200k US political texts |
| Zero-shot | mDeBERTa-v3-base-mnli | French (native) | NLI-based zero-shot classification |

Translation pipeline for DistilBERT: `facebook/nllb-200-1.3B`, 400-token chunks, 20-token overlap. Emotion scoring: ~500-token chunks, averaged across chunks.

**Validation results:**

Face validity (50 manifesto manual review) and corpus-level BERTopic testing show that DistilBERT best captures rhetorical tone. The zero-shot model conflates topic and tone (sadness assigned in 99.6–100% of cases across all BERTopic clusters). FEEL assigns joy in only 0.8% of texts. **DistilBERT is retained as the primary measure.**

---

## Key Results

**RQ1 — Ideology:** Candidates at the ideological extremes use more intense, negative language than mainstream candidates : a U-shape consistent across all three models and both years. Joy follows the opposite pattern, peaking among ecologists and the mainstream left.

**RQ2 — Incumbency and governing party:** Governing-party candidates are clearly more positive than the opposition, an effect about five times larger than the individual incumbency effect. The gap is strongest in 1981.

**RQ3 — Local economic context:** LDA identifies three unemployment-related topics in the 1993 corpus, but they map almost entirely onto party affiliation rather than local conditions. The share of unemployment-focused manifestos is flat across unemployment terciles, and emotional scores change very little with local rates.

**RQ4 — Face vs. text:** Facial emotion detection on 3,518 candidate photographs shows that face and text signals do not align: candidates project a neutral or positive image regardless of the rhetorical tone of their written text.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/selaazdoudi/manifesto-emotional-tone.git
cd manifesto-emotional-tone
```

### 2. Install dependencies
Python 3.9+ required.
```bash
pip install pandas numpy matplotlib seaborn transformers torch tqdm pyFeel gensim scikit-learn statsmodels
```

### 3. Run the notebooks in order

Run notebooks `0` through `4` sequentially. Note that notebook `1` runs the translation and emotion scoring pipeline, which is computationally intensive. 

---

## Limitations

- **Translation noise:** the DistilBERT model is trained on English and applied to French texts via machine translation, which may introduce errors.
- **512-token limit:** DistilBERT and mDeBERTa process text in chunks; very long manifestos are averaged across chunks rather than read holistically.
- **Incumbency measure:** the incumbency variable captures any active mandate (local to national), which does not distinguish between the strategic advantages of different office types.
- **Face data availability:** 3,518 faces were recovered out of 6,446 manifestos due to broken image URLs; the face sample may not be fully representative.
