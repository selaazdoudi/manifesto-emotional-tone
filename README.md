# Party Above All: The Determinants of Emotional Tone in French Campaign Manifestos

NLP analysis of discrete emotional tone in French legislative *professions de foi* from the 1981 and 1993 elections.  
Do ideology, incumbency, or local economic context shape how candidates write?

> **Main finding:** Party affiliation is the dominant driver of emotional tone. Candidates at the ideological extremes use more intense, negative language; governing parties adopt a more positive register. Local unemployment and individual incumbency have limited and heterogeneous effects.

---

## Research Questions

| # | Question | Hypothesis |
|---|----------|------------|
| RQ1 | Do candidates further from the political centre use a more emotionally intense register? | U-shape: intensity rises at both extremes |
| RQ2 | Do incumbents and governing-party candidates use more positive rhetoric? | Governing-party effect > individual incumbency effect |
| RQ3 | Does local unemployment shape the emotional tone of campaign messages? | Higher unemployment → more negative language |

---

## Repository Structure

| Notebook | What it does |
|----------|--------------|
| `0_dataset_construction.ipynb` | Corpus construction, party-label harmonisation, political classification, contextual variables (incumbency, unemployment) |
| `1_emotion_measurement.ipynb` | Three-model emotion pipeline (pyFeel, DistilBERT, mDeBERTa), convergent validity, face validity on 50 manifestos, corpus-level validation with BERTopic, model selection |
| `2_ideology_emotion.ipynb` | Descriptive patterns by political bloc, per-emotion lollipop charts, robustness checks across all three models |
| `3_economic_context.ipynb` | LDA topic modelling (K-selection via coherence + Jaccard stability), identification of unemployment-related discourse, emotion scores vs. departmental unemployment rate, interaction model by political bloc |
| `4_incumbency_and_emotionvstext.ipynb` | Incumbency coding, governing-party classification, positivity and joy comparisons, bloc × incumbency interaction plots, facial emotion extraction and face-vs-text comparison |

---

## Dataset Construction (`0_dataset_construction.ipynb`)

**Corpus** — OCR-scanned *professions de foi* from the CEVIPOF / Sciences Po Archelec archive are loaded from disk and matched to candidate metadata (name, party label, gender, age, profession, constituency). 98.25% of leaflets are successfully matched; the remaining 1.75% are dropped, leaving **6,446 candidates** across both elections (3,182 from 1981 and 5,936 raw documents from 1993 before deduplication).

**Political classification** — With 270 distinct party labels in 1981 and 307 in 1993, automated classification is not straightforward. Each label is matched to an official CDSP nuance code via dictionary lookup, followed by 639 manual corrections for unresolved cases. Regionalist and unaffiliated candidates are dropped. Nuance codes are then collapsed into five ideological positions (far-left, left, ecologist, right, far-right) and a continuous distance-from-centre score. Two analytical samples are derived: `df_full` (all five blocs) and `df_model` (left and right only, for incumbency analysis).

**Contextual variables** — Departmental unemployment rates are drawn from INSEE data. Lagged 1992 values are used for the 1993 election to capture the economic context at the time of writing. Coverage is 98.4% of the core sample. Incumbency is coded from the `titulaire-mandat-en-cours` metadata field: candidates with any active mandate (local, regional, national, or European) are coded 1. Overall, 54.6% of candidates held an office at the time of the election.

---

## Emotion Measurement (`1_emotion_measurement.ipynb`)

Emotional intensity is measured using three independent methods across Paul Ekman's six basic emotions (joy, anger, fear, sadness, disgust, surprise):

**Method 1 — pyFeel (lexicon-based)** scores each text by matching tokenised words against a 14,000-word French lexicon. Emotion scores are the proportion of matched words per category. 

**Method 2 — thomasrenault/emotion (DistilBERT)** is a transformer fine-tuned on ~200k GPT-4o-mini-annotated US political texts. Because the model is English-only, each manifesto is first translated using `facebook/nllb-200-1.3B` in 400-token chunks with 20-token overlap. The translated text is then passed through the emotion model in ~500-token chunks; the final score is the average across chunks. Eight sigmoid outputs are produced; Ekman emotions are retained for comparability.

**Method 3 — mDeBERTa (zero-shot NLI)** applies a multilingual NLI model directly to the original French text. Each chunk is scored by testing hypotheses of the form "this text expresses anger", and chunk scores are averaged. 

**Validation** — The three methods are compared on convergent validity (pairwise dominant-emotion agreement: FEEL × DistilBERT 77.5%, DistilBERT × zero-shot 33.3%, FEEL × zero-shot 11.2%), face validity (manual reading of 50 disagreement cases), and corpus-level validation. BERTopic is used to cluster 1993 unemployment-discourse texts into four rhetorical frames (conflictual, programmatic, principled, mobilising) to test whether the zero-shot model is sensitive to tone vs. topic : it is not (sadness assigned in 99.6–100% of cases across all clusters). FEEL assigns joy in only 0.8% of texts. DistilBERT is selected as the primary measure for textual emotions. 

---

## Results

### RQ1 — Ideology and Emotional Register (`2_ideology_emotion.ipynb`)

Candidates at the extremes use more intense, negative language than mainstream candidates : a U-shape consistent across all three models and both years Joy follows the opposite pattern, peaking among ecologists and the mainstream left.

### RQ2 — Incumbency, Governing Party, and Positive Rhetoric (`4_incumbency_and_emotionvstext.ipynb`)

Governing-party candidates are clearly more positive than the opposition. This is an effect about five times larger than the individual incumbency effect. The governing-party gap is strongest in 1981. 

### RQ3 — Local Economic Context and Emotional Tone (`3_economic_context.ipynb`)

LDA (K = 7) identifies three unemployment-related topics in the 1993 corpus, but they map almost entirely onto party affiliation rather than local conditions. The share of unemployment-focused manifestos is flat across unemployment terciles, and emotional scores change very little with local rates. 

### RQ4 — Are facial and textual tone aligned ?

Facial emotion detection on 3,518 candidate photographs shows that face and text signals do not align: candidates project a neutral or positive image regardless of the rhetorical tone of their written text.
---


## Data Sources

- **Archelec archive** — CEVIPOF / Sciences Po, hosted on Internet Archive. Digitised *professions de foi* from French presidential and legislative elections, 1958–2012.
- **CDSP nuance codes** — Centre de Données Socio-Politiques open dataset, used for party-label harmonisation.
- **INSEE unemployment data** — Departmental unemployment rates, lagged one year (1992 values for the 1993 election).

## Models

| Model | Reference |
|-------|-----------|
| pyFeel | Abdaoui et al. (2017), *Language Resources and Evaluation* |
| thomasrenault/emotion | Renault (2025), HuggingFace |
| facebook/nllb-200-1.3B | Meta AI (2022), HuggingFace |
| mDeBERTa-v3-base-mnli | Laurer et al. (2022) |
| facial_emotions_image_detection | dima806, HuggingFace |
