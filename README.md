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
| `2_ideology_emotion.ipynb` | Descriptive patterns by political bloc, per-emotion lollipop charts, OLS regressions with continuous distance-from-centre, robustness checks across all three models |
| `3_economic_context.ipynb` | LDA topic modelling (K-selection via coherence + Jaccard stability), identification of unemployment-related discourse, emotion scores vs. departmental unemployment rate, interaction model by political bloc |
| `4_incumbency_and_emotionvstext.ipynb` | Incumbency coding, governing-party classification, positivity and joy regressions, bloc × incumbency interaction plots, facial emotion extraction and face-vs-text comparison |

---

## Dataset Construction (`0_dataset_construction.ipynb`)

**Corpus** — OCR-scanned *professions de foi* from the CEVIPOF / Sciences Po Archelec archive are loaded from disk and matched to candidate metadata (name, party label, gender, age, profession, constituency). 98.25% of leaflets are successfully matched; the remaining 1.75% are dropped, leaving **6,446 candidates** across both elections (3,182 from 1981 and 5,936 raw documents from 1993 before deduplication).

**Political classification** — With 270 distinct party labels in 1981 and 307 in 1993, automated classification is not straightforward. Each label is matched to an official CDSP nuance code via dictionary lookup, followed by 639 manual corrections for unresolved cases. Regionalist and unaffiliated candidates are dropped. Nuance codes are then collapsed into five ideological positions (far-left, left, ecologist, right, far-right) and a continuous distance-from-centre score. Two analytical samples are derived: `df_full` (all five blocs) and `df_model` (left and right only, for incumbency analysis).

**Contextual variables** — Departmental unemployment rates are drawn from INSEE data. Lagged 1992 values are used for the 1993 election to capture the economic context at the time of writing. Coverage is 98.4% of the core sample. Incumbency is coded from the `titulaire-mandat-en-cours` metadata field: candidates with any active mandate (local, regional, national, or European) are coded 1. Overall, 54.6% of candidates held an office at the time of the election.

---

## Emotion Measurement (`1_emotion_measurement.ipynb`)

Emotional intensity is measured using three independent methods across Paul Ekman's six basic emotions (joy, anger, fear, sadness, disgust, surprise):

**Method 1 — pyFeel (lexicon-based)** scores each text by matching tokenised words against a 14,000-word French lexicon. Emotion scores are the proportion of matched words per category. A composite intensity score (`feel_intensity_nrc`) is computed as the mean of the six emotion dimensions, excluding the broad positivity aggregate.

**Method 2 — thomasrenault/emotion (DistilBERT)** is a transformer fine-tuned on ~200k GPT-4o-mini-annotated US political texts. Because the model is English-only, each manifesto is first translated using `facebook/nllb-200-1.3B` in 400-token chunks with 20-token overlap. The translated text is then passed through the emotion model in ~500-token chunks; the final score is the average across chunks. Eight sigmoid outputs are produced; Ekman emotions are retained for comparability.

**Method 3 — mDeBERTa (zero-shot NLI)** applies a multilingual NLI model directly to the original French text. Each chunk is scored by testing hypotheses of the form "this text expresses anger", and chunk scores are averaged. No translation is required.

**Validation** — The three methods are compared on convergent validity (pairwise dominant-emotion agreement: FEEL × DistilBERT 77.5%, DistilBERT × zero-shot 33.3%, FEEL × zero-shot 11.2%), face validity (manual reading of 50 disagreement cases), and corpus-level validation. BERTopic is used to cluster 1993 unemployment-discourse texts into four rhetorical frames (conflictual, programmatic, principled, mobilising) to test whether the zero-shot model is sensitive to tone vs. topic — it is not (sadness assigned in 99.6–100% of cases across all clusters). FEEL assigns joy in only 0.8% of texts. DistilBERT is selected as the primary measure for RQ2 and RQ3; all three intensity scores are used for RQ1.

---

## Results

### RQ1 — Ideology and Emotional Register (`2_ideology_emotion.ipynb`)

A clear U-shape holds across all three models and both electoral years. On DistilBERT, far-left and far-right candidates score 12–14 points higher on intensity than the mainstream left (β = +0.124 and β = +0.137, p < 0.001, R² = 0.36). The two extremes differ in profile: far-left intensity is driven primarily by anger and disgust; far-right intensity by fear and sadness alongside anger. Joy follows an inverted pattern — ecologists and the mainstream left score highest (ecologists: 0.39), confirming that positive affect is a feature of the centre, not the extremes.

### RQ2 — Incumbency, Governing Party, and Positive Rhetoric (`4_incumbency_and_emotionvstext.ipynb`)

Both dimensions of political power predict positivity, but the governing-party effect is substantially stronger. Incumbents score higher on joy than challengers (0.43 vs. 0.39; β = +0.019, p < 0.001), but the gap is small. The effect is concentrated in the far-left, where holding office markedly moderates rhetorical intensity (β = +0.068, p < 0.001), and is driven by 1993 rather than 1981. Governing-party candidates score nearly 10 points higher on joy than opposition candidates (β = +0.095, p < 0.001) — five times the individual incumbency effect. The effect is stronger in 1981 (β = +0.127), when the newly elected Socialists contrasted sharply with a frustrated opposition.

A face-vs-text comparison using `dima806/facial_emotions_image_detection` on 3,518 recovered candidate photographs shows that facial and textual signals do not align. Facial scores are systematically lower for all negative emotions; joy is the only dimension where face and text converge (face = 0.399, text = 0.416). Candidates project a neutral or positive visual image regardless of the rhetorical tone of their written text.

### RQ3 — Local Economic Context and Emotional Tone (`3_economic_context.ipynb`)

An LDA model with K = 7 topics (selected via joint coherence–Jaccard stability criterion) identifies three unemployment-related topics in the 1993 corpus: one linking unemployment to taxation and immigration (Topic 1, cv = 0.85), one framing it in terms of labour conflict (Topic 5, cv = 0.70), and one taking a programmatic-left approach (Topic 3, cv = 0.44). However, topic structure maps almost entirely onto party affiliation rather than local economic conditions — Topic 5 is 99.1% far-left, Topic 6 is 100% mainstream right.

The share of unemployment-focused manifestos is essentially flat across unemployment terciles (35.8% low / 34.3% average / 37.0% high). Emotional scores also change very little with local unemployment rates. A regression controlling for bloc and year finds small but significant effects: higher unemployment is associated with slightly more intensity (β = +0.0018, p < 0.01) and anger (β = +0.0038, p < 0.001). The interaction model reveals an asymmetry: left-wing candidates in high-unemployment departments write more intensely and less positively (rhetoric of denunciation), while right and far-right candidates become less intense and more joyful (rhetoric of optimistic contrast). The effect is absent in 1981 and emerges only in 1993.

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

