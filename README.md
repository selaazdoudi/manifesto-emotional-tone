# Words-in-Power-Words-in-Opposition
NLP analysis of emotional tone in French legislative campaign leaflets (1981 &amp; 1993) — does ideology, electoral position, or local context shape how candidates write?

## RQ1 — Ideology and Emotional Register

**Do candidates further from the political centre use a different emotional register 
in their *professions de foi*?**

I measure emotional intensity using three independent methods:

- **pyFeel** (Abdaoui et al., 2017) — a French emotion lexicon scoring each text on 
  six NRC dimensions: anger, fear, sadness, surprise, joy, and disgust
- **thomasrenault/emotion** — a DistilBERT model fine-tuned on ~200k US political texts 
  (campaign speeches, congressional speeches, tweets), applied after French→English 
  translation via Helsinki-NLP/opus-mt-fr-en
- **mDeBERTa zero-shot** — a multilingual transformer classifying emotional intensity 
  without any training on our data, used as an independent validation

I then test whether distance from the political centre predicts emotional intensity 
using OLS regression, and examine whether the relationship is U-shaped — with both 
far-left and far-right candidates writing more emotionally than mainstream candidates.
