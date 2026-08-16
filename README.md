# Hybrid Information Retrieval using LSA and BM25

**Improving a TF–IDF Search Engine with LSA, BM25, and a BM25+LSA Hybrid**

A complete Information Retrieval system built for the Cranfield test collection (1400 documents, 225 queries), starting from a classical TF–IDF Vector Space Model baseline and improving it with Latent Semantic Analysis, Okapi BM25, and a normalised linear hybrid of the two. All four systems share one preprocessing pipeline and one evaluation module, so every measured difference comes from the retrieval algorithm itself.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Headline Results](#headline-results)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Part 1 — Toy IR System (Theory)](#part-1--toy-ir-system-theory)
- [Part 2 — Building the IR System](#part-2--building-the-ir-system)
- [Part 3 — Evaluating the IR System](#part-3--evaluating-the-ir-system)
- [Part 4 — Analysis of VSM Limitations](#part-4--analysis-of-vsm-limitations)
- [Part 5 — Improving the IR System](#part-5--improving-the-ir-system)
- [File-by-File Reference](#file-by-file-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Failure Case Analysis](#failure-case-analysis)
- [Known Issues and Notes](#known-issues-and-notes)
- [Future Work](#future-work)
- [References](#references)
- [Authors](#authors)

---

## Project Overview

The task is to rank documents by how well they answer a natural-language query. The project is split into two components:

1. **Warm-up (Parts 1–4)** — build a working TF–IDF Vector Space Model search engine over Cranfield, implement six evaluation metrics from scratch, and analyse where the model breaks down.
2. **Main component (Part 5)** — diagnose the structural weaknesses of the baseline, propose improvements grounded in IR theory, tune them with parameter sweeps, and validate the improvements with paired statistical hypothesis tests.

The four retrieval systems compared:

| System | Core idea | Key parameters |
|---|---|---|
| **Baseline VSM** | TF–IDF weighting + cosine similarity | — |
| **LSA** | Truncated SVD of the TF–IDF matrix, cosine in latent space | `K = 200` |
| **BM25** | Okapi probabilistic ranking with TF saturation + length normalisation | `k1 = 1.5`, `b = 0.75` |
| **Hybrid** | Per-query min–max normalise both, then linearly combine | `α = 0.5` |

The central hypothesis: **LSA and BM25 fix *different* weaknesses of TF–IDF, so combining them should beat either alone.** The experiments confirm this.

---

## Headline Results

Four-way comparison at `k = 10` over all 225 Cranfield queries. Best per column in **bold**.

| System | P@10 | R@10 | F0.5@10 | MAP | nDCG@10 | MRR |
|---|---|---|---|---|---|---|
| Baseline TF–IDF | 0.2853 | 0.4115 | 0.2909 | 0.3122 | 0.4715 | 0.7692 |
| LSA (K = 200) | 0.3036 | 0.4351 | 0.3093 | 0.3280 | 0.4808 | 0.7432 |
| BM25 (k1 = 1.5, b = 0.75) | 0.2951 | 0.4333 | 0.3023 | 0.3390 | 0.4973 | 0.7776 |
| **Hybrid (α = 0.5)** | **0.3142** | **0.4544** | **0.3204** | **0.3578** | **0.5159** | **0.7987** |

The hybrid wins on **every single metric**.

- MAP: **0.3122 → 0.3578** (+14.6%)
- nDCG@10: **0.4715 → 0.5159** (+9.4%)

---

## Repository Structure

```
.
├── README.md
├── .gitignore
│
├── PART_1_2_3_4.pdf            # Written report / answers for Parts 1–4
├── PART_5.pdf                  # Research-paper-style report for Part 5
│
├── Part_2_3/                   # Baseline TF–IDF search engine (Parts 2 & 3)
│   ├── main.py                 # Driver — DO NOT MODIFY (assignment constraint)
│   ├── sentenceSegmentation.py # Naive / Punkt / spaCy sentence splitting
│   ├── tokenization.py         # Naive regex / Penn Treebank / spaCy tokenisers
│   ├── inflectionReduction.py  # Porter stemmer + WordNet lemmatiser
│   ├── stopwordRemoval.py      # NLTK English stopword filtering
│   ├── informationRetrieval.py # TF–IDF index build + cosine ranking
│   ├── evaluation.py           # P, R, F0.5, AP/MAP, nDCG, RR/MRR from scratch
│   ├── util.py                 # Utility slot + Q4.3 bottom-up stopword experiment
│   ├── time.py                 # Runtime breakdown of the IR pipeline
│   ├── README.txt              # Original assignment-template instructions
│   ├── cranfield/              # Dataset (docs, queries, qrels)
│   └── output/                 # Generated at runtime (git-ignored except the plot)
│       └── eval_plot.png
│
└── Part_5/                     # Improved retrieval systems
    ├── evaluation.py                       # Same evaluation module, reused
    ├── VSM_IMPLEMENTATION.py               # Baseline re-run under the Part 5 pipeline
    ├── LSA_IMPLEMENTATION.py               # LSA retriever (K = 200)
    ├── BM25_IMPLEMENTATION.py              # Okapi BM25 retriever
    ├── HYBRID_LSA+BM25_IMPLEMENTATION.py   # Final hybrid retriever (α = 0.5)
    ├── LSA_K_VALUE_FINDER.py               # Sweep over K ∈ {50…500}
    ├── HYBRID_ALPHA_VALUE_FINDER.py        # Sweep over α ∈ {0.0…1.0}
    ├── HYPOTHESIS_TESTING.py               # Paired Wilcoxon, t-test, Cohen's d
    ├── VSM_FAILURE_CASES.py                # Query-level failure analysis
    ├── cranfield/                          # Dataset (same copy, kept local for zero-config runs)
    └── Outputs/                            # Metric plots per experiment
        ├── vsm_output/vsm_output.png
        ├── lsa_output/lsa_output.png
        ├── bm25_output/bm25_output.png
        ├── hybrid_output/hybrid_output.png
        ├── lsa_K_value/lsa_k_value.png
        └── hybrid_alpha_finder/hybrid_alpha_value.png
```

---

## Dataset

The **Cranfield collection** — a classical aerodynamics test collection.

| File | Contents | Fields |
|---|---|---|
| `cran_docs.json` | 1400 documents | `id`, `title`, `author`, `bibliography`, `body` |
| `cran_queries.json` | 225 queries | `query number`, `query` |
| `cran_qrels.json` | 1837 relevance judgements | `query_num`, `position`, `id` |

**Graded relevance.** The `position` field in `cran_qrels.json` encodes the strength of the judgement:

| position | Meaning |
|---|---|
| 1 | A complete answer to the question |
| 2 | Highly relevant — its absence would have made the research impracticable |
| 3 | Useful as background or as a suggestion of method |
| 4 | Minimum interest, e.g. included for historical context |

Query–document pairs of no interest are simply excluded from the file.

**How relevance is used here:**
- For binary metrics (Precision, Recall, F0.5, AP/MAP, RR/MRR), any judged pair (positions 1–4) counts as relevant.
- For **nDCG**, the position is inverted into a gain: `gain = 5 − position`, so position 1 → gain 4 and position 4 → gain 1.

Reference: <http://ir.dcs.gla.ac.uk/resources/test_collections/cran/>

---

## Setup

**Python 3.8+** (developed and tested on Python 3.13).

```bash
pip install nltk spacy scikit-learn rank-bm25 numpy scipy matplotlib
```

Download the NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

Download the spaCy English model (imported at construction time by `sentenceSegmentation.py` and `tokenization.py`, so it is required even if you only use the Punkt/PTB paths):

```bash
python -m spacy download en_core_web_sm
```

Or, in one shot with a `requirements.txt`:

```
nltk
spacy
scikit-learn
rank-bm25
numpy
scipy
matplotlib
```

```bash
pip install -r requirements.txt
```

---

## How to Run

### Part 2 & 3 — Baseline search engine

Run from **inside `Part_2_3/`** (dataset paths are relative):

```bash
cd Part_2_3

# Full evaluation over all 225 queries
python main.py -dataset cranfield/ -out_folder output/

# Interactive single-query mode — prints the top 5 document IDs
python main.py -custom -dataset cranfield/ -out_folder output/
```

Command-line flags accepted by `main.py`:

| Flag | Default | Options |
|---|---|---|
| `-dataset` | `cranfield/` | path to the dataset folder |
| `-out_folder` | `output/` | where intermediate dumps and the plot are written |
| `-segmenter` | `punkt` | `naive`, `punkt` |
| `-tokenizer` | `ptb` | `naive`, `ptb` |
| `-custom` | off | flag — switches to interactive query mode |

Custom-query example:

```
Enter query below
Papers on Aerodynamics

Top five document IDs :
...
```

**Runtime profiling:**

```bash
python time.py
```

### Part 5 — Improved systems

Run from **inside `Part_5/`**:

```bash
cd Part_5

python VSM_IMPLEMENTATION.py              # Baseline re-run  → Outputs/vsm_output/
python LSA_IMPLEMENTATION.py              # LSA, K = 200     → Outputs/lsa_output/
python BM25_IMPLEMENTATION.py             # BM25             → Outputs/bm25_output/
python "HYBRID_LSA+BM25_IMPLEMENTATION.py"  # Hybrid, α = 0.5 → Outputs/hybrid_output/

python LSA_K_VALUE_FINDER.py              # K sweep          → Outputs/lsa_K_value/
python HYBRID_ALPHA_VALUE_FINDER.py       # α sweep          → Outputs/hybrid_alpha_finder/

python HYPOTHESIS_TESTING.py              # Paired tests, console output
python VSM_FAILURE_CASES.py               # Failure cases, console output
```

Each `*_IMPLEMENTATION.py` script prints all six metrics for `k = 1…10` and saves a metrics-vs-`k` plot.

---

## Part 1 — Toy IR System (Theory)

A hand-worked three-document IR system designed to expose **word-sense ambiguity**:

- `d1`: The star in our solar system provides heat and light.
- `d2`: That Hollywood star walked the red carpet for the movie premiere.
- `d3`: Astronomers observe distant stars and galaxies using telescopes.

Stopwords: `{the, in, our, and, that, for}`

**Worked through in `PART_1_2_3_4.pdf`:**

1. **Preprocessing and inverted index** — tokenisation, stopword removal, and the full term → document postings list. Note that `star` (d1, d2) and `stars` (d3) remain *separate* index terms without stemming — an early demonstration of vocabulary mismatch.
2. **TF–IDF term–document matrix** — full TF, DF, IDF (`log₁₀(N/df)`) and weight tables. Terms appearing in a single document get IDF 0.4771; `star`, appearing in two, gets 0.1761.
3. **Boolean retrieval** for the query *"star light"* — `star → {d1, d2}`, `light → {d1}`, so d1 satisfies both terms.
4. **Cosine similarity and ranking** for *"star light"*:

   | Rank | Document | Cosine similarity |
   |---|---|---|
   | 1 | d1 | 0.470 |
   | 2 | d2 | 0.052 |
   | 3 | d3 | 0 |

   The ranking is desirable: d1 has both terms, d2 has only the low-IDF `star`, d3 has neither.

5. **Word-sense ambiguity** for the query *"movie star"* — d2 is the correct answer, but d1 also surfaces because the astronomical `star` and the celebrity `star` are the same token to a bag-of-words model. The VSM has no mechanism to disambiguate senses, so precision degrades. This motivates the semantic methods in Part 5.

---

## Part 2 — Building the IR System

Implemented in `informationRetrieval.py`, driven by `main.py`.

### The preprocessing pipeline

```
raw text
  → sentence segmentation   (Punkt, default; naive and spaCy also implemented)
  → tokenisation            (Penn Treebank, default; naive regex and spaCy also implemented)
  → inflection reduction    (WordNet lemmatisation; Porter stemming also implemented)
  → stopword removal        (NLTK English list)
  → list of token lists
```

Every intermediate stage is dumped to `output/` as JSON so it can be inspected.

### `buildIndex(docs, docIDs)`

1. **Flatten** each document from a list of sentences into one token list.
2. **Document frequency** — count, for each term, the number of documents containing it.
3. **Inverse document frequency** — `idf(t) = log(N / df(t))`.
4. **TF–IDF vector** per document as a sparse `dict` of `term → tf × idf`, with the **L2 norm precomputed and cached** so cosine similarity is a single division at query time.

The index is stored as `{docIDs, idf, doc_vectors, doc_norms}`.

### `rank(queries)`

For each query: flatten → term frequencies → TF–IDF weights **using the document-collection IDF** → cosine similarity against every document → sort descending → return the full ranked list of document IDs.

Terms absent from the document vocabulary are silently dropped (see the OOV discussion in Part 4). Zero-norm vectors on either side short-circuit to a similarity of 0, avoiding division by zero.

---

## Part 3 — Evaluating the IR System

All six metrics are implemented from scratch in `evaluation.py` — no `sklearn.metrics`, no `pytrec_eval`. Each metric has a per-query function and a `mean*` aggregator that builds a `query_id → relevant docs` map from `cran_qrels.json` in one pass.

### Baseline metrics, k = 1 to 10 (225 queries)

| k | Precision | Recall | F0.5-score | MAP | nDCG | MRR |
|---|---|---|---|---|---|---|
| 1 | 0.6356 | 0.1086 | 0.3001 | 0.1086 | 0.5111 | 0.6356 |
| 2 | 0.5444 | 0.1801 | 0.3567 | 0.1735 | 0.4748 | 0.6844 |
| 3 | 0.4770 | 0.2254 | 0.3605 | 0.2112 | 0.4505 | 0.6978 |
| 4 | 0.4211 | 0.2580 | 0.3481 | 0.2333 | 0.4355 | 0.7067 |
| 5 | 0.3796 | 0.2851 | 0.3335 | 0.2483 | 0.4282 | 0.7129 |
| 6 | 0.3533 | 0.3149 | 0.3256 | 0.2623 | 0.4310 | 0.7181 |
| 7 | 0.3308 | 0.3396 | 0.3152 | 0.2746 | 0.4347 | 0.7212 |
| 8 | 0.3156 | 0.3640 | 0.3083 | 0.2856 | 0.4428 | 0.7212 |
| 9 | 0.2968 | 0.3807 | 0.2963 | 0.2929 | 0.4459 | 0.7227 |
| 10 | 0.2769 | 0.3931 | 0.2815 | 0.2970 | 0.4479 | 0.7227 |

**Global metrics at k = 10:** MAP = **0.2970**, MRR = **0.7227**.

Plot: `Part_2_3/output/eval_plot.png`

### Observations

- **Precision falls** monotonically (0.6356 → 0.2769): the top of the ranking is high quality, and everything below it dilutes.
- **Recall rises** (0.1086 → 0.3931): deeper cutoffs recover more relevant documents.
- **F0.5 peaks at k ≈ 2–3** (≈ 0.36). Because β = 0.5 weights precision twice as heavily as recall, the optimum sits where precision is still high and recall has just begun to climb.
- **MAP rises** with k, since more ranks means more chances to accumulate relevant hits.
- **nDCG** starts high (0.5111 at k = 1), dips to ≈ 0.43 near k = 5, and recovers to 0.4479 — a highly relevant document is usually at rank 1, then the gain-per-rank discount takes over.
- **MRR** saturates at ≈ 0.72 by k = 7: the first relevant document is typically at rank 1 or 2 (1 / 0.7227 ≈ 1.38).
- **Precision ≈ recall around k = 7** — the crossover point where relevant and irrelevant returns are balanced.

### Runtime (`time.py`)

| Stage | Time (s) |
|---|---|
| Query preprocessing (225 queries) | 2.18 |
| Document preprocessing (1400 docs) | 0.97 |
| Index build (TF–IDF + norms) | 0.06 |
| Ranking (cosine for all 225 queries) | 0.48 |
| **Total** | **3.69** |

Query preprocessing dominates (≈ 59%) because the spaCy and WordNet models are loaded and invoked per query. Ranking costs ≈ 2 ms per query. The whole baseline runs in under four seconds.

---

## Part 4 — Analysis of VSM Limitations

### Structural limitations of the Vector Space Model

1. **High computational cost** — full TF–IDF weighting and dense cosine comparison scale poorly to large collections.
2. **Word order is ignored** — bag-of-words means two sentences with the same words but different meanings are indistinguishable.
3. **No semantic understanding** — only exact term matches contribute; there is no notion of context.
4. **Synonymy (hurts recall)** — relevant documents phrased differently are missed.
5. **Polysemy (hurts precision)** — one surface form with several meanings pulls in irrelevant documents.
6. **Vocabulary mismatch and preprocessing errors** — stemming artefacts, spelling variants and parsing mistakes silently break matching.

### Concrete Cranfield examples

- **Synonymy:** queries using *airfoil* miss documents that write *aerofoil*; *viscous flow detachment* and *laminar boundary layer separation* describe the same phenomenon but share no terms.
- **Polysemy:** *flow* spans fluid flow, heat flow and mass flow — a heat-flow query drags in unrelated aerodynamic-flow documents.
- **Word-order insensitivity:** permuting the words of a query yields an identical ranking.

### The zero-result / OOV problem

If a query term never appears in the corpus, its document frequency is 0 and the IDF is undefined. This implementation **drops such terms from the query vector**, so they contribute nothing to the score.

Consequences:

- **All terms OOV** → every document scores 0 → an empty or arbitrary ranking.
- **Some terms OOV** → the system ranks on the surviving terms only, which can silently miss the actual intent.
- Common causes: spelling errors, rare technical terms, missing named entities, and word forms mangled by stemming.

Example: for *"ramjet engine combustor efficiency"*, if `ramjet` is out of vocabulary the ranking is driven by `engine`, `combustor`, `efficiency` alone and returns generic engine papers instead of ramjet-specific ones. Likewise, *"aerodynammics of slender bodies"* silently discards the misspelled token.

---

## Part 5 — Improving the IR System

Full write-up in `PART_5.pdf` (formatted as a Springer LNCS-style paper).

### Problem statement

The Part 5 baseline reaches **MAP@10 = 0.3122** and **nDCG@10 = 0.4715** — respectable aggregate numbers that hide three distinct failure modes. The two structural assumptions responsible:

1. **Linear term frequency.** Repeating a term keeps raising the score, even when the extra occurrences add no information. Long documents are unfairly favoured.
2. **Term independence.** Every term is its own orthogonal dimension, so synonyms are invisible and polysemes are conflated.

### Common preprocessing (Part 5)

All four Part 5 systems share one pipeline, so measured differences are attributable to the ranking algorithm:

```
lowercase → nltk.word_tokenize → keep alphabetic tokens only
          → drop NLTK English stopwords → Porter stemming
```

Documents are represented as `title + ' ' + body`. LSA and TF–IDF consume the tokens re-joined into strings (for `TfidfVectorizer`); BM25 consumes the token lists directly (for `BM25Okapi`). Same information, two formats.

### Method 1 — Latent Semantic Analysis

Build the TF–IDF term–document matrix and factor it with truncated SVD:

```
M ≈ U_K Σ_K V_Kᵀ
```

Documents and queries are projected into the K-dimensional latent space, L2-normalised, and compared by cosine similarity. Semantically related terms collapse onto shared latent dimensions, so a query and a document can match without sharing a single surface word. The trade-off: truncation can wash out exact keyword evidence.

Implementation: `sklearn.decomposition.TruncatedSVD`, `random_state=42`, vectors normalised before scoring.

### Method 2 — Okapi BM25

BM25 replaces linear TF with a **saturating** function and normalises by document length:

```
                       N − df(t) + 0.5        tf(t,d) · (k1 + 1)
BM25(d,q) =  Σ    log ─────────────────  ·  ────────────────────────────────
            t∈q          df(t) + 0.5        tf(t,d) + k1(1 − b + b·|d|/avgdl)
```

- `k1 = 1.5` controls TF saturation — the first few occurrences of a term carry most of the weight.
- `b = 0.75` controls length normalisation.

These are the standard literature defaults, also used by Lucene and Elasticsearch. **They were deliberately not tuned on Cranfield**, so BM25 gets no unfair advantage in the headline comparison.

Implementation: the `rank_bm25` library (`BM25Okapi`). Note BM25 returns unbounded positive scores rather than cosines in [−1, 1] — which is exactly why the hybrid needs normalisation.

### Method 3 — The BM25 + LSA hybrid

Because the two score families live on different scales, apply **per-query min–max normalisation** first:

```
s̃_q(d) = ( s_q(d) − min_d' s_q(d') ) / ( max_d' s_q(d') − min_d' s_q(d') )
```

then combine linearly:

```
s_hybrid(d,q) = α · s̃_BM25(d,q) + (1 − α) · s̃_LSA(d,q)
```

A document ranks well if *either* signal likes it, which is why the hybrid inherits both methods' strengths instead of averaging them away.

### Hyperparameter sweep — LSA dimensionality K

`LSA_K_VALUE_FINDER.py`, evaluated at k = 10. Best per column in **bold**.

| K | P@10 | R@10 | F0.5@10 | MAP | nDCG@10 | MRR |
|---|---|---|---|---|---|---|
| 50 | 0.2516 | 0.3628 | 0.2567 | 0.2443 | 0.3885 | 0.6232 |
| 100 | 0.2902 | 0.4172 | 0.2961 | 0.3061 | 0.4555 | 0.7050 |
| 150 | 0.2996 | 0.4296 | 0.3056 | 0.3259 | 0.4779 | 0.7478 |
| **200** | **0.3036** | **0.4351** | **0.3093** | **0.3280** | 0.4808 | 0.7432 |
| 300 | 0.2991 | 0.4302 | 0.3052 | 0.3226 | **0.4810** | 0.7551 |
| 500 | 0.2942 | 0.4241 | 0.3000 | 0.3180 | 0.4799 | **0.7685** |

At `K = 50` the space is over-compressed — MAP drops to 0.2443, *below* the baseline. At `K ≥ 300` the extra low-singular-value components add noise rather than structure. `K = 200` takes P, R, F0.5 and MAP outright, lands within 0.0003 of the best nDCG@10, and is used everywhere downstream.

Plot: `Part_5/Outputs/lsa_K_value/lsa_k_value.png`

### Hyperparameter sweep — hybrid mixing weight α

`HYBRID_ALPHA_VALUE_FINDER.py`, evaluated at k = 10. `α = 0.0` is pure LSA, `α = 1.0` is pure BM25.

| α | P@10 | R@10 | F0.5@10 | MAP | nDCG@10 | MRR |
|---|---|---|---|---|---|---|
| 0.0 | 0.3036 | 0.4351 | 0.3093 | 0.3280 | 0.4808 | 0.7432 |
| 0.1 | 0.3084 | 0.4428 | 0.3141 | 0.3380 | 0.4938 | 0.7662 |
| 0.2 | 0.3107 | 0.4474 | 0.3167 | 0.3446 | 0.5008 | 0.7771 |
| 0.3 | 0.3084 | 0.4453 | 0.3144 | 0.3487 | 0.5033 | 0.7870 |
| 0.4 | 0.3116 | 0.4513 | 0.3178 | 0.3543 | 0.5104 | 0.7950 |
| **0.5** | **0.3142** | **0.4544** | **0.3204** | 0.3578 | **0.5159** | **0.7987** |
| 0.6 | 0.3124 | 0.4505 | 0.3185 | **0.3583** | 0.5147 | 0.7948 |
| 0.7 | 0.3080 | 0.4456 | 0.3143 | 0.3529 | 0.5105 | 0.7934 |
| 0.8 | 0.3022 | 0.4393 | 0.3087 | 0.3467 | 0.5047 | 0.7862 |
| 0.9 | 0.2960 | 0.4308 | 0.3025 | 0.3420 | 0.4992 | 0.7781 |
| 1.0 | 0.2951 | 0.4333 | 0.3023 | 0.3390 | 0.4973 | 0.7776 |

Three things worth noting:

1. The **endpoints reproduce the standalone systems exactly** (α = 0 ≡ LSA, α = 1 ≡ BM25) — a built-in correctness check on the hybrid implementation.
2. Performance **peaks near α = 0.5**, which is the empirical signature of genuine complementarity rather than one method dominating.
3. At α = 0.5 the hybrid beats **both** constituents on every metric.

Plot: `Part_5/Outputs/hybrid_alpha_finder/hybrid_alpha_value.png`

### Hypothesis testing

`HYPOTHESIS_TESTING.py` computes **per-query** scores at k = 10 for all four systems and runs paired tests on the 225 paired observations:

- **Paired Wilcoxon signed-rank test** (primary — non-parametric, no normality assumption)
- **Paired t-test** (cross-check)
- **Cohen's d** on the paired differences (effect size)

Significance: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001.

**Each proposed method vs. the baseline** (Δ positive favours the proposed method):

| Metric | LSA Δ | LSA p | LSA d | BM25 Δ | BM25 p | BM25 d | Hybrid Δ | Hybrid p | Hybrid d |
|---|---|---|---|---|---|---|---|---|---|
| P | +0.018 | 0.003 ** | +0.20 | +0.010 | 0.190 | +0.10 | +0.029 | 4.0e−6 *** | +0.33 |
| R | +0.024 | 0.010 * | +0.18 | +0.022 | 0.040 * | +0.15 | +0.043 | 6.7e−6 *** | +0.32 |
| F0.5 | +0.018 | 0.007 ** | +0.20 | +0.011 | 0.171 | +0.12 | +0.029 | 2.4e−6 *** | +0.33 |
| AP | +0.016 | 0.125 | +0.15 | +0.027 | 0.002 ** | +0.21 | +0.046 | 5.7e−9 *** | +0.39 |
| nDCG | +0.009 | 0.317 | +0.09 | +0.026 | 0.001 ** | +0.18 | +0.044 | 4.9e−8 *** | +0.37 |
| RR | −0.026 | 0.050 | −0.12 | +0.008 | 0.606 | +0.03 | +0.030 | 0.046 * | +0.13 |

The split is clean: **LSA wins the precision family (P, R, F0.5); BM25 wins the ranking-quality family (AP, nDCG).** Neither dominates. The hybrid is significant on all six.

**Hybrid vs. each constituent:**

| Metric | vs LSA Δ | p | d | vs BM25 Δ | p | d |
|---|---|---|---|---|---|---|
| P | +0.011 | 0.035 * | +0.16 | +0.019 | 0.001 *** | +0.23 |
| R | +0.019 | 0.043 * | +0.15 | +0.021 | 0.008 ** | +0.19 |
| F0.5 | +0.011 | 0.022 * | +0.16 | +0.018 | 0.001 *** | +0.22 |
| AP | +0.030 | 1.0e−4 *** | +0.30 | +0.019 | 0.002 ** | +0.21 |
| nDCG | +0.035 | 2.3e−5 *** | +0.31 | +0.019 | 0.005 ** | +0.21 |
| RR | +0.056 | 2.1e−4 *** | +0.24 | +0.021 | 0.043 * | +0.11 |

All twelve differences are significant. Crucially, the hybrid's largest gains over LSA are exactly where LSA was weak (RR, nDCG, AP), and its largest gains over BM25 are exactly where BM25 was weak (P, R, F0.5) — the pattern you would predict if the combination really is capturing both sets of strengths.

**BM25 vs. LSA head-to-head:**

| Metric | Δ | Wilcoxon p | Cohen's d |
|---|---|---|---|
| P | −0.008 | 0.194 | −0.08 |
| R | −0.002 | 0.583 | −0.01 |
| F0.5 | −0.007 | 0.245 | −0.06 |
| AP | +0.011 | 0.527 | +0.07 |
| nDCG | +0.017 | 0.168 | +0.10 |
| RR | +0.034 | 0.086 | +0.11 |

**No metric shows a significant difference** (p > 0.085 throughout). The two methods are statistically indistinguishable overall while helping on *different* metrics — which is precisely the complementarity the hybrid exploits.

### Formal claims

1. The hybrid (α · BM25 + (1 − α) · LSA at α = 0.5, K = 200, k1 = 1.5, b = 0.75) is better than the baseline TF–IDF VSM on Cranfield at k = 10 with respect to **every one of the six metrics**, with paired Wilcoxon p < 0.05 minimum and p < 0.001 on five of six (P, R, F0.5, AP, nDCG).
2. The hybrid is better than **each constituent method** on every metric (paired Wilcoxon p < 0.05 minimum), confirming a genuine improvement rather than a lucky interpolation.
3. LSA and BM25 improve the baseline through **statistically distinguishable channels**, and are indistinguishable head-to-head (p > 0.085). This complementarity is the empirical mechanism behind the hybrid's success.

---

## File-by-File Reference

### `Part_2_3/`

| File | Purpose |
|---|---|
| `main.py` | Assignment-provided driver. Wires the pipeline together, parses CLI args, creates the output folder, dumps every intermediate stage as JSON, loops k = 1…10, prints all six metrics, and saves `eval_plot.png`. **Not to be modified** per the assignment. |
| `sentenceSegmentation.py` | Three segmenters: `naive()` splits on `. ? !` character-by-character and strips whitespace; `punkt()` wraps NLTK `sent_tokenize`; `spacySegmenter()` uses the spaCy pipeline's `doc.sents`. |
| `tokenization.py` | Three tokenisers: `naive()` splits on whitespace then applies a regex to separate words from punctuation; `pennTreeBank()` wraps NLTK's `TreebankWordTokenizer`; `spacyTokenizer()` uses spaCy tokens. |
| `inflectionReduction.py` | `porterStemmer()` and `wordnetLemmatizer()`. The `reduce()` wrapper called by `main.py` dispatches to the **lemmatiser**. |
| `stopwordRemoval.py` | `fromList()` — case-insensitive filtering against NLTK's English stopword set, preserving original token casing in the output. |
| `informationRetrieval.py` | The retrieval engine. `buildIndex()` computes DF → IDF → sparse TF–IDF vectors with cached L2 norms. `rank()` scores every document by cosine similarity and returns the full ranked ID list per query. |
| `evaluation.py` | All six metrics, per-query and averaged. Handles the qrels format (`query_num` / `id` are strings, `position` is an int) and casts everything to `int` before comparison — a mismatch here is the classic cause of all-zero metrics. |
| `util.py` | Utility slot from the template. Also holds the **Question 4.3** experiment (commented out): a corpus-driven bottom-up stopword list built by keeping terms whose relative frequency exceeds 0.002, then compared against NLTK's list for overlap and disagreement. |
| `time.py` | Standalone runtime profiler — times query preprocessing, document preprocessing, index building and ranking separately, then prints a breakdown table. |
| `README.txt` | The original assignment template instructions. |

### `Part_5/`

| File | Purpose |
|---|---|
| `evaluation.py` | The same evaluation module carried over from Part 3 (identical logic; one extra explanatory comment). Reusing it verbatim is what makes the four-way comparison fair. |
| `VSM_IMPLEMENTATION.py` | Baseline TF–IDF + cosine, re-run under the Part 5 preprocessing pipeline (Porter stemming, `title + body`). This — not the Part 3 number — is the reference point for all Part 5 comparisons. |
| `LSA_IMPLEMENTATION.py` | `TfidfVectorizer` → `TruncatedSVD(n_components=200, random_state=42)` → `normalize` → cosine via a single matrix product. |
| `BM25_IMPLEMENTATION.py` | `BM25Okapi(doc_tokens, k1=1.5, b=0.75)`, scoring all 1400 documents per query. |
| `HYBRID_LSA+BM25_IMPLEMENTATION.py` | Computes both score matrices, applies `per_query_minmax`, blends at α = 0.5, ranks, evaluates, plots. |
| `LSA_K_VALUE_FINDER.py` | Refits SVD for each `K ∈ {50, 100, 150, 200, 300, 500}`, evaluates all six metrics at k = 1…10, prints a k = 10 summary table plus the best K per metric, and plots metric-vs-K with the selected K marked. |
| `HYBRID_ALPHA_VALUE_FINDER.py` | Computes the LSA and BM25 score matrices **once**, then sweeps `α ∈ {0.0, 0.1, …, 1.0}` over the cached matrices — cheap, and it guarantees the endpoints match the standalone systems. Plots metric-vs-α. |
| `HYPOTHESIS_TESTING.py` | Rebuilds all four rankings in one process, computes per-query P/R/F/AP/nDCG/RR at k = 10, then runs `scipy.stats.wilcoxon` and `ttest_rel` plus Cohen's d for every system pair, and prints a mean-score summary table. |
| `VSM_FAILURE_CASES.py` | Ranks all 225 queries by `AP@10(hybrid) − AP@10(baseline)`, takes the top 5, and for each prints the query text, all four AP scores, the ground-truth relevant IDs, each system's top 5 marked ✓/✗, plus title-and-snippet context for the documents the hybrid found and the ones the baseline wrongly promoted. |

---

## Evaluation Metrics

All implemented from scratch in `evaluation.py`.

**Precision@k** — fraction of the top k that is relevant:

```
P@k = |{relevant} ∩ {top-k retrieved}| / k
```

**Recall@k** — fraction of all relevant documents recovered in the top k:

```
R@k = |{relevant} ∩ {top-k retrieved}| / |{relevant}|
```

**F0.5-score@k** — weighted harmonic mean with β = 0.5, weighting precision **twice as heavily** as recall:

```
F_β@k = (1 + β²) · P·R / (β²·P + R)   →   F0.5@k = 1.25 · P·R / (0.25·P + R)
```

**Average Precision@k** — the mean of P@i over the ranks i ≤ k at which a relevant document appears, divided by the total number of relevant documents. **MAP** is AP averaged over all queries.

**nDCG@k** — graded relevance, discounted by rank:

```
DCG@k  = Σ_{i=1..k} gain(i) / log₂(i + 1)
nDCG@k = DCG@k / IDCG@k
```

Cranfield positions 1–4 map to gains `5 − position`, so position 1 (most relevant) gives gain 4. IDCG is computed from the ideal descending ordering of that query's gains.

**Reciprocal Rank@k** — `1 / rank of the first relevant document`, or 0 if none appears in the top k. **MRR** averages this over all queries.

### Implementation details worth knowing

- All document and query IDs are cast to `int` before set membership tests. The qrels JSON stores `query_num` and `id` as **strings** while `position` is an **int** — a silent type mismatch here produces all-zero metrics, which is easy to misread as a broken retrieval model.
- Queries with no relevance judgements default to an empty relevant set and score 0, rather than raising.
- Precision divides strictly by `k`, following the standard definition, even when fewer than k documents are returned.
- nDCG uses a `{doc_id → gain}` dict rather than a list, so it can read graded relevance while the other metrics read the same qrels as a binary set.

---

## Failure Case Analysis

Produced by `VSM_FAILURE_CASES.py` — the five queries where the hybrid improves AP@10 most over the baseline. These are grouped into three failure modes.

### Mode 1 — Term-frequency inflation and length bias (fixed by BM25)

**Case 1 — Query 81:** *"wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel"*
Relevant: 631, 672, 799. The baseline finds only one in the top five, promoting long documents (249, 714) that simply repeat *aerofoil*, *tunnel* and *corrections* many times. BM25's saturation and length normalisation put all three relevant documents in the top three: **AP@10 0.331 → 1.000**.

**Case 2 — Query 119:** *"axisymmetric deviations from circularity"* in *"cylinders under hydrostatic pressure"*
Relevant: 897, 926. The baseline misses 897 and returns unrelated cylinder papers (789, 1133). BM25 ranks 897 first (**AP@10 = 0.833**) while **LSA fails completely (AP@10 = 0.000)** — a clear demonstration that LSA's semantic smoothing can actively hurt on some queries, and that the hybrid needs BM25 to cover those.

### Mode 2 — Synonymy and topical paraphrase (fixed by LSA)

**Case 4 — Query 25:** *"the interaction between adjacent blade rows of a supersonic cascade"*
10 relevant documents. The baseline retrieves only the literal matches (277, 214) and then drifts to documents sharing *blade* and *flow* (772, 990). LSA lifts **AP@10 0.300 → 0.716** by surfacing document 511, which describes the same physics as *double cascade rotor–nozzle interaction* — almost no lexical overlap with the query.

**Case 3 — Query 208:** *"the shape of the drag polar of a lifting spacecraft"* and reducing deceleration during re-entry
7 relevant documents; the baseline finds two, then returns documents sharing only the word *drag* (566, 1124). LSA reaches **AP@10 = 0.800** by pulling the correct re-entry topic cluster, including document 163, whose title shares essentially no query terms.

### Mode 3 — Polysemy and term overloading (fixed by LSA)

**Case 5 — Query 180:** *"how does scale height vary with altitude in an atmosphere?"*
8 relevant documents. The baseline returns document 314, where *height* means the height of roughness particles in boundary-layer theory — a completely different sense. LSA achieves **AP@10 = 0.947**, ranking all 8 relevant documents highly and correctly grouping related concepts such as air-density variation and atmospheric drag.

### Why the hybrid wins all three modes

BM25 fixes Cases 1–2; LSA fixes Cases 3–5. On Query 119 the two even disagree sharply. Because the hybrid promotes a document when *either* signal scores it well, it inherits both fixes — which is why it achieves the best MRR (0.7987), higher than BM25 alone (0.7776), despite LSA alone *lowering* MRR relative to the baseline (0.7432).

The overall MAP gain is a modest +0.046, but it is assembled from three genuinely different repairs. The baseline's problems are **structural**, not a matter of parameter tuning — which is exactly why adding information the model lacks (probabilistic term weighting, latent semantics) works where re-tuning would not.

---

## Known Issues and Notes

- **`Part_2_3/time.py` shadows the Python standard library.** A module named `time.py` sitting in the script directory can be picked up instead of the stdlib `time` module by anything imported afterwards. Renaming it to `runtime_analysis.py` is safer, and costs nothing.
- **spaCy is a hard dependency of the baseline.** `SentenceSegmentation.__init__` and `Tokenization.__init__` both call `spacy.load("en_core_web_sm")` unconditionally, even though the default pipeline uses Punkt and Penn Treebank. Either install the model, or move the `spacy.load` call into the methods that actually use it.
- **Two different "baseline" numbers appear in this repo, and both are correct.** Part 3 reports MAP@10 = 0.2970 / MRR = 0.7227; Part 5 reports MAP@10 = 0.3122 / MRR = 0.7692 for its baseline. The Part 5 baseline uses a different (documented) preprocessing pipeline — Porter stemming instead of WordNet lemmatisation, and `title + body` instead of `body` alone. All Part 5 comparisons are made against the Part 5 baseline, so they remain internally consistent.
- **The dataset is duplicated** in `Part_2_3/cranfield/` and `Part_5/cranfield/` (identical files, verified by checksum). This is intentional: every script resolves `cranfield/` relative to its own directory, so both copies keep the scripts runnable with zero configuration.
- **Determinism.** `TruncatedSVD` is seeded with `random_state=42`, so LSA and hybrid results reproduce exactly. Everything else is deterministic.
- **`output/` is regenerated on every run.** The intermediate JSON dumps total roughly 7 MB and are not committed — run `main.py` to recreate them.
- Filenames such as `HYBRID_LSA+BM25_IMPLEMENTATION.py` contain a `+`. It is fine on GitHub and on disk, but quote the name when invoking it from a shell.

---

## Future Work

- **Broader retrieval methods** — extend the hybrid framework with Explicit Semantic Analysis (ESA) or modern neural / dense retrievers. The per-query normalisation scheme generalises to combining any number of systems.
- **Query-side improvements** — spelling correction, query auto-completion, and query expansion via pseudo-relevance feedback or word embeddings (which would also address the OOV problem from Part 4).
- **Larger benchmarks** — Cranfield is small and confined to aerodynamics. Validating on TREC or MS MARCO would show whether `K = 200` and `α = 0.5` generalise or are collection-specific.
- **Efficiency** — the current implementation optimises for correctness and legibility. Inverted-index-based scoring, approximate nearest-neighbour search over the LSA space, and an optimised BM25 backend would cut latency without changing retrieval quality.

---

## References

1. Deerwester, S., Dumais, S.T., Furnas, G.W., Landauer, T.K., Harshman, R. — *Indexing by Latent Semantic Analysis.* Journal of the American Society for Information Science **41**(6), 391–407 (1990).
2. Robertson, S., Zaragoza, H. — *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval **3**(4), 333–389 (2009).
3. Robertson, S.E., Walker, S. — *Some Simple Effective Approximations to the 2-Poisson Model for Probabilistic Weighted Retrieval.* SIGIR '94, 232–241 (1994).
4. The Cranfield collection — <http://ir.dcs.gla.ac.uk/resources/test_collections/cran/>

---

## Authors

**Team OG** — Indian Institute of Technology Madras, Chennai 600036, Tamil Nadu, India
Course: **CS6370 — Natural Language Processing**

| Name | Roll No. |
|---|---|
| Pali Praveen | CS25M035 |
| Malepati Jayyanth Sai | CS25M029 |
| Bantu Vijayendra Varma | CS25M016 |
| Budda Reddy Gari Ajay Kumar Reddy | CS25M018 |

---

*Academic coursework submission. The Cranfield collection is used under its original terms for research and educational purposes.*
