"""
failure_cases
-------------------
Identifies concrete Cranfield queries where the baseline TF-IDF VSM fails,
and shows what LSA, BM25, and Hybrid each retrieve in comparison.

"""

import json
import os
import numpy as np
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from evaluation import Evaluation

DATASET     = "cranfield/"
K_LSA       = 200
BM25_K1     = 1.5
BM25_B      = 0.75
ALPHA       = 0.5
K_EVAL      = 10        # k for AP comparison
N_CASES     = 5         # how many failure cases to print
N_DISPLAY   = 5         # top-N retrieved/relevant docs to show per case
SNIPPET_LEN = 220       # chars of doc body to show for context


with open(os.path.join(DATASET, "cran_docs.json")) as f:
    docs_json = json.load(f)
with open(os.path.join(DATASET, "cran_queries.json")) as f:
    queries_json = json.load(f)
with open(os.path.join(DATASET, "cran_qrels.json")) as f:
    qrels = json.load(f)

doc_ids     = [int(d['id']) for d in docs_json]
doc_texts   = [d['title'] + ' ' + d['body'] for d in docs_json]
query_ids   = [int(q['query number']) for q in queries_json]
query_texts = [q['query'] for q in queries_json]

# Lookup: doc_id -> {title, body}
doc_lookup = {int(d['id']): d for d in docs_json}
# Lookup: query_id -> query text (raw, unprocessed)
query_lookup = {int(q['query number']): q['query'] for q in queries_json}


# ---------- Tokenization and Stopword Removal ----------
stemmer = PorterStemmer()
stop    = set(stopwords.words('english'))

def preprocess_string(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stop]
    return ' '.join(tokens)

def preprocess_tokens(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stop]
    return tokens

doc_texts_str   = [preprocess_string(d) for d in doc_texts]
query_texts_str = [preprocess_string(q) for q in query_texts]
doc_tokens      = [preprocess_tokens(d) for d in doc_texts]
query_tokens    = [preprocess_tokens(q) for q in query_texts]


# ---------- Compute rankings for all 4 systems ----------

vectorizer  = TfidfVectorizer()
doc_tfidf   = vectorizer.fit_transform(doc_texts_str)
query_tfidf = vectorizer.transform(query_texts_str)

# VSM
doc_tfidf_n   = normalize(doc_tfidf)
query_tfidf_n = normalize(query_tfidf)
sim_baseline  = (query_tfidf_n @ doc_tfidf_n.T).toarray()
baseline_ranked = [[doc_ids[j] for j in row]
                   for row in np.argsort(-sim_baseline, axis=1)]

# LSA
svd       = TruncatedSVD(n_components=K_LSA, random_state=42)
doc_lsa   = normalize(svd.fit_transform(doc_tfidf))
query_lsa = normalize(svd.transform(query_tfidf))
sim_lsa   = query_lsa @ doc_lsa.T
lsa_ranked = [[doc_ids[j] for j in row]
              for row in np.argsort(-sim_lsa, axis=1)]

# BM25
bm25 = BM25Okapi(doc_tokens, k1=BM25_K1, b=BM25_B)
sim_bm25 = np.zeros((len(query_tokens), len(doc_tokens)))
for i, q_tok in enumerate(query_tokens):
    sim_bm25[i] = bm25.get_scores(q_tok)
bm25_ranked = [[doc_ids[j] for j in row]
               for row in np.argsort(-sim_bm25, axis=1)]

# Hybrid
def per_query_minmax(M):
    M = np.asarray(M, dtype=float)
    mn = M.min(axis=1, keepdims=True)
    mx = M.max(axis=1, keepdims=True)
    rng = np.where(mx > mn, mx - mn, 1.0)
    return (M - mn) / rng
sim_hybrid = ALPHA * per_query_minmax(sim_bm25) + (1 - ALPHA) * per_query_minmax(sim_lsa)
hybrid_ranked = [[doc_ids[j] for j in row]
                 for row in np.argsort(-sim_hybrid, axis=1)]


# ---------- Build qrels lookup ----------
true_docs = defaultdict(list)
for qrel in qrels:
    qid_ = int(qrel['query_num'])
    did_ = int(qrel['id'])
    true_docs[qid_].append(did_)


# ---------- Per-query AP@k for ranking failures ----------
ev = Evaluation()

def per_query_ap(ranked, k):
    aps = []
    for r, qid in zip(ranked, query_ids):
        aps.append(ev.queryAveragePrecision(
            [int(d) for d in r], qid, true_docs.get(qid, []), k))
    return np.array(aps)

ap_baseline = per_query_ap(baseline_ranked, K_EVAL)
ap_lsa      = per_query_ap(lsa_ranked,      K_EVAL)
ap_bm25     = per_query_ap(bm25_ranked,     K_EVAL)
ap_hybrid   = per_query_ap(hybrid_ranked,   K_EVAL)


# ---------- Find Vsm failures that Hybrid fixes ----------
improvement = ap_hybrid - ap_baseline
top_failure_indices = np.argsort(-improvement)[:N_CASES]


# ---------- Helpers for nice output ----------
def doc_snippet(doc_id):
    d = doc_lookup.get(doc_id)
    if d is None:
        return f"[unknown doc {doc_id}]"
    body = d['body'].replace('\n', ' ').strip()
    if len(body) > SNIPPET_LEN:
        body = body[:SNIPPET_LEN].rstrip() + " ..."
    return f"[{doc_id}] {d['title'].strip()}  --  {body}"

def mark_relevant(retrieved, relevant_set):
    """Return list of '✓ id' or '✗ id' to highlight hits."""
    return [f"{'✓' if d in relevant_set else '✗'} {d}" for d in retrieved]


# ---------- Print failure cases ----------

for case_num, idx in enumerate(top_failure_indices, start=1):
    qid     = query_ids[idx]
    rel_set = set(true_docs.get(qid, []))

    print(f"\n\n Case {case_num}: Query {qid}")
    print(f"Query: \"{query_lookup[qid]}\"")
    print()
    print(f"  AP@{K_EVAL} scores:  "
          f"VSM={ap_baseline[idx]:.3f}  "
          f"LSA={ap_lsa[idx]:.3f}  "
          f"BM25={ap_bm25[idx]:.3f}  "
          f"Hybrid={ap_hybrid[idx]:.3f}")
    print(f"  Improvement (Hybrid - Baseline) = {improvement[idx]:+.3f}")
    print(f"  Ground-truth relevant doc IDs ({len(rel_set)}): "
          f"{sorted(rel_set)[:12]}{'...' if len(rel_set) > 12 else ''}")
    print()
    print(f"  Top-{N_DISPLAY} Baseline:  {mark_relevant(baseline_ranked[idx][:N_DISPLAY], rel_set)}")
    print(f"  Top-{N_DISPLAY} LSA:       {mark_relevant(lsa_ranked[idx][:N_DISPLAY],      rel_set)}")
    print(f"  Top-{N_DISPLAY} BM25:      {mark_relevant(bm25_ranked[idx][:N_DISPLAY],     rel_set)}")
    print(f"  Top-{N_DISPLAY} Hybrid:    {mark_relevant(hybrid_ranked[idx][:N_DISPLAY],   rel_set)}")
    print()
    print(f"  Sample relevant docs (that the Hybrid finds):")
    relevant_retrieved_by_hybrid = [d for d in hybrid_ranked[idx][:N_DISPLAY] if d in rel_set]
    for d in relevant_retrieved_by_hybrid[:3]:
        print(f"     {doc_snippet(d)}")
    print()
    print(f"  Sample irrelevant docs the Baseline returned at top:")
    irrelevant_at_top = [d for d in baseline_ranked[idx][:N_DISPLAY] if d not in rel_set]
    for d in irrelevant_at_top[:2]:
        print(f"     {doc_snippet(d)}")