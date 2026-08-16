"""
hypothesis_testing
---------------------
Paired statistical tests comparing all four retrieval systems: VSM , LSA, BM25, Hybrid.

Tests run:
  - Paired Wilcoxon signed-rank test 
  - Paired t-test 
  - Cohen's d 

Output: per-metric p-values for each system pair, plus a summary table of mean scores.
"""

import json
import os
import math
import numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon, ttest_rel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from evaluation import Evaluation


DATASET = "cranfield/"
K_LSA   = 200
BM25_K1 = 1.5
BM25_B  = 0.75
ALPHA   = 0.5
K_EVAL  = 10   # k at which per-query metrics are computed for tests


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

#  Shared TF-IDF (for VSM and LSA)
vectorizer  = TfidfVectorizer()
doc_tfidf   = vectorizer.fit_transform(doc_texts_str)
query_tfidf = vectorizer.transform(query_texts_str)

#  SYSTEM 1: VSM TF-IDF + cosine similarity
doc_tfidf_n   = normalize(doc_tfidf)
query_tfidf_n = normalize(query_tfidf)
sim_VSM  = (query_tfidf_n @ doc_tfidf_n.T).toarray()
ranked_b      = np.argsort(-sim_VSM, axis=1)
VSM_ranked = [[doc_ids[j] for j in row] for row in ranked_b]

#  SYSTEM 2: LSA
svd       = TruncatedSVD(n_components=K_LSA, random_state=42)
doc_lsa   = normalize(svd.fit_transform(doc_tfidf))
query_lsa = normalize(svd.transform(query_tfidf))
sim_lsa   = query_lsa @ doc_lsa.T
ranked_l  = np.argsort(-sim_lsa, axis=1)
lsa_ranked = [[doc_ids[j] for j in row] for row in ranked_l]

#  SYSTEM 3: BM25
bm25 = BM25Okapi(doc_tokens, k1=BM25_K1, b=BM25_B)
sim_bm25 = np.zeros((len(query_tokens), len(doc_tokens)))
for i, q_tok in enumerate(query_tokens):
    sim_bm25[i] = bm25.get_scores(q_tok)
ranked_bm  = np.argsort(-sim_bm25, axis=1)
bm25_ranked = [[doc_ids[j] for j in row] for row in ranked_bm]

# SYSTEM 4: Hybrid (per-query min-max normalize, then linear combine)
def per_query_minmax(M):
    M  = np.asarray(M, dtype=float)
    mn = M.min(axis=1, keepdims=True)
    mx = M.max(axis=1, keepdims=True)
    rng = np.where(mx > mn, mx - mn, 1.0)
    return (M - mn) / rng

bm25_n     = per_query_minmax(sim_bm25)
lsa_n      = per_query_minmax(sim_lsa)
sim_hybrid = ALPHA * bm25_n + (1 - ALPHA) * lsa_n
ranked_h   = np.argsort(-sim_hybrid, axis=1)
hybrid_ranked = [[doc_ids[j] for j in row] for row in ranked_h]


# ---------- Build per-query relevance maps from qrels ----------
true_docs_list = defaultdict(list)
true_docs_gain = defaultdict(dict)
for qrel in qrels:
    qid_  = int(qrel['query_num'])
    did_  = int(qrel['id'])
    pos_  = int(qrel['position'])
    gain_ = 5.0 - pos_ 
    true_docs_list[qid_].append(did_)
    true_docs_gain[qid_][did_] = gain_


# ---------- Compute per-query scores for each system ----------
ev = Evaluation()

def per_query_scores(ranked_lists, query_ids, k):
    P, R, F, AP, NDCG, RR = [], [], [], [], [], []
    for ranked, qid in zip(ranked_lists, query_ids):
        ranked_int = [int(d) for d in ranked]
        true_list  = true_docs_list.get(qid, [])
        true_dict  = true_docs_gain.get(qid, {})
        P.append(   ev.queryPrecision(        ranked_int, qid, true_list, k))
        R.append(   ev.queryRecall(           ranked_int, qid, true_list, k))
        F.append(   ev.queryFscore(           ranked_int, qid, true_list, k))
        AP.append(  ev.queryAveragePrecision( ranked_int, qid, true_list, k))
        NDCG.append(ev.queryNDCG(             ranked_int, qid, true_dict, k))
        RR.append(  ev.queryReciprocalRank(   ranked_int, qid, true_list, k))
    return {
        'P':    np.array(P),
        'R':    np.array(R),
        'F':    np.array(F),
        'AP':   np.array(AP),
        'nDCG': np.array(NDCG),
        'RR':   np.array(RR),
    }

print(f"\nComputing per-query scores at k={K_EVAL} ...")
scores = {
    'VSM': per_query_scores(VSM_ranked, query_ids, K_EVAL),
    'LSA':      per_query_scores(lsa_ranked,      query_ids, K_EVAL),
    'BM25':     per_query_scores(bm25_ranked,     query_ids, K_EVAL),
    'Hybrid':   per_query_scores(hybrid_ranked,   query_ids, K_EVAL),
}


# ---------- Paired hypothesis tests ----------
def compare(a, b, name):
    """Test whether system 'b' significantly differs from system 'a'."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    mean_diff = diff.mean()

    # Wilcoxon 
    if np.any(diff):
        try:
            _, w_p = wilcoxon(a, b)
        except ValueError:
            w_p = 1.0
    else:
        w_p = 1.0

    # Paired t-test 
    _, t_p = ttest_rel(b, a)

    # Effect size: Cohen's d 
    sd = diff.std(ddof=1)
    cohens_d = mean_diff / sd if sd > 1e-12 else 0.0


    print(f"  {name:8s}  mean_diff={mean_diff:+.4f}  "
          f"Wilcoxon p={w_p:.4g}  t-test p={t_p:.4g}  "
          f"Cohen's d={cohens_d:+.3f}")


def compare_systems(sys_a, sys_b):
    print(f"\n{sys_b} vs {sys_a}   (positive mean_diff => {sys_b} better)")
    print("-" * 80)
    for metric in ['P', 'R', 'F', 'AP', 'nDCG', 'RR']:
        compare(scores[sys_a][metric], scores[sys_b][metric], metric)



# Each improved method vs VSM (the main claims)
compare_systems('VSM', 'LSA')
compare_systems('VSM', 'BM25')
compare_systems('VSM', 'Hybrid')

# Hybrid vs each individual method 
compare_systems('LSA',      'Hybrid')
compare_systems('BM25',     'Hybrid')

# LSA vs BM25 
compare_systems('LSA',      'BM25')


# ---------- Summary table of mean scores ----------
print("\n" + "=" * 80)
print(f"MEAN SCORES AT k={K_EVAL}  (averaged across {len(query_ids)} queries)")
print("=" * 80)
print(f"{'System':10s}  {'P':>8s}  {'R':>8s}  {'F':>8s}  {'AP':>8s}  {'nDCG':>8s}  {'RR':>8s}")
print("-" * 80)
for name in ['VSM', 'LSA', 'BM25', 'Hybrid']:
    row = scores[name]
    print(f"{name:10s}  "
          f"{row['P'].mean():>8.4f}  "
          f"{row['R'].mean():>8.4f}  "
          f"{row['F'].mean():>8.4f}  "
          f"{row['AP'].mean():>8.4f}  "
          f"{row['nDCG'].mean():>8.4f}  "
          f"{row['RR'].mean():>8.4f}")
