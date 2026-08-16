import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from evaluation import Evaluation


DATASET      = "cranfield/"
OUT_FOLDER = os.path.join("Outputs", "hybrid_alpha_finder")
K_LSA        = 200
BM25_K1      = 1.5
BM25_B       = 0.75
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)


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


# ---------- LSA score matrix  ----------
print(f"Computing LSA scores (K={K_LSA}) ...")
vectorizer  = TfidfVectorizer()
doc_tfidf   = vectorizer.fit_transform(doc_texts_str)
query_tfidf = vectorizer.transform(query_texts_str)
svd         = TruncatedSVD(n_components=K_LSA, random_state=42)
doc_lsa     = normalize(svd.fit_transform(doc_tfidf))
query_lsa   = normalize(svd.transform(query_tfidf))
sim_lsa     = query_lsa @ doc_lsa.T


# ---------- BM25 score matrix ----------
print(f"Computing BM25 scores (k1={BM25_K1}, b={BM25_B}) ...")
bm25 = BM25Okapi(doc_tokens, k1=BM25_K1, b=BM25_B)
sim_bm25 = np.zeros((len(query_tokens), len(doc_tokens)))
for i, q_tok in enumerate(query_tokens):
    sim_bm25[i] = bm25.get_scores(q_tok)


# ---------- Per-query min-max normalize ----------
def per_query_minmax(M):
    M  = np.asarray(M, dtype=float)
    mn = M.min(axis=1, keepdims=True)
    mx = M.max(axis=1, keepdims=True)
    rng = np.where(mx > mn, mx - mn, 1.0)
    return (M - mn) / rng

bm25_n = per_query_minmax(sim_bm25)
lsa_n  = per_query_minmax(sim_lsa)


# ---------- Sweep alpha ----------
evaluator = Evaluation()
results_by_alpha = {}

print(f"\n Running alpha over {ALPHA_VALUES} ...")
print("=" * 80)
print(f"{'alpha':>6}  {'P@10':>8}  {'R@10':>8}  {'F@10':>8}  {'MAP@10':>8}  {'nDCG@10':>8}  {'MRR@10':>8}")
print("-" * 80)

for alpha in ALPHA_VALUES:
    # Combine scores
    sim_hybrid = alpha * bm25_n + (1 - alpha) * lsa_n
    ranked     = np.argsort(-sim_hybrid, axis=1)
    doc_IDs_ordered = [[doc_ids[j] for j in row] for row in ranked]

    # Evaluate
    P, R, F, MAP, NDCG, MRR = [], [], [], [], [], []
    for k in range(1, 11):
        P.append(   evaluator.meanPrecision(        doc_IDs_ordered, query_ids, qrels, k))
        R.append(   evaluator.meanRecall(           doc_IDs_ordered, query_ids, qrels, k))
        F.append(   evaluator.meanFscore(           doc_IDs_ordered, query_ids, qrels, k))
        MAP.append( evaluator.meanAveragePrecision( doc_IDs_ordered, query_ids, qrels, k))
        NDCG.append(evaluator.meanNDCG(             doc_IDs_ordered, query_ids, qrels, k))
        MRR.append( evaluator.meanReciprocalRank(   doc_IDs_ordered, query_ids, qrels, k))

    results_by_alpha[alpha] = {'P': P, 'R': R, 'F': F, 'MAP': MAP, 'nDCG': NDCG, 'MRR': MRR}

    print(f"{alpha:>6.2f}  {P[9]:>8.4f}  {R[9]:>8.4f}  {F[9]:>8.4f}  "
          f"{MAP[9]:>8.4f}  {NDCG[9]:>8.4f}  {MRR[9]:>8.4f}")


# ---------- Plot — each metric as a function of alpha ----------
plt.figure(figsize=(10, 6))
metrics = ['P', 'R', 'F', 'MAP', 'nDCG', 'MRR']
labels  = ['Precision@10', 'Recall@10', 'F-Score@10', 'MAP@10', 'nDCG@10', 'MRR@10']
markers = ['o', 's', '^', 'd', 'v', 'P']

for metric, label, m in zip(metrics, labels, markers):
    values = [results_by_alpha[a][metric][9] for a in ALPHA_VALUES]
    plt.plot(ALPHA_VALUES, values, marker=m, label=label)

plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.4, label='Selected α=0.5')
plt.xlabel("Mixing weight α   (0.0 = pure LSA, 1.0 = pure BM25)")
plt.ylabel("Metric value at k=10")
plt.title("Hybrid Performance vs Mixing Weight α  (Cranfield Dataset)")
plt.legend(loc='best', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(OUT_FOLDER, "hybrid_alpha_value.png")
plt.savefig(plot_path, dpi=120, bbox_inches='tight')


# ---------- Identify and report best alpha per metric ----------
print("\n" + "=" * 80)
print("BEST α PER METRIC (at k=10)")
print("=" * 80)
for metric, label in zip(metrics, labels):
    values = [(a, results_by_alpha[a][metric][9]) for a in ALPHA_VALUES]
    best_a, best_v = max(values, key=lambda x: x[1])
    print(f"  {label:15s}  best α = {best_a:.2f}  (value = {best_v:.4f})")
