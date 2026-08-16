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



DATASET    = "cranfield/"
OUT_FOLDER = os.path.join("Outputs", "hybrid_output")
K_LSA      = 200    # LSA latent dimensions
BM25_K1    = 1.5    # BM25 term-frequency saturation
BM25_B     = 0.75   # BM25 length normalization
ALPHA      = 0.5    # mixing weight: score = ALPHA*BM25 + (1-ALPHA)*LSA
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

# LSA/TF-IDF wants strings; BM25 wants token lists
doc_texts_str   = [preprocess_string(d) for d in doc_texts]
query_texts_str = [preprocess_string(q) for q in query_texts]
doc_tokens      = [preprocess_tokens(d) for d in doc_texts]
query_tokens    = [preprocess_tokens(q) for q in query_texts]


# ---------- TF-IDF for LSA ----------
vectorizer  = TfidfVectorizer()
doc_tfidf   = vectorizer.fit_transform(doc_texts_str)
query_tfidf = vectorizer.transform(query_texts_str)


# ---------- LSA scores ----------
print(f"Running LSA with K = {K_LSA} ...")
svd       = TruncatedSVD(n_components=K_LSA, random_state=42)
doc_lsa   = svd.fit_transform(doc_tfidf)
query_lsa = svd.transform(query_tfidf)
doc_lsa   = normalize(doc_lsa)
query_lsa = normalize(query_lsa)
lsa_scores = query_lsa @ doc_lsa.T   


# ---------- BM25 scores ----------
print(f"Building BM25 index (k1={BM25_K1}, b={BM25_B}) ...")
bm25 = BM25Okapi(doc_tokens, k1=BM25_K1, b=BM25_B)
bm25_scores = np.zeros((len(query_tokens), len(doc_tokens)))
for i, q_tok in enumerate(query_tokens):
    bm25_scores[i] = bm25.get_scores(q_tok)


# ---------- Per-query min-max normalize so scores are comparable ----------
def per_query_minmax(M):
    M  = np.asarray(M, dtype=float)
    mn = M.min(axis=1, keepdims=True)
    mx = M.max(axis=1, keepdims=True)
    rng = np.where(mx > mn, mx - mn, 1.0)
    return (M - mn) / rng

bm25_n = per_query_minmax(bm25_scores)
lsa_n  = per_query_minmax(lsa_scores)


# ---------- Hybrid combination + ranking ----------
print(f"Combining with alpha = {ALPHA} (alpha*BM25 + (1-alpha)*LSA) ...")
hybrid_scores   = ALPHA * bm25_n + (1 - ALPHA) * lsa_n
ranked          = np.argsort(-hybrid_scores, axis=1)
doc_IDs_ordered = [[doc_ids[j] for j in row] for row in ranked]


# ---------- Evaluation ----------
evaluator = Evaluation()

precisions, recalls, fscores, MAPs, nDCGs, MRRs = [], [], [], [], [], []

print(f"\nHybrid Results (alpha={ALPHA}, K={K_LSA}, BM25 k1={BM25_K1} b={BM25_B})")
print("=" * 70)

for k in range(1, 11):
    precision = evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
    recall    = evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
    fscore    = evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)
    print(f"Precision, Recall, F-score @ {k}: {precision}, {recall}, {fscore}")

    MAP  = evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
    nDCG = evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
    MRR  = evaluator.meanReciprocalRank(doc_IDs_ordered, query_ids, qrels, k)
    MAPs.append(MAP)
    nDCGs.append(nDCG)
    MRRs.append(MRR)
    print(f"MAP, nDCG, MRR @ {k}: {MAP}, {nDCG}, {MRR}")


# ---------- Plot ----------
plt.figure()
plt.plot(range(1, 11), precisions, label="Precision")
plt.plot(range(1, 11), recalls,    label="Recall")
plt.plot(range(1, 11), fscores,    label="F-Score")
plt.plot(range(1, 11), MAPs,       label="MAP")
plt.plot(range(1, 11), nDCGs,      label="nDCG")
plt.plot(range(1, 11), MRRs,       label="MRR")
plt.legend()
plt.title(f"Combination of LSA & BM25 (α ={ALPHA}) Evaluation Metrics - Cranfield Dataset")
plt.xlabel("k")
plt.savefig(os.path.join(OUT_FOLDER, "hybrid_output.png"))