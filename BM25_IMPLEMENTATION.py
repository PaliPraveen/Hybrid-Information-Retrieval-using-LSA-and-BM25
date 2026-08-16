import json
import os
import numpy as np
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from evaluation import Evaluation

DATASET    = "cranfield/"
OUT_FOLDER = os.path.join("Outputs", "bm25_output")
BM25_K1    = 1.5 
BM25_B     = 0.75  
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

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stop]
    return tokens

doc_tokens   = [preprocess(d) for d in doc_texts]
query_tokens = [preprocess(q) for q in query_texts]


# ----------  BM25 indexing ----------
print(f"Building BM25 index (k1={BM25_K1}, b={BM25_B}) ...")
bm25 = BM25Okapi(doc_tokens, k1=BM25_K1, b=BM25_B)

# ---------- Rank docs per query ----------
doc_IDs_ordered = []
for q_tok in query_tokens:
    scores = bm25.get_scores(q_tok)
    ranked = np.argsort(-scores)
    doc_IDs_ordered.append([doc_ids[j] for j in ranked])

# ---------- Evaluation ----------
evaluator = Evaluation()

precisions, recalls, fscores, MAPs, nDCGs, MRRs = [], [], [], [], [], []

print(f"\nBM25 Results (k1={BM25_K1}, b={BM25_B})")
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


# ---------- Plot----------
plt.figure()
plt.plot(range(1, 11), precisions, label="Precision")
plt.plot(range(1, 11), recalls,    label="Recall")
plt.plot(range(1, 11), fscores,    label="F-Score")
plt.plot(range(1, 11), MAPs,       label="MAP")
plt.plot(range(1, 11), nDCGs,      label="nDCG")
plt.plot(range(1, 11), MRRs,       label="MRR")
plt.legend()
plt.title(f"BM25 Evaluation Metrics (k1={BM25_K1}, b={BM25_B}) - Cranfield Dataset")
plt.xlabel("k")
plt.savefig(os.path.join(OUT_FOLDER, "bm25_output.png"))