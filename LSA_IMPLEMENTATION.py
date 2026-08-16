import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from evaluation import Evaluation


DATASET    = "cranfield/"
OUT_FOLDER = os.path.join("Outputs", "lsa_output")
K_LSA      = 200  

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


# ---------- Tokenization and stopword removal ----------
stemmer = PorterStemmer()
stop    = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stop]
    return ' '.join(tokens)

doc_texts   = [preprocess(d) for d in doc_texts]
query_texts = [preprocess(q) for q in query_texts]


# ---------- TF-IDF ----------
vectorizer  = TfidfVectorizer()
doc_tfidf   = vectorizer.fit_transform(doc_texts)
query_tfidf = vectorizer.transform(query_texts)


# ----------  LSA: Truncated SVD ----------
print(f"Running LSA with K = {K_LSA} ...")
svd       = TruncatedSVD(n_components=K_LSA, random_state=42)
doc_lsa   = svd.fit_transform(doc_tfidf)
query_lsa = svd.transform(query_tfidf)

doc_lsa   = normalize(doc_lsa)
query_lsa = normalize(query_lsa)


# ---------- Rank docs per query ----------
similarities    = query_lsa @ doc_lsa.T
ranked          = np.argsort(-similarities, axis=1)
doc_IDs_ordered = [[doc_ids[j] for j in row] for row in ranked]


# ---------- Evaluation ----------

evaluator = Evaluation()

precisions, recalls, fscores, MAPs, nDCGs, MRRs = [], [], [], [], [], []

print(f"\nLSA Results (K = {K_LSA})")
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


# ---------- PLot ----------

plt.figure()
plt.plot(range(1, 11), precisions, label="Precision")
plt.plot(range(1, 11), recalls,    label="Recall")
plt.plot(range(1, 11), fscores,    label="F-Score")
plt.plot(range(1, 11), MAPs,       label="MAP")
plt.plot(range(1, 11), nDCGs,      label="nDCG")
plt.plot(range(1, 11), MRRs,       label="MRR")
plt.legend()
plt.title(f"LSA Evaluation Metrics (K={K_LSA}) - Cranfield Dataset")
plt.xlabel("k")
plt.savefig(os.path.join(OUT_FOLDER, "lsa_output.png"))