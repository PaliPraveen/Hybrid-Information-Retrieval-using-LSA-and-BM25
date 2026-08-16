import json
import os
import math
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from evaluation import Evaluation

DATASET    = "cranfield/"
OUT_FOLDER = os.path.join("Outputs", "vsm_output")
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

stemmer = PorterStemmer()
stop    = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stop]
    return tokens

doc_tokens   = [preprocess(d) for d in doc_texts]
query_tokens = [preprocess(q) for q in query_texts]



num_docs = len(doc_tokens)

df = {}
for doc in doc_tokens:
    for term in set(doc):
        df[term] = df.get(term, 0) + 1

idf = {term: math.log(num_docs / count) for term, count in df.items()}

doc_vectors = []
doc_norms   = []
for doc in doc_tokens:
    tf = {}
    for term in doc:
        tf[term] = tf.get(term, 0) + 1
    vec = {}
    norm_sq = 0.0
    for term, count in tf.items():
        w = count * idf[term]
        vec[term] = w
        norm_sq += w * w
    doc_vectors.append(vec)
    doc_norms.append(math.sqrt(norm_sq))


doc_IDs_ordered = []

for query in query_tokens:
    # Query TF
    q_tf = {}
    for term in query:
        q_tf[term] = q_tf.get(term, 0) + 1

    # Query TF-IDF + norm
    q_vec = {}
    q_norm_sq = 0.0
    for term, count in q_tf.items():
        if term in idf:
            w = count * idf[term]
            q_vec[term] = w
            q_norm_sq += w * w
    q_norm = math.sqrt(q_norm_sq)

    # Cosine similarity against every doc
    scores = []
    for idx, doc_vec in enumerate(doc_vectors):
        d_norm = doc_norms[idx]
        if q_norm == 0 or d_norm == 0:
            scores.append((doc_ids[idx], 0.0))
            continue
        dot = 0.0
        for term, qw in q_vec.items():
            if term in doc_vec:
                dot += qw * doc_vec[term]
        scores.append((doc_ids[idx], dot / (q_norm * d_norm)))

    scores.sort(key=lambda x: x[1], reverse=True)
    doc_IDs_ordered.append([d for d, _ in scores])


# ------------------------------------Evaluation ----------------------------------------------------
evaluator = Evaluation()

precisions, recalls, fscores, MAPs, nDCGs, MRRs = [], [], [], [], [], []

print(f"\nBaseline TF-IDF Results")
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


# ------------------------------------------ Plots  --------------------------------------
plt.figure()
plt.plot(range(1, 11), precisions, label="Precision")
plt.plot(range(1, 11), recalls,    label="Recall")
plt.plot(range(1, 11), fscores,    label="F-Score")
plt.plot(range(1, 11), MAPs,       label="MAP")
plt.plot(range(1, 11), nDCGs,      label="nDCG")
plt.plot(range(1, 11), MRRs,       label="MRR")
plt.legend()
plt.title("VSM  Evaluation Metrics - Cranfield Dataset")
plt.xlabel("k")
plt.savefig(os.path.join(OUT_FOLDER, "vsm_output.png"))
