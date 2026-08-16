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
OUT_FOLDER = os.path.join("Outputs", "lsa_K_value")
K_VALUES   = [50, 100, 150, 200, 300, 500]  

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


# ---------- Tokenization and stopword Removal ----------
stemmer = PorterStemmer()
stop    = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stop]
    return ' '.join(tokens)

doc_texts   = [preprocess(d) for d in doc_texts]
query_texts = [preprocess(q) for q in query_texts]


# ---------- TF-IDF----------
vectorizer  = TfidfVectorizer()
doc_tfidf   = vectorizer.fit_transform(doc_texts)
query_tfidf = vectorizer.transform(query_texts)


evaluator = Evaluation()
results_by_K = {}

print(f"\nRunning K over {K_VALUES} ...")
print("=" * 80)
print(f"{'K':>4}  {'P@10':>8}  {'R@10':>8}  {'F@10':>8}  {'MAP@10':>8}  {'nDCG@10':>8}  {'MRR@10':>8}")
print("-" * 80)

for K in K_VALUES:
    # Run LSA at this K
    svd       = TruncatedSVD(n_components=K, random_state=42)
    doc_lsa   = normalize(svd.fit_transform(doc_tfidf))
    query_lsa = normalize(svd.transform(query_tfidf))
    sim       = query_lsa @ doc_lsa.T
    ranked    = np.argsort(-sim, axis=1)
    doc_IDs_ordered = [[doc_ids[j] for j in row] for row in ranked]

    # Evaluate at k=1..10
    P, R, F, MAP, NDCG, MRR = [], [], [], [], [], []
    for k in range(1, 11):
        P.append(   evaluator.meanPrecision(        doc_IDs_ordered, query_ids, qrels, k))
        R.append(   evaluator.meanRecall(           doc_IDs_ordered, query_ids, qrels, k))
        F.append(   evaluator.meanFscore(           doc_IDs_ordered, query_ids, qrels, k))
        MAP.append( evaluator.meanAveragePrecision( doc_IDs_ordered, query_ids, qrels, k))
        NDCG.append(evaluator.meanNDCG(             doc_IDs_ordered, query_ids, qrels, k))
        MRR.append( evaluator.meanReciprocalRank(   doc_IDs_ordered, query_ids, qrels, k))

    results_by_K[K] = {'P': P, 'R': R, 'F': F, 'MAP': MAP, 'nDCG': NDCG, 'MRR': MRR}

    # Print row for k=10 only (compact summary)
    print(f"{K:>4}  {P[9]:>8.4f}  {R[9]:>8.4f}  {F[9]:>8.4f}  "
          f"{MAP[9]:>8.4f}  {NDCG[9]:>8.4f}  {MRR[9]:>8.4f}")


# ---------- Plot — each metric as a function of K ----------
plt.figure(figsize=(10, 6))
metrics = ['P', 'R', 'F', 'MAP', 'nDCG', 'MRR']
labels  = ['Precision@10', 'Recall@10', 'F-Score@10', 'MAP@10', 'nDCG@10', 'MRR@10']
markers = ['o', 's', '^', 'd', 'v', 'P']

for metric, label, m in zip(metrics, labels, markers):
    values = [results_by_K[K][metric][9] for K in K_VALUES]   # k=10
    plt.plot(K_VALUES, values, marker=m, label=label)

plt.axvline(x=200, color='red', linestyle='--', alpha=0.4, label='Selected K=200')
plt.xlabel("Number of LSA latent dimensions (K)")
plt.ylabel("Metric value at k=10")
plt.title("LSA Performance vs Dimensionality K  (Cranfield Dataset)")
plt.legend(loc='best', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(OUT_FOLDER, "lsa_k_value.png")
plt.savefig(plot_path, dpi=120, bbox_inches='tight')


print("\n" + "=" * 80)
print("BEST K PER METRIC (at k=10)")
print("=" * 80)
for metric, label in zip(metrics, labels):
    values = [(K, results_by_K[K][metric][9]) for K in K_VALUES]
    best_K, best_v = max(values, key=lambda x: x[1])
    print(f"  {label:15s}  best K = {best_K}  (value = {best_v:.4f})")
