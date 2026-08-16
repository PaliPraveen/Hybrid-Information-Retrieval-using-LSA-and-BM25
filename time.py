import time
import json
import os

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval


DATASET = "cranfield/"

segmenter   = SentenceSegmentation()
tokenizer   = Tokenization()
reducer     = InflectionReduction()
stopRemover = StopwordRemoval()
retriever   = InformationRetrieval()


# ---------- Preprocessing pipeline ----------
def preprocess(text_list):
    out = []
    for text in text_list:
        seg = segmenter.punkt(text)
        tok = tokenizer.pennTreeBank(seg)
        red = reducer.reduce(tok)
        rem = stopRemover.fromList(red)
        out.append(rem)
    return out


with open(os.path.join(DATASET, "cran_docs.json"), 'r') as f:
    docs_json = json.load(f)
with open(os.path.join(DATASET, "cran_queries.json"), 'r') as f:
    queries_json = json.load(f)

doc_ids  = [item["id"]            for item in docs_json]
docs     = [item["body"]          for item in docs_json]
query_ids = [item["query number"] for item in queries_json]
queries   = [item["query"]        for item in queries_json]



t_start = time.time()

# 1. Preprocess queries
t0 = time.time()
processedQueries = preprocess(queries)
t_query_preproc = time.time() - t0

# 2. Preprocess documents
t0 = time.time()
processedDocs = preprocess(docs)
t_doc_preproc = time.time() - t0

# 3. Build the index
t0 = time.time()
retriever.buildIndex(processedDocs, doc_ids)
t_index = time.time() - t0

# 4. Rank documents for all queries
t0 = time.time()
doc_IDs_ordered = retriever.rank(processedQueries)
t_rank = time.time() - t0

t_total = time.time() - t_start


# ---------- Report ----------
print("\n" + "=" * 50)
print("Runtime breakdown")
print("=" * 50)
print(f"  Query preprocessing  : {t_query_preproc:7.2f} s")
print(f"  Doc   preprocessing  : {t_doc_preproc:7.2f} s")
print(f"  Index build          : {t_index:7.2f} s")
print(f"  Ranking (all queries): {t_rank:7.2f} s")
print(f"  ----------------------------")
print(f"  Total IR runtime     : {t_total:7.2f} s")
print("=" * 50)