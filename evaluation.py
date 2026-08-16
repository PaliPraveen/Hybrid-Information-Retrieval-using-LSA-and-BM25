import math
# from util import *

# Add your import statements here


class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """
        if k <= 0:
            return 0.0

        # 1. Get the top k retrieved document IDs
        top_k_retrieved = query_doc_IDs_ordered[:k]
        
        # If the system returned fewer documents than k, we adjust our denominator
        # or keep it as k depending on strict metric definitions. 
        # Standard P@k strictly divides by k.
        if not top_k_retrieved:
            return 0.0
            
        # 2. Convert ground truth list to a set for O(1) lookup time
        true_docs_set = set(true_doc_IDs)
        
        # 3. Count how many of the top k retrieved documents are relevant
        relevant_count = sum(1 for doc_id in top_k_retrieved if doc_id in true_docs_set)
        
        # 4. Calculate precision
        precision = relevant_count / k
        
        return float(precision)


    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries
        """
        meanPrecision = -1
        
        # 1. Edge case handling
        if not query_ids or k <= 0:
            return 0.0

        # 2. Map ground-truth relevant documents for each query.
        # Grouping by query_num allows O(1) retrieval during the loop.
        true_docs_map = {}
        for qrel in qrels:
            # Extract query ID and relevant document ID from the dictionary
            q_id = int(qrel['query_num'])
            doc_id = int(qrel['id'])
            
            if q_id not in true_docs_map:
                true_docs_map[q_id] = []
            true_docs_map[q_id].append(doc_id)

        # 3. Iterate over all provided query IDs and calculate their individual P@k
        total_precision = 0.0
        
        for idx, q_id in enumerate(query_ids):
            q_id_int = int(q_id)
            
            # Retrieve predicted ordering for this specific query
            predicted_docs = doc_IDs_ordered[idx]
            predicted_docs = [int(doc) for doc in predicted_docs]
            
            # Retrieve ground truth relevant docs (default to empty list if none exist)
            true_docs = true_docs_map.get(q_id_int, [])
            
            # Utilize the queryPrecision function to evaluate this single query
            q_prec = self.queryPrecision(predicted_docs, q_id_int, true_docs, k)
            total_precision += q_prec

        # 4. Compute the average precision across the total number of queries
        meanPrecision = total_precision / len(query_ids)

        return float(meanPrecision)

    
    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query
        """
        # Edge case: if there are no true relevant documents or k is invalid
        if not true_doc_IDs or k <= 0:
            return 0.0

        # 1. Get the top k retrieved document IDs
        top_k_retrieved = query_doc_IDs_ordered[:k]

        # 2. Convert ground truth list to a set for O(1) lookup time
        true_docs_set = set(true_doc_IDs)

        # 3. Count how many of the top k retrieved documents are relevant
        relevant_count = sum(1 for doc_id in top_k_retrieved if doc_id in true_docs_set)

        # 4. Calculate recall: fraction of total relevant documents successfully retrieved
        recall = relevant_count / len(true_doc_IDs)

        return float(recall)


    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries
        """
        # 1. Edge case handling
        if not query_ids or k <= 0:
            return 0.0

        # 2. Map ground-truth relevant documents for each query using qrels.
        # Grouping by query_num allows O(1) retrieval during the loop.
        true_docs_map = {}
        for qrel in qrels:
            q_id = int(qrel['query_num'])
            doc_id = int(qrel['id'])
            
            if q_id not in true_docs_map:
                true_docs_map[q_id] = []
            true_docs_map[q_id].append(doc_id)

        # 3. Iterate over all provided query IDs and calculate their individual Recall@k
        total_recall = 0.0
        
        for idx, q_id in enumerate(query_ids):
            q_id_int = int(q_id)
            
            # Retrieve predicted ordering for this specific query
            predicted_docs = doc_IDs_ordered[idx]
            predicted_docs = [int(doc) for doc in predicted_docs]
            
            # Retrieve ground truth relevant docs (default to empty list if none exist)
            true_docs = true_docs_map.get(q_id_int, [])
            
            # Utilize the queryRecall function to evaluate this single query
            q_rec = self.queryRecall(predicted_docs, q_id_int, true_docs, k)
            total_recall += q_rec

        # 4. Compute the average recall across the total number of queries
        meanRecall = total_recall / len(query_ids)

        return float(meanRecall)


    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of F0.5-score of the Information Retrieval System
        at a given value of k for a single query.
        
        F0.5 uses beta = 0.5, which weights precision twice as heavily as recall.
        Formula: F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
        """
        # 1. Compute Precision@k and Recall@k using your internal class methods
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

        # 2. Edge case: If both precision and recall are 0, F-score is defined as 0.0
        if precision + recall == 0.0:
            return 0.0

        # 3. Compute F0.5-score (beta = 0.5)
        beta = 0.5
        beta_sq = beta * beta  # 0.25

        # Edge case: avoid division by zero when the denominator collapses
        denominator = beta_sq * precision + recall
        if denominator == 0.0:
            return 0.0

        fscore = (1 + beta_sq) * precision * recall / denominator

        return float(fscore)


    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of F0.5-score of the Information Retrieval System
        at a given value of k, averaged over all the queries.
        """
        # 1. Edge case handling
        if not query_ids or k <= 0:
            return 0.0

        # 2. Map ground-truth relevant documents for each query using qrels.
        # Grouping by query_num allows O(1) retrieval during the loop.
        true_docs_map = {}
        for qrel in qrels:
            q_id = int(qrel['query_num'])
            doc_id = int(qrel['id'])

            if q_id not in true_docs_map:
                true_docs_map[q_id] = []
            true_docs_map[q_id].append(doc_id)

        # 3. Iterate over all provided query IDs and calculate their individual F0.5-score@k
        total_fscore = 0.0

        for idx, q_id in enumerate(query_ids):
            q_id_int = int(q_id)

            # Retrieve predicted ordering for this specific query
            predicted_docs = doc_IDs_ordered[idx]
            predicted_docs = [int(doc) for doc in predicted_docs]

            # Retrieve ground truth relevant docs (default to empty list if none exist)
            true_docs = true_docs_map.get(q_id_int, [])

            # Utilize the queryFscore function to evaluate this single query
            q_f = self.queryFscore(predicted_docs, q_id_int, true_docs, k)
            total_fscore += q_f

        # 4. Compute the average F0.5-score across the total number of queries
        meanFscore = total_fscore / len(query_ids)

        return float(meanFscore)



    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query
        """
        nDCG = -1

        # 1. Edge case handling
        if k <= 0 or not true_doc_IDs:
            return 0.0

        top_k_retrieved = query_doc_IDs_ordered[:k]
        if not top_k_retrieved:
            return 0.0

        # 2. Calculate Discounted Cumulative Gain (DCG@k)
        dcg = 0.0
        for rank_idx, doc_id in enumerate(top_k_retrieved):
            # true_doc_IDs is our dictionary mapping doc -> gain
            gain = true_doc_IDs.get(doc_id, 0.0)
            if gain > 0:
                dcg += gain / math.log2(rank_idx + 2)

        # 3. Calculate Ideal DCG (IDCG@k)
        # Sort all available ground-truth gains descending
        ideal_gains = sorted(true_doc_IDs.values(), reverse=True)
        
        idcg = 0.0
        for rank_idx, gain in enumerate(ideal_gains[:k]):
            idcg += gain / math.log2(rank_idx + 2)

        # 4. Normalize
        if idcg == 0.0:
            return 0.0

        nDCG = dcg / idcg
        return float(nDCG)


    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries
        """
        meanNDCG = -1

        if not query_ids or k <= 0:
            return 0.0

        # 1. Map each query to a dictionary of {doc_id: relevance_gain}
        true_docs_map = {}
        for qrel in qrels:
            q_id = int(qrel['query_num'])
            doc_id = int(qrel['id'])
            
            # Invert Cranfield position (1-4) into an IR gain score (4-1)
            pos = int(qrel['position'])
            gain = 5.0 - pos 
            
            if q_id not in true_docs_map:
                true_docs_map[q_id] = {}
            
            true_docs_map[q_id][doc_id] = gain

        # 2. Accumulate nDCG scores across all queries
        total_ndcg = 0.0
        for idx, q_id in enumerate(query_ids):
            q_id_int = int(q_id)
            predicted_docs = [int(doc) for doc in doc_IDs_ordered[idx]]
            
            # Extract the dictionary of true gains for this query
            query_true_gains = true_docs_map.get(q_id_int, {})
            
            # Pass the dictionary cleanly into the 'true_doc_IDs' parameter
            total_ndcg += self.queryNDCG(predicted_docs, q_id_int, query_true_gains, k)

        # 3. Compute the mean
        meanNDCG = total_ndcg / len(query_ids)
        return float(meanNDCG)


    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)
        """
        # 1. Edge case handling: invalid k or no ground-truth relevant documents
        if k <= 0 or not true_doc_IDs:
            return 0.0

        top_k_retrieved = query_doc_IDs_ordered[:k]
        if not top_k_retrieved:
            return 0.0

        # 2. Convert ground truth to a set for O(1) lookup efficiency.
        # This works perfectly whether true_doc_IDs is passed as a list or a dict.
        true_docs_set = set(true_doc_IDs)

        sum_precisions = 0.0
        relevant_hits = 0

        # 3. Single-pass loop to calculate Precision@i at each relevant rank
        for rank_idx, doc_id in enumerate(top_k_retrieved):
            if doc_id in true_docs_set:
                relevant_hits += 1
                # Calculate P@i where i is the 1-based rank (rank_idx + 1)
                precision_at_i = relevant_hits / (rank_idx + 1)
                sum_precisions += precision_at_i

        # 4. Compute Average Precision
        # Standard IR divides by the total number of relevant documents available.
        avgPrecision = sum_precisions / len(true_doc_IDs)

        return float(avgPrecision)


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries
        """
        map_score = -1

        # 1. Edge case handling
        if not query_ids or k <= 0:
            return 0.0

        # 2. Map ground-truth relevant documents for each query using qrels.
        # Grouping by query_num allows O(1) retrieval during the loop.
        true_docs_map = {}
        for qrel in qrels:
            q_id = int(qrel['query_num'])
            doc_id = int(qrel['id'])
            
            if q_id not in true_docs_map:
                true_docs_map[q_id] = []
            true_docs_map[q_id].append(doc_id)

        # 3. Iterate over all provided query IDs and calculate their individual AP@k
        total_ap = 0.0
        
        for idx, q_id in enumerate(query_ids):
            q_id_int = int(q_id)
            
            # Retrieve predicted ordering for this specific query safely cast to ints
            predicted_docs = [int(doc) for doc in doc_IDs_ordered[idx]]
            
            # Retrieve ground truth relevant docs (default to empty list if none exist)
            true_docs = true_docs_map.get(q_id_int, [])
            
            # Utilize the queryAveragePrecision function to evaluate this single query
            q_ap = self.queryAveragePrecision(predicted_docs, q_id_int, true_docs, k)
            total_ap += q_ap

        # 4. Compute the Mean Average Precision across all queries
        map_score = total_ap / len(query_ids)

        return float(map_score)


    def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of reciprocal rank for a single query

        Parameters
        ----------
        arg1 : list
            Ranked list of document IDs
        arg2 : int
            Query ID
        arg3 : list
            List of relevant document IDs
        arg4 : int
            The k value

        Returns
        -------
        float
            Reciprocal rank value
        """
        # 1. Edge case handling: invalid k or missing data
        if k <= 0 or not true_doc_IDs or not query_doc_IDs_ordered:
            return 0.0

        # 2. Extract the top k retrieved document IDs
        top_k_retrieved = query_doc_IDs_ordered[:k]

        # 3. Convert ground truth to a set for O(1) lookup efficiency.
        # This works perfectly whether true_doc_IDs is passed as a list or a dict keys view.
        true_docs_set = set(true_doc_IDs)

        # 4. Find the first relevant document
        for rank_idx, doc_id in enumerate(top_k_retrieved):
            if doc_id in true_docs_set:
                # The 1-based rank is rank_idx + 1
                reciprocalRank = 1.0 / (rank_idx + 1)
                return float(reciprocalRank)

        # 5. If no relevant document is found within top k results
        return 0.0


    def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of Mean Reciprocal Rank (MRR)
        averaged over all queries

        Parameters
        ----------
        arg1 : list
            List of ranked document lists
        arg2 : list
            Query IDs
        arg3 : list
            Relevance judgments
        arg4 : int
            The k value

        Returns
        -------
        float
            MRR value
        """
        if not query_ids or k <= 0:
            return 0.0

        # 1. Map ground-truth relevant documents for each query using qrels
        true_docs_map = {}
        for qrel in qrels:
            q_id = int(qrel['query_num'])
            doc_id = int(qrel['id'])
            
            if q_id not in true_docs_map:
                true_docs_map[q_id] = []
            true_docs_map[q_id].append(doc_id)

        # 2. Accumulate Reciprocal Rank scores across all queries
        total_rr = 0.0
        for idx, q_id in enumerate(query_ids):
            q_id_int = int(q_id)
            predicted_docs = [int(doc) for doc in doc_IDs_ordered[idx]]
            true_docs = true_docs_map.get(q_id_int, [])
            
            total_rr += self.queryReciprocalRank(predicted_docs, q_id_int, true_docs, k)

        # 3. Compute the Mean Reciprocal Rank
        mean_rr = total_rr / len(query_ids)
        return float(mean_rr)