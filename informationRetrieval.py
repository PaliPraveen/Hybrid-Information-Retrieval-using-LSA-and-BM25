import math
#from util import *

# Add your import statements here

class InformationRetrieval():

    def __init__(self):
        self.index = None

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            A document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        # 1. Flatten documents: combine sentences into a single list of tokens per doc
        print("Flattening documents...")
        flat_docs = []
        for doc in docs:
            flat_doc = [token for sentence in doc for token in sentence]
            flat_docs.append(flat_doc)

        num_docs = len(flat_docs)
        
        # 2. Compute Document Frequency (DF) for each unique term
        df = {}
        for doc in flat_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1

        # 3. Compute Inverse Document Frequency (IDF)
        idf = {}
        for term, count in df.items():
            # Standard IDF formulation
            idf[term] = math.log(num_docs / count)

        # 4. Compute TF-IDF vectors and norms for each document
        doc_vectors = []
        doc_norms = []
        
        for doc in flat_docs:
            # Term Frequency (raw count)
            tf = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            
            # TF-IDF vector for the document
            tfidf_vec = {}
            norm_sq = 0.0
            for term, count in tf.items():
                weight = count * idf[term]
                tfidf_vec[term] = weight
                norm_sq += weight ** 2
                
            doc_vectors.append(tfidf_vec)
            doc_norms.append(math.sqrt(norm_sq))

        # Store everything needed for ranking in self.index
        self.index = {
            'docIDs': docIDs,
            'idf': idf,
            'doc_vectors': doc_vectors,
            'doc_norms': doc_norms
        }


    def rank(self, queries):
        """
        Rank the documents according to relevance for each query
        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            Each sub-sub-list is a sentence of the query 
        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            Of documents in their predicted order of relevance to the ith query
        """
        doc_IDs_ordered = []
        
        # Retrieve indexed data
        docIDs = self.index['docIDs']
        idf = self.index['idf']
        doc_vectors = self.index['doc_vectors']
        doc_norms = self.index['doc_norms']

        for query in queries:
            # 1. Flatten query sentences into a single list of tokens
            flat_query = [token for sentence in query for token in sentence]
            
            # 2. Compute Query TF
            query_tf = {}
            for term in flat_query:
                query_tf[term] = query_tf.get(term, 0) + 1
                
            # 3. Compute Query TF-IDF vector and its norm
            query_vec = {}
            q_norm_sq = 0.0
            for term, count in query_tf.items():
                # Only consider terms that exist in our document index vocabulary
                if term in idf:
                    weight = count * idf[term]
                    query_vec[term] = weight
                    q_norm_sq += weight ** 2
            
            q_norm = math.sqrt(q_norm_sq)
            
            # 4. Compute Cosine Similarity scores against all documents
            scores = []
            for idx, doc_vec in enumerate(doc_vectors):
                d_norm = doc_norms[idx]
                
                # If either query or doc vector is empty/zero-norm, similarity is 0
                if q_norm == 0 or d_norm == 0:
                    scores.append((docIDs[idx], 0.0))
                    continue
                
                # Compute dot product
                dot_product = 0.0
                for term, q_weight in query_vec.items():
                    if term in doc_vec:
                        dot_product += q_weight * doc_vec[term]
                        
                cosine_sim = dot_product / (q_norm * d_norm)
                scores.append((docIDs[idx], cosine_sim))
            
            # 5. Sort document IDs by score in descending order
            scores.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [doc_id for doc_id, score in scores]
            doc_IDs_ordered.append(ranked_ids)
            
        return doc_IDs_ordered