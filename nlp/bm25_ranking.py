import numpy as np

def compute_bm25(documents, query, k1=1.5, b=0.75):
    all_terms = list(set([term for doc in documents for term in doc] + query))
    vocab = {term: idx for idx, term in enumerate(all_terms)}
    
    doc_lengths = np.array([len(doc) for doc in documents])
    avgdl = np.mean(doc_lengths)
    
    tf = np.zeros((len(documents), len(vocab)))
    for doc_idx, doc in enumerate(documents):
        for term in doc:
            if term in vocab:
                tf[doc_idx, vocab[term]] += 1
    
    df = np.sum(tf > 0, axis=0)
    idf = np.log((len(documents) - df + 0.5) / (df + 0.5)) + 1
    
    query_vec = np.zeros(len(vocab))
    for term in query:
        if term in vocab:
            query_vec[vocab[term]] += 1
    
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * (doc_lengths / avgdl).reshape(-1, 1))
    tf_component = numerator / denominator
    
    scores = np.sum(tf_component * idf * query_vec, axis=1)
    
    return scores

def rank_documents_bm25(documents, query, k1=1.5, b=0.75):
    scores = compute_bm25(documents, query, k1, b)
    ranked = sorted(zip(scores, range(len(scores))), key=lambda x: x[0], reverse=True)
    return ranked