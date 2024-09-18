import numpy as np
from collections import Counter
import math

def compute_tf(doc):
    tf = Counter(doc)
    total_terms = len(doc)
    for term in tf:
        tf[term] /= total_terms
    return tf

def compute_idf(docs):
    n_docs = len(docs)
    idf = {}
    doc_count = Counter(term for doc in docs for term in set(doc))
    
    for term, count in doc_count.items():
        idf[term] = math.log((n_docs + 1) / (count + 1)) + 1
    
    return idf

def compute_tfidf(docs):
    idf = compute_idf(docs)
    tfidf_matrix = np.zeros((len(docs), len(idf)))
    terms = list(idf.keys())
    
    for i, doc in enumerate(docs):
        tf = compute_tf(doc)
        for j, term in enumerate(terms):
            tfidf_matrix[i, j] = tf.get(term, 0) * idf[term]
    
    return tfidf_matrix, terms

documents = [
    "Python is a great programming language",
    "I love programming in Python",
    "Natural language processing is fascinating",
    "Machine learning and NLP are important fields"
]

docs = [doc.lower().split() for doc in documents]

tfidf_matrix, feature_names = compute_tfidf(docs)

print("TF-IDF Matrix:\n", tfidf_matrix)
print()
print("Feature Names:\n", feature_names)