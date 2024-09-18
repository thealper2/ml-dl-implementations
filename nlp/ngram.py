from collections import Counter
import numpy as np

def generate_ngrams(text, n):
    ngrams = [tuple(text[i:i+n]) for i in range(len(text) - n + 1)]
    return ngrams

def compute_ngram_frequencies(texts, n):
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = generate_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
    
    return Counter(all_ngrams)

documents = [
    "Python is a great programming language",
    "I love programming in Python",
    "Natural language processing is fascinating",
    "Machine learning and NLP are important fields"
]

n = 2
ngram_frequencies = compute_ngram_frequencies(documents, n)

print(f"{n}-gram frequencies:\n", ngram_frequencies)
