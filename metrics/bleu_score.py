from collections import Counter
import numpy as np

def n_gram_precision(reference, hypothesis, n):
    ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)])
    hyp_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) - n + 1)])
    
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    total = sum(hyp_ngrams.values())
    
    return overlap / total if total > 0 else 0

def brevity_penalty(reference, hypothesis):
    ref_length = len(reference)
    hyp_length = len(hypothesis)
    
    if hyp_length > ref_length:
        return 1
    else:
        return np.exp(1 - ref_length / hyp_length) if hyp_length > 0 else 0

def bleu_score(reference, hypothesis, max_n=4):
    precisions = []
    
    for n in range(1, max_n + 1):
        precision = n_gram_precision(reference, hypothesis, n)
        precisions.append(precision)
    
    geo_mean_precision = np.exp(np.mean([np.log(p) for p in precisions if p > 0]))
    bp = brevity_penalty(reference, hypothesis)
    
    return geo_mean_precision * bp

reference = "the cat is on the mat".split()
hypothesis = "the cat on the mat".split()

bleu_score_value = bleu_score(reference, hypothesis)
print(f"BLEU Score: {bleu_score_value:.4f}")
