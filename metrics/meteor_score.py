import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from collections import defaultdict


def meteor_score(references, hypothesis, alpha=0.9, beta=3, gamma=0.5):
    stemmer = PorterStemmer()
    
    def preprocess(text):
        tokens = word_tokenize(text.lower())
        return [stemmer.stem(token) for token in tokens]
    
    refs_processed = [preprocess(ref) for ref in references]
    hyp_processed = preprocess(hypothesis)
    
    best_score = -1
    
    for ref in refs_processed:
        exact_matches = set(ref) & set(hyp_processed)
        exact_precision = len(exact_matches) / len(hyp_processed) if hyp_processed else 0
        exact_recall = len(exact_matches) / len(ref) if ref else 0
        
        synonym_matches = 0
        for h_word in hyp_processed:
            for r_word in ref:
                h_synsets = wn.synsets(h_word)
                r_synsets = wn.synsets(r_word)
                if h_synsets and r_synsets:
                    if h_synsets[0].wup_similarity(r_synsets[0]) is not None:
                        if h_synsets[0].wup_similarity(r_synsets[0]) > 0.6:
                            synonym_matches += 1
                            break
        
        syn_precision = synonym_matches / len(hyp_processed) if hyp_processed else 0
        syn_recall = synonym_matches / len(ref) if ref else 0
        
        precision = alpha * exact_precision + (1 - alpha) * syn_precision
        recall = alpha * exact_recall + (1 - alpha) * syn_recall
        
        if precision == 0 or recall == 0:
            f_score = 0
        else:
            f_score = (10 * precision * recall) / (9 * precision + recall)
        
        matched_indices = []
        for i, word in enumerate(hyp_processed):
            if word in ref or any(wn.synsets(word) and wn.synsets(r_word) and 
                                wn.synsets(word)[0].wup_similarity(wn.synsets(r_word)[0]) > 0.6
                                for r_word in ref):
                matched_indices.append(i)
        
        if not matched_indices:
            frag_penalty = 0
        else:
            gaps = [matched_indices[i+1] - matched_indices[i] - 1 
                   for i in range(len(matched_indices)-1)]
            frag_penalty = (sum(1 for gap in gaps if gap > 0) + 1) / (len(matched_indices) + 1)
        
        score = f_score * (1 - beta * frag_penalty ** gamma)
        
        if score > best_score:
            best_score = score
    
    return np.clip(best_score, 0, 1)