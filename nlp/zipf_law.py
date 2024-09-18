import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def zipf_law(texts):
    all_words = []
    for text in texts:
        tokens = text.lower().split()
        all_words.extend(tokens)
    
    word_counts = Counter(all_words)
    sorted_counts = sorted(word_counts.values(), reverse=True)
    ranks = np.arange(1, len(sorted_counts) + 1)
    return ranks, sorted_counts

documents = [
    "Python is a great programming language",
    "I love programming in Python",
    "Natural language processing is fascinating",
    "Machine learning and NLP are important fields"
]

ranks, frequencies = zipf_law(documents)

plt.figure(figsize=(10, 6))
plt.plot(ranks, frequencies, marker='o', linestyle='-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('Zipf\'s Law')
plt.grid(True)
plt.savefig("zipf_law.png")