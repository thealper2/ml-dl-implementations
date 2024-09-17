import numpy as np

def calculate_perplexity(probabilities):
    log_probs = np.log(probabilities + 1e-10)
    avg_log_prob = -np.mean(log_probs)
    perplexity = np.exp(avg_log_prob)
    return perplexity

probabilities = [0.1, 0.2, 0.05, 0.4, 0.15]

perplexity_value = calculate_perplexity(probabilities)
print(f"Perplexity Score: {perplexity_value:.2f}")
