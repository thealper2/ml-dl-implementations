import math

def softmax(scores):
    probabilities = []

    for score in scores:
        prob = math.exp(score) / sum([math.exp(_) for _ in scores])
        probabilities.append(prob)

    return probabilities

scores = [1, 2, 3]
probabilities = softmax(scores)
print(probabilities) # [0.0900, 0.2447, 0.6652]