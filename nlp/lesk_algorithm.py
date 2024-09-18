from collections import Counter
import nltk
from nltk.corpus import wordnet as wn

def lesk(word, context):
    senses = wn.synsets(word)
    
    if not senses:
        return None
    
    max_overlap = 0
    best_sense = None
    
    context_set = set(context)
    
    for sense in senses:
        definition = sense.definition()
        definition_set = set(definition.split())
        overlap = len(context_set.intersection(definition_set))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense

context = "I went to the bank to withdraw money".split()
word = "bank"
best_sense = lesk(word, context)

if best_sense:
    print(f"Best sense: {best_sense.name()}")
    print(f"definition: {best_sense.definition()}")
else:
    print("Not found.")
