from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Byte pair encoding
def tokenize(text):
    words = [word for word, _ in tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)]
    splits = [[char for char in word] for word in words]
    for pair, merge in merges.items():
        for split in splits:
            for i in range(len(split) - 1):
                if split[i:i+2] == list(pair):
                    split[i:i+2] = [merge]
    return sum(splits, [])

print(tokenize("This is not a token."))  