from collections import Counter

def count_encoding(categories):
    category_counts = Counter(categories)
    encoded_values = [category_counts[category] for category in categories]
    return encoded_values

x = ["t-shirt", "hat", "hat", "trousers", "coat", "t-shirt", "boots", "hat"]
encoded_values = count_encoding(x)
print("Count Encoding:")
for i in range(len(x)):
    print(f"{x[i]} => {encoded_values[i]}")