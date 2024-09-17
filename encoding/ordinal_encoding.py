def ordinal_encoding(categories):
    unique_categories = sorted(set(x))
    category_to_number = {category: idx for idx, category in enumerate(unique_categories)}
    encoded_values = [category_to_number[category] for category in categories]
    return encoded_values

x = ["t-shirt", "hat", "hat", "trousers", "coat", "t-shirt", "boots", "hat"]
encoded_values = ordinal_encoding(x)
print("Ordinal Encoding:")
for i in range(len(x)):
    print(f"{x[i]} => {encoded_values[i]}")