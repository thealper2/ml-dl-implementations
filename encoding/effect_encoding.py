def effect_encoding(categories):
    unique_categories = sorted(set(categories))

    reference_category = unique_categories[-1]
    reduced_categories = unique_categories[:-1]

    encoded_values = []
    for category in categories:
        encoded_row = [1 if category == unique_cat else 0 for unique_cat in reduced_categories]
        encoded_values.append(encoded_row)

    return encoded_values

categories = ["t-shirt", "hat", "hat", "coat", "coat", "t-shirt", "boots", "hat", "boots"]
encoded_values = effect_encoding(categories)
print("Effect Encoding:")
for i in range(len(categories)):
    print(f"{categories[i]} => {encoded_values[i]}")