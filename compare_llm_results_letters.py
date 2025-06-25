import json
from collections import Counter

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_labels_by_flags(labels, flags):
    return [
        label_row for label_row, flag_row in zip(labels, flags)
        if flag_row[0] == 1 and flag_row[1] == 1
    ]

def calculate_letter_percentages(data):
    transposed = list(zip(*data))  # column-wise
    results = []
    for col in transposed:
        count = Counter(col)
        total = len(col)
        results.append({
            'A': round((count.get('A', 0) / total) * 100, 2),
            'B': round((count.get('B', 0) / total) * 100, 2),
            'N': round((count.get('N', 0) / total) * 100, 2),
        })
    return results

def main():
    types = ['Topic','Style','Syntactic Structure','Named entities','Sentiment','General meaning','Function','Register']


    flags = load_json('results/jsons/my_list_prompt3a_convert_laptop_to_reddit_games__intermediate_sport_in_english__final_games__seed_42.json' )
    labels = load_json('results/jsons/my_list_prompt4_convert_laptop_to_reddit_games__intermediate_sport_in_english__final_games__seed_42.json')

    # flags = load_json('results/jsons/my_list_prompt3a_convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.json' )
    # labels = load_json('results/jsons/my_list_prompt4_convert_laptop_to_reddit_games__intermediate_sport_in_english__final_games__seed_42.json')

    filtered_labels = filter_labels_by_flags(labels, flags)
    # filtered_labels = labels
    if not filtered_labels:
        print("No data left after filtering.")
        return

    # filtered_labels = labels
    percentages = calculate_letter_percentages(filtered_labels)

    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Item & A (\\%) & B (\\%) & N (\\%) \\\\")
    print("\\hline")
    for i, p in enumerate(percentages):
        print(f"{types[i]} & {p['A']:.2f} & {p['B']:.2f} & {p['N']:.2f} \\\\")
    print("\\hline")
    print("\\end{tabular}")


    # Print results
    print(f"{'Item':<30} {'A (%)':>6} {'B (%)':>6} {'N (%)':>6}")
    for i, p in enumerate(percentages):
        print(f"{types[i]:<30} {p['A']:>6.2f} {p['B']:>6.2f} {p['N']:>6.2f}")

if __name__ == '__main__':
    main()
