import json

#PAth coherent file
# flags_file = 'results/jsons/my_list_prompt3a_convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.json'
flags_file = 'results/jsons/my_list_prompt3b_convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.json'

# Path to the JSON file
# file_path = "results/jsons/my_list_prompt1_convert_financial_to_financial_pt__llama__final_financial_in_portuguese__seed_42.json"
# file_path = "results/jsons/my_list_prompt1_convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.json"
# file_path = 'results/jsons/my_list_prompt2a_convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.json'
# file_path = 'results/jsons/my_list_prompt2b_convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.json'
# file_path = 'results/jsons/my_list_prompt3_convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.json'
# file_path = 'results/jsons/my_list_prompt3a_convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.json'
# file_path = 'results/jsons/my_list_prompt3b_convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.json'


file_path = 'results/jsons/my_list_prompt2a_convert_laptop_to_reddit_games__intermediate_sport_in_english__final_games__seed_42.json'
file_path2 = 'results/jsons/my_list_prompt2b_convert_laptop_to_reddit_games__intermediate_sport_in_english__final_games__seed_42.json'
# file_path = 'results/jsons/my_list_prompt3a_convert_laptop_to_reddit_games__intermediate_sport_in_english__final_games__seed_42.json'
# file_path = 'results/jsons/my_list_prompt3b_convert_laptop_to_reddit_games__intermediate_sport_in_english__final_games__seed_42.json'
only_make_sense = False
current_model = 'Learned Domain'
# current_model = 'Fixed Domain'

types = ['Topic','Style','Syntactic Structure','Named entities','Sentiment','General meaning','Function','Register']
# types = ['Coherence', 'Portuguese', 'Both']
plus_one = 0

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_labels_by_flags(labels, flags):
    return [
        label_row for label_row, flag_row in zip(labels, flags)
        if flag_row[0] == 1 and flag_row[1] == 1
    ]

# Load the JSON file (expected to be a list of lists with 0s and 1s)
binary_matrix = load_json( file_path )
binary_matrix2 = load_json( file_path2 )


if only_make_sense:
    flags = load_json( flags_file )
    binary_matrix = filter_labels_by_flags(binary_matrix, flags)
    binary_matrix2 = filter_labels_by_flags(binary_matrix2, flags)

# Validate input
if not binary_matrix or not all(isinstance(row, list) for row in binary_matrix):
    raise ValueError("The JSON file must contain a list of lists.")

num_rows = len(binary_matrix)
num_columns = len(binary_matrix[0])

# Count the number of 1s for each column
ones_count = [0] * (num_columns+plus_one)
ones_count2 = [0] * (num_columns+plus_one)


for row,row2 in zip(binary_matrix, binary_matrix2):
    for i, value in enumerate(row):
        ones_count[i] += value
    if plus_one == 1:
        ones_count[len(row)] += sum(row) == len(row)
    for i, value in enumerate(row2):
        ones_count2[i] += value
    if plus_one == 1:
        ones_count2[len(row2)] += sum(row2) == len(row2)

# num_rows = tot
# Compute the percentage of 1s for each position (column)
percentages = [(count / num_rows) * 100 for count in ones_count]
percentages2 = [(count / num_rows) * 100 for count in ones_count2]

# Generate LaTeX tablesave_cache()
# print("\\begin{tabular}{\\textwidth}{" + "c" * (num_columns+plus_one) + "}")
# print("\\hline")
# print("\\hline")
# print("Model & " + " & ".join([f"{types[i]}" for i in range(num_columns+plus_one)]) + " \\\\")
# print("\\hline")
# print( current_model+" & " + " & ".join([f"{pct:.2f}\\%" for pct in percentages]) + " \\\\")
# print("\\hline")
# print("\\end{tabular}")

print("\\begin{tabular}{lcc}")
print("\\hline")
print("\\hline")
print("Item & Learned Domain & LLAMA 8B \\\\")
for i in range(num_columns + plus_one):
    print(f"{types[i]} & {percentages[i]:.2f}\\% & {percentages2[i]:.2f}\\% \\\\")
print("\\hline")
print("\\end{tabular}")
