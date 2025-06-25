# Domain Learning from Data for Large Language Model Translation

This repository is the official repository to ``Domain Learning from Data for Large Language Model Translation''

## Datasets

/datasets - contain the source and target datasets

## How to install

After download the code, you must download the LLAMA 8B-Instruct to the folder Meta-Llama-3-8B-Instruct.

Create python environment and set dependencies running:
install.sh

## How to run

The code must be runned in this sequence:
* train_domain.py - this script train the model for differents intermediate and final domain prompts and save it in:

    /results/final_models

* generate_adapted_datasets.py - this script generate the datasets using the created models and save the new datasets in:

    /results/generated_datasets

* compare_datasets.py - Calculates the statistics of the generated datasets and compares them with LLAMA and the sentences before the transformation.

For qualitative analysis:
* call_serveral_times_llm.py - call llm to compare adapted sentences
* compare_llm_results.pt / compare_llm_results_letters - put in a table the llm answers

For interpretation:
* interpretable_prompt.py - explain the learned domain
