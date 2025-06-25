import random
import subprocess
import requests

from typing import List, Dict, Any
from x_llm import llm_query
# from openai import OpenAI
import time
from tqdm.notebook import tqdm
import json
import ast
import hashlib
import numpy as np
import torch
import os

MODEL_1 = 'openai/gpt-4.1'

MODEL_2 = 'openai/gpt-4.1'

MODEL_PARAMS = {'top_k':1,'temperature':0.1, 'top_p':0.0001}

MY_KEY = ''
CACHE_PATH = "cache_llm_respostas_ANALISES.json"
HOST_OLLAMA = ""
seed = 42


llama_generate_file = None



# ################## sp_DE/fi_PT p/ Financial in Portuguese from Financial in English
dst_file = 'datasets/dst/frases_financeiro_pt.txt'
src_file = 'datasets/src/SEnFIN11_test.txt'
generate_file = 'results/generated_datasets/convert_financial_to_financial_pt__intermediate_sport_in_german__final_financial_in_portuguese__seed_42.txt'
llama_generate_file = 'results/generated_datasets/convert_financial_to_financial_pt__llama__final_financial_in_portuguese__seed_42.txt'

src_domain = 'financial in English'
dst_domain = 'financial twitter in Portuguese'
domain = 1



# ################# Games Reddit sp_en p/ games - Laptop    (4b)
# dst_file = 'datasets/dst/frases_reddit_games.txt'
# src_file = 'datasets/src/laptop_test_frases.txt'
# generate_file = 'results/generated_datasets/convert_laptop_to_reddit_games__intermediate_sport_in_english__final_games__seed_42.txt'
# llama_generate_file = 'results/generated_datasets/convert_laptop_to_reddit_games__llama__final_games_in_portuguese__seed_42.txt'

# src_domain = 'Laptop in English'
# dst_domain = 'game sentences from the Reddit forum in Portuguese'
# domain = 2





def fixar_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # para múltiplas GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#######################################################
# --- CACHE LOCAL ---
fixar_seed( seed )
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        cache = json.load(f)
        print('cache lido', flush=True)
else:
    cache = {}

class OpenRouterBatchInference:
    def __init__(self, api_key: str, models: List[str] = [], models_params: List[Dict[str, str]] = None, system_prompt: str = None, structured_output_instructions: Dict[str, Any] = None):
        """
        Initialize the batch inference class using OpenAI client for OpenRouter.

        Args:
            api_key (str): Your OpenRouter API key.
            models (List[str]): List of model names to query.
            system_prompt (str): Common system prompt for all queries.
            structured_output_instructions (Dict[str, Any]): Instructions to guide the output structure.
        """
        # self.client = OpenAI(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=api_key
        # )
        self.models = models
        if models_params is None:
            self.models_params = []*len(models)
        else:
            self.models_params = models_params
        self.system_prompt = system_prompt
        self.structured_output_instructions = structured_output_instructions
        self.api_key = api_key


    def _create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        """
        Create the message payload.

        Args:
            user_prompt (str): The user-specific prompt.

        Returns:
            List[Dict[str, str]]: Messages to send to the model.
        """
        messages = []
        if self.system_prompt:
            messages.append( {"role": "system", "content": self.system_prompt} )
        messages.append({"role": "user", "content": user_prompt})
        if self.structured_output_instructions:
            messages.append( {"role": "user", "content": f"Format your output according to: {self.structured_output_instructions}"} )
        return messages

    def _query_model(self, model: str, model_params: Dict[str,Any], user_prompt: str) -> str:
        """
        Query a single model using OpenAI client.

        Args:
            model (str): Model name.
            user_prompt (str): User-specific prompt.

        Returns:
            str: Model's output content.
        """
        # completion = self.client.chat.completions.create(
        #     model=model,
        #     messages=self._create_messages(user_prompt),
        #     top_k=1,
        #     temperature=0.1
        # )
        # return completion.choices[0].message.content

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
        }
        payload = {
        "model": model,
        "messages": self._create_messages(user_prompt)
        }
        payload.update(model_params)

        response = requests.post(url, headers=headers, json=payload, timeout = 300)
        response.raise_for_status()  # dispara exceção se status >= 400

        return response.json()['choices'][0]['message']['content']

    def generate_outputs(self, dataset: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate outputs for each model given a labeled dataset.

        Args:
            dataset (List[Dict[str, Any]]): List of data points with 'text' and 'label'.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Outputs organized by model name.
        """
        all_outputs = {model: [] for model in self.models}

        for data_point in tqdm(dataset):
            user_prompt = f"Text: {data_point['text']}\nLabel: {data_point['label']}"

            for model, model_params in zip(self.models, self.models_params):
                try:
                    print(f"Querying model {model}...")
                    output = self._query_model(model, model_params, user_prompt)
                    all_outputs[model].append({
                        "input": data_point,
                        "output": output
                    })
                    time.sleep(1)  # Sleep between requests to be polite
                except Exception as e:
                    print(f"Error querying model {model}: {e}")
                    all_outputs[model].append({
                        "input": data_point,
                        "output": {"error": str(e)}
                    })

        return all_outputs


def limpar_lista(s:str):
    s = s.strip()
    if s.endswith(','):
        s = s[:-1]
    return s
    

def limpar_e_converter(s):
    # Remove blocos de markdown (```json ... ```)
    s = s.strip()
    start = s.find("JSON LIST:")
    if start!=-1:
        s = s[start+len("JSON LIST:"):].strip()
    start = s.find("```json")
    if start != -1:
        s = s[start:]
    end = s[7:].find("```")
    if end != -1:
        s = s[:end+7+3]
    if s.startswith("```json"):
        s = s.strip()[7:-3].strip()

    # Tenta converter como JSON válido
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Tenta como expressão Python segura
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Formato inesperado: {s}") from e


def call_ollama(MODEL, prompt):
    resposta = llm_query(prompt, host=HOST_OLLAMA, model=MODEL, temperature=0.1, seed=None)
    # print( resposta['response'] )
    # resposta.raise_for_status()
    return resposta['response']


def call_openrouter( MODEL, MODEL_PARAMS, prompt_text, system_prompt = None ):

    openRouter = OpenRouterBatchInference( MY_KEY, system_prompt=system_prompt )
    continua = True
    while continua:
        try:
            resposta  = openRouter._query_model( MODEL, MODEL_PARAMS, prompt_text )
            continua = False
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                save_cache()
                print("Erro 429: too many requests / rate limit exceeded. Sleeping...")
                time.sleep( 60 )
            else:
                raise

    
    # time.sleep(1)
    # return limpar_e_converter(resposta)
    return resposta


cache_counter = 0
def cache_llm(prompt, system_prompt = None, MODEL=None, MODEL_PARAMS=MODEL_PARAMS):
    global cache_counter
    key = hashlib.md5(prompt.encode()+MODEL.encode()).hexdigest()
    if key in cache:
        print( 'USed from cache.', flush=True)
        return cache[key]
    try:
        # response_TXT = call_ollama(MODEL, prompt)
        response_TXT = call_openrouter(MODEL, MODEL_PARAMS, prompt, system_prompt=system_prompt)
    except Exception as e:
        save_cache()
        raise

    reply = response_TXT.strip()
    cache[key] = reply
    cache_counter+=1
    if cache_counter % 10 == 0:
        save_cache()
        print('Cache saved to disk.', flush=True)
    print(f"Cache hit: {cache_counter} - {key}", flush=True)
    return reply

def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)    

def read_sentences( f ):
    sentences = []
    with open(f, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
        sentences = [s.split("####")[0] for s in sentences]
    return sentences

############### PROMPT 1
src_file_sentences = read_sentences(src_file)

if generate_file is None:
    all_dst_senteces = read_sentences(dst_file)
    random.shuffle(all_dst_senteces)
    total_train_sentences = int(len(all_dst_senteces)*0.8)
    generate_file_sentences = all_dst_senteces[:total_train_sentences]
    dst_file_sentences = all_dst_senteces[total_train_sentences:]
else:
    dst_file_sentences = read_sentences(dst_file)
    generate_file_sentences = read_sentences(generate_file)

if llama_generate_file is not None:
    llama_generate_sentences = read_sentences(llama_generate_file)


def call_prompt_2( my_list, original, adapted, max = 999999, ignore=True ):
    prompt = f"""Consider the attributes for a domain below:

1. Topic: is the subject of a text, as financial, medical, computer games, etc. Each topic has a vocabulary distribution, and a text can be viewed as a topic mixture. So, the topic often is not preserved when I adapt the phrase to other domain. Be careful, a Indian financial twitter has a different topic from a Brazilian topic.
2. Genre: is a macro concept that incorporates function, register, syntax, and style. For instance, an e-mail, a product description, and a product advertisement can have different genres for the same topics, even if they may share the same vocabulary and entity names. Automatic genre detection in text can analyze the relative frequency of different syntactic categories, the use of particular characters such as punctuation marks, and sentence length distribution.
    2.1. function: texts can be grouped by the communicative purpose. In other words, the text function can be informing, persuading, narrating, instructing, etc.
    2.2. register: conventionalized, functional configuration of language tied to certain broad social situations. Example: a legal discourse has technical vocabulary, a rigid structure, and an impersonal tone.
    2.3. syntax: the grammatical structures typically used in a genre. Example: short sentences and predominance of active voice.
    2.4. style: essentially to do with an individual’s use of language. 


I am evaluating a LLM adaptation activity. Consider the sentence below that a small LLM tried to translate and adapt from "{src_domain}" domain/language (A) to the "{dst_domain}" domain/language (B):

"""
    prompt += "Sentence:\n"
    prompt += "[[TEXT]]"
    prompt += "Now, is sentence B EXACTLY the same of the original A for the items below? Answer the following with 0 or 1 (only numbers, separated by commas). Be strict and answer 0 only when you can proof:\n"
    prompt += "a) Topic\n"
    prompt += "b) Style\n"
    prompt += "c) Similar syntactic structure (disregarding translation)\n"
    prompt += "d) Named entities (Answer 1 if there is none entity)\n"
    prompt += "e) Sentiment\n"
    prompt += "f) General meaning\n"
    prompt += "g) Function (informing, narrating, questioning, etc.)\n"
    prompt += "h) register\n"
    prompt += f'\nIMPORTANT: the output is only the list. None other comment.\n'

    if len(adapted[:max]) != len(original[:max]) or len(adapted[:max]) != len(my_list):
        raise ValueError("Lengths do not match! Error!")

    total_empty = 0
    j = 0
    all_answers = []
    for i in range(len(adapted[:max])):
        # if my_list[i].starts
        f0 = original[i].strip()
        f1 = adapted[i].strip()
        j+=1
        if my_list[i][0] == 1 or ignore:
            current_sentence_str = f"A: {f0}\n"
            current_sentence_str += f"B: {f1}\n\n"
            myprompt = prompt.replace("[[TEXT]]", current_sentence_str)
            # subprocess.run("xclip -selection clipboard", input=myprompt.encode(), shell=True)
            s = cache_llm( myprompt, system_prompt = 'You are a linguist.', MODEL=MODEL_2, MODEL_PARAMS=MODEL_PARAMS )
            # s = limpar_e_converter(s)
            s = limpar_lista(s)
            try:
                all_answers.append( [int(x) for x in s.split(',')] )
            except:
                print( "Failed to parse the answer:", s )
                print( "Sentence:", i )
                save_cache()
                exit(1)
        else:
            all_answers.append( [-1, 0, 0, 0, 0, 0, 0, 0] )
            total_empty += 1


    return all_answers,total_empty

def call_prompt_3( sentences, max = 999999):
    str = """For the built sentence below, answer the following with 0 or 1 (only numbers, separated by commas):

a) Is the sentence clear, complete, and coherent in any language or domain? (Answer 1 only if the sentence is well-formed, intelligible, and fully complete — no abrupt endings, unfinished thoughts, or unnecessary repetitions (including repeated hashtags, words, or phrases that harm naturalness or readability). If there are repeated hashtags, words, or phrases that cause redundancy or reduce clarity or fluency, answer 0. Ignore repetitions in the URL. IMPORTANT: It can be a twitter phrase.
b) Is it in Portuguese?

IMPORTANT: the output is only the list. None other comment.

Sentence: """

    total_empty = 0
    j = 0
    all_answers = []
    for i in range(len(sentences)):
        f = sentences[i].strip()
        j+=1
        if f:
            str_sentence=f"{str}\n{f}\n"
            # s = call_openrouter( str_sentence, system_prompt = 'You are a linguist.', MODEL=MODEL )
            s = cache_llm( str_sentence, system_prompt = 'You are a linguist.', MODEL=MODEL_1, MODEL_PARAMS=MODEL_PARAMS )
            s = limpar_lista(s)
            try:
                all_answers.append( [int(x) for x in s.split(',')] )
            except:
                print( "Failed to parse the answer:", s )
                print( "Sentence:", i )
                save_cache()
                exit(1)
        else:
            all_answers.append( [-1, 0] )
            total_empty += 1
        if j >= max:
            break
    return all_answers, total_empty

def call_prompt_4( original, my_list, group1, group2, max = 999999, ignore=True):
    total_examples = 20
    str = f"""Consider the attributes for a domain below:

1. Topic: is the subject of a text, as financial, medical, computer games, etc. Each topic has a vocabulary distribution, and a text can be viewed as a topic mixture. So, the topic often is not preserved when I adapt the phrase to other domain. 
2. Genre: is a macro concept that incorporates function, register, syntax, and style. For instance, an e-mail, a product description, and a product advertisement can have different genres for the same topics, even if they may share the same vocabulary and entity names. Automatic genre detection in text can analyze the relative frequency of different syntactic categories, the use of particular characters such as punctuation marks, and sentence length distribution.
    2.1. function: texts can be grouped by the communicative purpose. In other words, the text function can be informing, persuading, narrating, instructing, etc.
    2.2. register: conventionalized, functional configuration of language tied to certain broad social situations. Example: a legal discourse has technical vocabulary, a rigid structure, and an impersonal tone.
    2.3. syntax: the grammatical structures typically used in a genre. Example: short sentences and predominance of active voice.
    2.4. style: essentially to do with an individual’s use of language. 

Here are {total_examples} example sentences from a reference domain:
"""
    sample_dst_sentences = random.sample( original, total_examples )
    for i in range(len(sample_dst_sentences)):
        str += f"{i+1}. {sample_dst_sentences[i].strip()}\n"


    str+=f"""
I am comparing two LLM adaptation activity. Consider the two sentences that the two different LLMs tried to translate and adapt to the exemplified domain/language:

Sentences:
A: [[SENTENCE1]]
B: [[SENTENCE2]]

Now, for each item below indicate which is more similar to the exemplified domain. Answer sentence A or B, or answer N if it is a tie:
a) Topic
b) Style
c) Similar syntactic structure (disregarding translation)
d) Named entities
e) Sentiment
f) General meaning
g) Function (informing, narrating, questioning, etc.)
h) register


IMPORTANT: the output is only a list which the answers separeted by commas. None other comment.

"""

    if len(group1[:max]) != len(group2[:max])  or len(group1[:max]) != len(my_list):
        raise ValueError("Lengths do not match! Error!")

    total_empty = 0
    j = 0
    all_answers = []
    for i in range(len(generate_file_sentences[:max])):
        # if my_list[i].starts
        f1 = group1[i].strip()
        f2 = group2[i].strip()
        j+=1
        if my_list[i][0] == 1 or ignore:
            myprompt = str.replace("[[SENTENCE1]]", f1)
            myprompt = myprompt.replace("[[SENTENCE2]]", f2)
            # subprocess.run("xclip -selection clipboard", input=myprompt.encode(), shell=True)
            s = cache_llm( myprompt, system_prompt = 'You are a linguist.', MODEL=MODEL_2, MODEL_PARAMS=MODEL_PARAMS )
            # s = limpar_e_converter(s)
            s = limpar_lista(s)
            try:
                all_answers.append( [x.strip() for x in s.split(',')] )
            except:
                print( "Failed to parse the answer:", s )
                print( "Sentence:", i )
                save_cache()
                exit(1)
        else:
            all_answers.append( ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'] )
            total_empty += 1

    return all_answers, total_empty




max_registers = 2300

print("Calling prompt3 - generated", flush=True)
my_list_prompt3a, total_empty = call_prompt_3( generate_file_sentences, max = max_registers )
save_cache()
print("Calling prompt3 - llama", flush=True)
my_list_prompt3b, total_empty_2 = call_prompt_3( llama_generate_sentences, max = max_registers )
save_cache()
print("Calling prompt4", flush=True)
my_list_prompt4, _ = call_prompt_4( original=dst_file_sentences, my_list=my_list_prompt3a, group1=generate_file_sentences, group2=llama_generate_sentences, max = max_registers )
save_cache()
print("Calling prompt2 - generated", flush=True)
my_list2a, _ = call_prompt_2( my_list_prompt3a, src_file_sentences, generate_file_sentences, max = max_registers )
save_cache()
print("Calling prompt2 - llama", flush=True)
my_list2b, _ = call_prompt_2( my_list_prompt3a, src_file_sentences, llama_generate_sentences, max = max_registers )

print("\n\n", len(my_list_prompt3a), "total sentences generated.")
print("\n\n", total_empty, "empty sentences in the generated file.")
print(total_empty_2, "empty sentences in the llama generated file.\n\n", flush=True)

save_cache()

if generate_file is None:
    output_extension = dst_file.replace('.txt', '.json')
else:
    output_extension = generate_file.split('/')[-1].replace('.txt', '.json')
# Save to a JSON file
# with open(f"results/jsons/my_list_prompt1_{output_extension}", "w", encoding="utf-8") as f:
#     json.dump(my_list, f, ensure_ascii=False, indent=4)

with open(f"results/jsons/my_list_prompt2a_{output_extension}", "w", encoding="utf-8") as f:
    json.dump(my_list2a, f, ensure_ascii=False, indent=4)

with open(f"results/jsons/my_list_prompt2b_{output_extension}", "w", encoding="utf-8") as f:
    json.dump(my_list2b, f, ensure_ascii=False, indent=4)

with open(f"results/jsons/my_list_prompt3a_{output_extension}", "w", encoding="utf-8") as f:
    json.dump(my_list_prompt3a, f, ensure_ascii=False, indent=4)

with open(f"results/jsons/my_list_prompt3b_{output_extension}", "w", encoding="utf-8") as f:
    json.dump(my_list_prompt3b, f, ensure_ascii=False, indent=4)

with open(f"results/jsons/my_list_prompt4_{output_extension}", "w", encoding="utf-8") as f:
    json.dump(my_list_prompt4, f, ensure_ascii=False, indent=4)