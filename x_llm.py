
import ollama
import concurrent.futures

def llm_query(prompt, host='local', model='llama3.1', temperature=0.1, seed=None):
    if host == 'local':
        client = ollama
    else:
        client = ollama.Client(host=host)

    options = {
        'temperature': temperature,
    }

    if seed is not None:
        options['seed'] = seed

    response = client.generate(
        model=model, 
        prompt=prompt, 
        stream=False,
        options=options,
    )

    return response


def llm_query_resp(prompt, host='local', model='llama3.1', temperature=0.1):
    if host == 'local':
        client = ollama
    else:
        client = ollama.Client(host=host)

    response = client.generate(
        model=model, 
        prompt=prompt, 
        stream=False,
        options={
            'temperature': temperature
        },
    )

    return response.response

def parallel_llm_queries(prompts, max_workers=5):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(llm_query, prompt) for prompt in prompts]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(f"Error: {e}")
    return results


def parallel_listmodel_queries(prompt, models, host='local', temperature=0.1, max_workers=5):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(llm_query, prompt, host, model, temperature): model for model in models}
        for future in concurrent.futures.as_completed(futures):
            model = futures[future]
            try:
                results[model] = future.result()
            except Exception as e:
                results[model] = f"Error: {e}"
    return results


def llm_chat(messages, host='local', model='llama3.1', options={}):
    if host == 'local':
        client = ollama
    else:
        client = ollama.Client(host=host)

    response = client.chat(
        model=model,
        messages=messages,
        stream=False,
        options=options
    )

    return response

#response = llm_query("O que é IA?")
#print(response['response'])

#response = parallel_llm_queries(["O que é IA?","O que é ASR?", "O que é LLM?"])
#for r in response:
#    print(r['response'])