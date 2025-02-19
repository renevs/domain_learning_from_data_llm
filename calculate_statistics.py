import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import pdist

def load_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines()]
    return [s for s in sentences if s!= '']

def compute_embeddings(sentences, model):
    return model.encode(sentences, convert_to_tensor=True)


def compute_gamma_median(vectors):
    dist_matrix = torch.cdist(vectors, vectors, p=2) ** 2
    
    median_dist = torch.median(dist_matrix[dist_matrix > 0])
    
    gamma = 1 / (2 * median_dist)
    return gamma.item()

def compute_mmd(X, Y, gama=None):
    """Calculate Maximum Mean Discrepancy (MMD)."""
    
    def rbf_kernel(X, Y, gamma=None):
        XX = torch.cdist(X, X, p=2) ** 2
        YY = torch.cdist(Y, Y, p=2) ** 2
        XY = torch.cdist(X, Y, p=2) ** 2

        if gamma is None:
            gamma = 1.0 / (2 * X.shape[1])  

        K_xx = torch.exp(-gamma * XX)
        K_yy = torch.exp(-gamma * YY)
        K_xy = torch.exp(-gamma * XY)

        return K_xx, K_yy, K_xy

    K_xx, K_yy, K_xy = rbf_kernel(X, Y, gamma=gama)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd.item()

def compute_std( X ):
    return torch.norm(torch.std(X, axis = 0)).item()

def remove_bar_list( lista ):
    return '_'.join([ l.replace('/','_') for l in lista ]).replace(' ','_')

def bases_list_as_string( lista ):
    if len(lista) == 1:
        return lista[0]
    else:
        return '-'.join( lista )

def print_latex_table( bases_src, bases_dst, modelos_bases, intermediate_domains_map, final_domains_map, results, metric, min=True ):
    print("*********************************")
    print(f"Metric: {metric}")
    print("*********************************")
    for base_dst in bases_dst:
        print()
        print("Dataset:", base_dst)
        print("*********************************")
        valor_minimo_por_base ={}
        print( f"Model&", end="" )
        for src_name in bases_src:
            if min:
                valor_minimo_por_base[src_name] = ["---",9999999]
            else:
                valor_minimo_por_base[src_name] = ["---",-9999999]
            print(f"{src_name}&", end="")
        print()
        for model in modelos_bases[base_dst]:
            for intermediate_domain in intermediate_domains_map[model]:
                for final_domain in final_domains_map[model]:
                    if intermediate_domain != final_domain or final_domain is None:
                        for src_name in bases_src:
                            chave = f"{base_dst}{src_name}{model}{intermediate_domain}{final_domain}"
                            current_val = results[chave][metric]
                            if min:
                                if current_val < valor_minimo_por_base[src_name][1]:
                                    valor_minimo_por_base[src_name] = [chave, current_val]
                            else:
                                if current_val > valor_minimo_por_base[src_name][1]:
                                    valor_minimo_por_base[src_name] = [chave, current_val]

        for model in modelos_bases[base_dst]:
            for intermediate_domain in intermediate_domains_map[model]:
                for final_domain in final_domains_map[model]:
                    if intermediate_domain != final_domain or final_domain is None:
                        if final_domain is None:
                            print( f"LLAMA 7B", end="" )
                        else:
                            print( f"{bases_list_as_string(  intermediate_domain )}/{final_domain}", end="" )
                        for src_name in bases_src:
                            chave = f"{base_dst}{src_name}{model}{intermediate_domain}{final_domain}"
                            if valor_minimo_por_base[src_name][0] == chave:
                                print(f"&\\textbf{{{results[chave][metric]:.4f}}}", end="")
                            else:
                                print(f"&{results[chave][metric]:.4f}", end="")
                        print("\\\\")
    print()
            

bases_src = ['restaurant', 'laptop', 'financial']

bases_dst = ['financial_pt','reddit_games']
modelos_bases = {'financial_pt':['llama', 'financial_pt'],
                 'reddit_games':['llama', 'reddit_games']}

intermediate_domains_map = {'financial_pt':[['sport in english', 'celebrity in german', 'movie in russian'],['sport in english'],['sport in german'],['financial in portuguese']],
                            'reddit_games':[['sport in english', 'celebrity in german', 'movie in russian'],['sport in english'],['sport in german'],['games in portuguese']],
                            'llama':[None]}

final_domains_map = {'financial_pt':['movie','portuguese','financial','financial in portuguese'],
                    'reddit_games': ['movie','portuguese','games','games in portuguese'],
                    'llama':[None]}
target_map = {'financial_pt':'financial in portuguese',
              'reddit_games':'games in portuguese'}


bases = {
    'financial_pt':'../bases_dados/financeiro_pt/frases_financeiro_pt.txt',
    'reddit_games':'../bases_dados/reddits/posts_unicos_geral.txt'
}

# ====== Carregar modelo Sentence-BERT ======
model_sentence_bert = SentenceTransformer("all-MiniLM-L6-v2")

results = {}
for base_dst in bases_dst:
    print("Dataset analisys:", base_dst)
    print("*********************************")
    print("*********************************")
    ref_file = bases[base_dst]
    sentences_ref = load_sentences(ref_file)
    embeddings_ref = compute_embeddings(sentences_ref, model_sentence_bert)

    gama_rbf = compute_gamma_median(embeddings_ref)

    # calcula desvio padrão
    std1 = compute_std(embeddings_ref)
    print(f"Standard desviation to {base_dst}: {std1:.6f}")
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    seeds = [42]

    for s in seeds:
        for src_name in bases_src:
            for model in modelos_bases[base_dst]:
                # target_file = bases[model]
                intermediate_domains = intermediate_domains_map[model]
                for intermediate_domain in intermediate_domains:
                    final_domains = final_domains_map[model]
                    for final_domain in final_domains:
                        if intermediate_domain != final_domain or final_domain is None:
                            if intermediate_domain:
                                current_file = f"results/bases_geradas/convert_{src_name}_to_{base_dst}__intermediate_{remove_bar_list(intermediate_domain)}__final_{final_domain.replace('/','_').replace(' ','_')}__seed_{s}.txt"
                            else:
                                target = target_map[base_dst]
                                current_file = f"results/bases_geradas/convert_{src_name}_to_{base_dst}__llama__final_{target.replace('/','_').replace(' ','_')}__seed_{s}.txt"
                            sentences = load_sentences(current_file)
                            embeddings = compute_embeddings(sentences, model_sentence_bert)
                            std1 = compute_std(embeddings)
                            print(f"File: {current_file}: {std1:.6f}")
                            print(f"     STD: {std1:.6f}")

                            mmd_score = compute_mmd(embeddings, embeddings_ref, gama=gama_rbf)
                            print(f"     MMD: {mmd_score:.6f}")
                            print("-------------")

                            chave = f"{base_dst}{src_name}{model}{intermediate_domain}{final_domain}"
                            results[chave] = {'std':std1, 'mmd':mmd_score}

print_latex_table( bases_src, bases_dst, modelos_bases, intermediate_domains_map, final_domains_map, results, metric='mmd' )
print_latex_table( bases_src, bases_dst, modelos_bases, intermediate_domains_map, final_domains_map, results, metric='std', min=False )

