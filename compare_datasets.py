import spacy
import numpy as np
from scipy.spatial.distance import jensenshannon, cosine
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from collections import Counter
from functools import reduce
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import defaultdict


def load_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines()]
    sentences = [s.split('####')[0] if s!= '' else '' for s in sentences]
    return [s for s in sentences if s!= '']

def remove_bar_list( lista ):
    return '_'.join([ l.replace('/','_') for l in lista ]).replace(' ','_')

def bases_list_as_string( lista ):
    if len(lista) == 1:
        return lista[0]
    else:
        return '-'.join( lista )


# Carregamento dos modelos
nlp = spacy.load("pt_core_news_sm")
# sbert = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
# sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Using this:
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")



# === LÉXICO ===
def get_vocab(texts):
    return set(word.lower() for text in texts for word in text.split())

def ngram_freqs(texts, n=2):
    def ngrams(text):
        tokens = text.lower().split()
        return list(zip(*[tokens[i:] for i in range(n)]))
    all_ngrams = [ng for text in texts for ng in ngrams(text)]
    counts = Counter(all_ngrams)
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

def jaccard_vocab(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

def jsd_from_freq_dicts(d1, d2):
    keys = sorted(set(d1) | set(d2))
    p = np.array([d1.get(k, 0) for k in keys])
    q = np.array([d2.get(k, 0) for k in keys])
    return jensenshannon(p, q)

# === SINTÁTICO ===
def pos_distribution(texts):
    total = Counter()
    for text in texts:
        doc = nlp(text)
        total.update([token.pos_ for token in doc])
    sum_all = sum(total.values())
    tags = spacy.parts_of_speech.NAMES.values()
    return np.array([total.get(tag, 0)/sum_all for tag in tags])

def dep_path_distribution(texts):
    total = Counter()
    for text in texts:
        doc = nlp(text)
        paths = [f"{token.dep_}_{token.head.pos_}_{token.pos_}" for token in doc]
        total.update(paths)
    sum_all = sum(total.values())
    return total, sum_all

# === SEMÂNTICO ===
def avg_semantic_embedding(texts, method="sbert"):
    if method == "sbert":
        vecs = sbert.encode(texts, normalize_embeddings=True)
    elif method == "spacy":
        vecs = np.array([nlp(text).vector for text in texts])
    return np.mean(vecs, axis=0)

def simililary_mean(distrib1, distrib2):
    sim_matrix = cosine_similarity(distrib1, distrib2)
    return np.mean(sim_matrix)

def compute_gamma_median(vectors):
    # Calcula a matriz de distâncias euclidianas ao quadrado
    dist_matrix = torch.cdist(vectors, vectors, p=2) ** 2
    
    # Obtém a mediana das distâncias (ignorando a diagonal com zeros)
    median_dist = torch.median(dist_matrix[dist_matrix > 0])
    
    # Calcula gamma como 1 / (2 * mediana^2)
    gamma = 1 / (2 * median_dist)
    return gamma.item()

def compute_mmd(X, Y, gama=None):
    """Calcula a Maximum Mean Discrepancy (MMD) entre duas distribuições."""
    
    def rbf_kernel(X, Y, gamma=None):
        """Computa a matriz do kernel RBF."""
        XX = torch.cdist(X, X, p=2) ** 2
        YY = torch.cdist(Y, Y, p=2) ** 2
        XY = torch.cdist(X, Y, p=2) ** 2

        if gamma is None:
            gamma = 1.0 / (2 * X.shape[1])  # Heurística para gamma

        K_xx = torch.exp(-gamma * XX)
        K_yy = torch.exp(-gamma * YY)
        K_xy = torch.exp(-gamma * XY)

        return K_xx, K_yy, K_xy

    K_xx, K_yy, K_xy = rbf_kernel(X, Y, gamma=gama)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd.item()

cache = {}

def get_from_cache(key):
    if key in cache:
        return cache[key]
    else:
        set1 = load_sentences(key)
        vocab1 = get_vocab(set1)
        ngram1 = ngram_freqs(set1)
        pos1 = pos_distribution(set1)
        dep1, total1 = dep_path_distribution(set1)
        vecs1 = sbert.encode(set1, normalize_embeddings=True)
        tensors1 = torch.tensor(vecs1, device='cuda')
        gama_rbf = compute_gamma_median(tensors1)
        std = torch.norm(torch.std(tensors1, axis = 0)).item()
        cache[key] = (set1, vocab1, ngram1, pos1, dep1, total1, vecs1, tensors1, gama_rbf, std)
        return set1, vocab1, ngram1, pos1, dep1, total1, vecs1, tensors1, gama_rbf, std
# === FUNÇÃO PRINCIPAL ===
def compare_sets(set1_str, set2_str, gama_ref_str):

    _, vocab1, ngram1, pos1, dep1, total1, vecs1, tensors1, gama_rbf1, _ = get_from_cache(set1_str)
    _, vocab2, ngram2, pos2, dep2, total2, vecs2, tensors2, _, std = get_from_cache(set2_str)
    if gama_ref_str!=set1_str:
        _, _, _, _, _, _, _, _, gama_rbf1, _ = get_from_cache(gama_ref_str)
        
    # Léxico
    vocab_sim = jaccard_vocab(vocab1, vocab2)
    ngram_jsd = jsd_from_freq_dicts(ngram1, ngram2)

    # Sintático - POS
    pos_jsd = jensenshannon(pos1, pos2)

    # Sintático - dependências
    all_keys = sorted(set(dep1) | set(dep2))
    p = np.array([dep1.get(k, 0)/total1 for k in all_keys])
    q = np.array([dep2.get(k, 0)/total2 for k in all_keys])
    dep_jsd = jensenshannon(p, q)

    # Semântica
    # emb1 = avg_semantic_embedding(set1)
    # emb2 = avg_semantic_embedding(set2)
    # sem_cosine = cosine(emb1, emb2)
    sem_cosine = simililary_mean(vecs1, vecs2)

    # gama_rbf = compute_gamma_median(tensors1)
    mmd = compute_mmd(tensors1, tensors2, gama=gama_rbf1)

    

    return {
        "lexical_vocab_jaccard": vocab_sim,
        "lexical_ngram_jsd": ngram_jsd,
        "syntactic_pos_jsd": pos_jsd,
        "syntactic_dep_jsd": dep_jsd,
        "semantic_cosine": sem_cosine,
        "semantic_mmd": mmd,
        "std": std,
    }

def safe_domain_string(domain):
    if domain is None:
        return "None"
    elif isinstance(domain, list):
        return remove_bar_list(domain)
    else:
        return domain.replace('/', '_').replace(' ', '_')
    
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj
    

def latex_table(results, subset, distance_original = False):
    print("\n\n========= TABELAS EM FORMATO LATEX POR PAR DST–SRC =========\n")

    # Agrupar os resultados por (dst, src)
    grouped = defaultdict(dict)
    for chave, val in results.items():
        parts = chave.split("__")
        dst, src, model, interm, final = (parts + ['?'] * 5)[:5]
        grouped[(dst, src)][chave] = val

    for (dst, src), group in grouped.items():
        # print(f"\n\\textbf{{Destino: {dst.replace('_','\\_')} -- Origem: {src.replace('_','\\_')}}}\n")
        print("\\begin{table*}")
        if distance_original:
            print( f"\\caption{{Portuguese Translation of \\textit{{{beautiful_text[src]}}} vs. Adaptation to \\textit{{{beautiful_text[dst]}}} Domain}}" )
        else:
            print( f"\\caption{{Adaptation to \\textit{{{beautiful_text[dst]}}} from \\textit{{{beautiful_text[src]}}} dataset}}" )
        print("\\centering")
        # print("\\begin{tabular*}{l|c|c|c|c|c|c|c}")
        print("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}")
        print("\\hline")
        print("\\hline")
        print("Itermediate/Initial Final Domain & Vocab Jacc & 2-gram JSD & POS JSD & Dep JSD & MMD & STD\\\\")
        print("\\hline")

        if distance_original:
            # Calcular melhores valores por grupo
            min_metrics = {
                'lexical_vocab_jaccard': min(val[subset].get('lexical_vocab_jaccard', 0) for val in group.values()),
                'std': min(val[subset].get('std', 0) for val in group.values()),
                'semantic_cosine': min(val[subset].get('semantic_cosine', 0) for val in group.values()),
            }
            max_metrics = {
                'lexical_ngram_jsd': max(val[subset].get('lexical_ngram_jsd', 0) for val in group.values()),
                'syntactic_pos_jsd': max(val[subset].get('syntactic_pos_jsd', 0) for val in group.values()),
                'syntactic_dep_jsd': max(val[subset].get('syntactic_dep_jsd', 0) for val in group.values()),
                'semantic_mmd': max(val[subset].get('semantic_mmd', 0) for val in group.values()),
            }
        else:
            # Calcular melhores valores por grupo
            max_metrics = {
                'lexical_vocab_jaccard': max(val[subset].get('lexical_vocab_jaccard', 0) for val in group.values()),
                'std': max(val[subset].get('std', 0) for val in group.values()),
                'semantic_cosine': max(val[subset].get('semantic_cosine', 0) for val in group.values()),
            }
            min_metrics = {
                'lexical_ngram_jsd': min(val[subset].get('lexical_ngram_jsd', 0) for val in group.values()),
                'syntactic_pos_jsd': min(val[subset].get('syntactic_pos_jsd', 0) for val in group.values()),
                'syntactic_dep_jsd': min(val[subset].get('syntactic_dep_jsd', 0) for val in group.values()),
                'semantic_mmd': min(val[subset].get('semantic_mmd', 0) for val in group.values()),
            }

        def fmt(m, v):
            if m in max_metrics and v == max_metrics[m]:
                return f"\\textbf{{{v:.4f}}}"
            elif m in min_metrics and v == min_metrics[m]:
                return f"\\textbf{{{v:.4f}}}"
            else:
                return f"{v:.4f}"

        for chave, val in group.items():
            parts = chave.split("__")
            _, _, model, interm, final = (parts + ['?'] * 5)[:5]
            r = val[subset]

            if final is None or final=='None':
                line = "LLAMA 7B"
            else:
                line = f"{interm.replace('_',' ')}/{final.replace('_',' ')}"
                line = line.replace(' in ', '\\_')
                line = line.replace('portuguese', 'PT')
                line = line.replace('english', 'EN')
                line = line.replace('russian', 'RU')
                line = line.replace('german', 'DE')

                line = line.replace('sport', 'sp')
                line = line.replace('movie', 'mv')
                line = line.replace('games', 'ga')
                line = line.replace('celebrity', 'ce')
                line = line.replace('financial', 'fi')

            print(f"{line} & "
                  f"{fmt('lexical_vocab_jaccard', r.get('lexical_vocab_jaccard', 0))} & "
                  f"{fmt('lexical_ngram_jsd', r.get('lexical_ngram_jsd', 0))} & "
                  f"{fmt('syntactic_pos_jsd', r.get('syntactic_pos_jsd', 0))} & "
                  f"{fmt('syntactic_dep_jsd', r.get('syntactic_dep_jsd', 0))} & "
                #   f"{fmt('semantic_cosine', r.get('semantic_cosine', 0))} & "
                  f"{fmt('semantic_mmd', r.get('semantic_mmd', 0))} & "
                  f"{fmt('std', r.get('std', 0))} \\\\")
            # print("\\hline")
        print("\\hline")
        print("\\hline")
        print("\\end{tabular*}")
        print("\\end{table*}\n")


def latex_table_v2(results, subset, distance_original = False):
    print("\n\n========= TABELAS EM FORMATO LATEX POR PAR DST–SRC =========\n")

    # Agrupar os resultados por (dst, src)
    grouped = defaultdict(dict)
    for chave, val in results.items():
        parts = chave.split("__")
        dst, src, model, interm, final = (parts + ['?'] * 5)[:5]
        grouped[dst][chave] = val

    for dst, group in grouped.items():
        # print(f"\n\\textbf{{Destino: {dst.replace('_','\\_')} -- Origem: {src.replace('_','\\_')}}}\n")
        print("\\begin{table*}")
        if distance_original:
            print( f"\\caption{{Portuguese Translation vs. Adaptation to \\textit{{{beautiful_text[dst]}}} Domain}}" )
        else:
            print( f"\\caption{{Adaptation to \\textit{{{beautiful_text[dst]}}}}}" )
        print("\\centering")
        print("\\color{blue}{")
        # print("\\begin{tabular*}{l|c|c|c|c|c|c|c}")
        print("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcccccc@{}}")
        print("\\hline")
        print("\\hline")
        print("Source dataset & Itermediate Domain & Vocab Jacc & 2-gram JSD & POS JSD & Dep JSD & MMD & STD\\\\")
        print("\\hline")

        subgrouped = defaultdict(dict)
        for chave, val in group.items():
            parts = chave.split("__")
            dst, src, model, interm, final = (parts + ['?'] * 5)[:5]
            if (src, interm) not in subgrouped:
                subgrouped[(src, interm)] = []
            subgrouped[(src, interm)] += [val[subset]]

        subgrouped_mean = defaultdict(dict)
        columns = next(iter(subgrouped.items()))[1][0].keys()
        for src,interm in subgrouped:
            if src not in subgrouped_mean:
                subgrouped_mean[ src ] = {}
            for c in columns:
                values =  [d[c] for d in subgrouped[(src, interm)] if c in d]
                if interm not in subgrouped_mean[ src ]:
                    subgrouped_mean[ src ][ interm ] = {}
                subgrouped_mean[ src ][interm][c] = np.array(values).mean()
                # subgrouped_mean[ (src,iterm) ][c] = np.array(values).mean()

        for src, subsubgroup in subgrouped_mean.items():

            if distance_original:
                # Calcular melhores valores por grupo
                min_metrics = {
                    'lexical_vocab_jaccard': min( val.get('lexical_vocab_jaccard', 0) for val in subsubgroup.values() ),
                    'std': min(val.get('std', 0) for val in subsubgroup.values()),
                    'semantic_cosine': min(val.get('semantic_cosine', 0) for val in subsubgroup.values()),
                }
                max_metrics = {
                    'lexical_ngram_jsd': max(val.get('lexical_ngram_jsd', 0) for val in subsubgroup.values()),
                    'syntactic_pos_jsd': max(val.get('syntactic_pos_jsd', 0) for val in subsubgroup.values()),
                    'syntactic_dep_jsd': max(val.get('syntactic_dep_jsd', 0) for val in subsubgroup.values()),
                    'semantic_mmd': max(val.get('semantic_mmd', 0) for val in subsubgroup.values()),
                }
            else:
                # Calcular melhores valores por grupo
                max_metrics = {
                    'lexical_vocab_jaccard': max( val.get('lexical_vocab_jaccard', 0) for val in subsubgroup.values() ),
                    'std': max(val.get('std', 0) for val in subsubgroup.values()),
                    'semantic_cosine': max(val.get('semantic_cosine', 0) for val in subsubgroup.values()),
                }
                min_metrics = {
                    'lexical_ngram_jsd': min(val.get('lexical_ngram_jsd', 0) for val in subsubgroup.values()),
                    'syntactic_pos_jsd': min(val.get('syntactic_pos_jsd', 0) for val in subsubgroup.values()),
                    'syntactic_dep_jsd': min(val.get('syntactic_dep_jsd', 0) for val in subsubgroup.values()),
                    'semantic_mmd': min(val.get('semantic_mmd', 0) for val in subsubgroup.values()),
                }

            def fmt(m, v):
                if m in max_metrics and v == max_metrics[m]:
                    return f"\\textbf{{{v:.4f}}}"
                elif m in min_metrics and v == min_metrics[m]:
                    return f"\\textbf{{{v:.4f}}}"
                else:
                    return f"{v:.4f}"

            for interm, r in subsubgroup.items():

                if interm is None or interm=='None':
                    line = f"{beautiful_text_short[src]} & LLAMA 7B"
                else:
                    # interm = interm.replace('_',' ')
                    # interm = interm.replace(' in ', '\\_')
                    # interm = interm.replace('portuguese', 'PT')
                    # interm = interm.replace('english', 'EN')
                    # interm = interm.replace('russian', 'RU')
                    # interm = interm.replace('german', 'DE')

                    # interm = interm.replace('sport', 'sp')
                    # interm = interm.replace('movie', 'mv')
                    # interm = interm.replace('games', 'ga')
                    # interm = interm.replace('celebrity', 'ce')
                    # interm = interm.replace('financial', 'fi')
                    
                    line = f"{beautiful_text_short[src]} & {beautiful_text_domains[interm]}"

                print(f"{line} & "
                    f"{fmt('lexical_vocab_jaccard', r.get('lexical_vocab_jaccard', 0))} & "
                    f"{fmt('lexical_ngram_jsd', r.get('lexical_ngram_jsd', 0))} & "
                    f"{fmt('syntactic_pos_jsd', r.get('syntactic_pos_jsd', 0))} & "
                    f"{fmt('syntactic_dep_jsd', r.get('syntactic_dep_jsd', 0))} & "
                    #   f"{fmt('semantic_cosine', r.get('semantic_cosine', 0))} & "
                    f"{fmt('semantic_mmd', r.get('semantic_mmd', 0))} & "
                    f"{fmt('std', r.get('std', 0))} \\\\")
            print("\\hline")
        print("\\hline")
        print("\\end{tabular*}")
        print("}")
        print("\\end{table*}\n")


def latex_table_v3(data):
    for start_name in ['financial_pt', 'reddit_games']:
        for src_name in ['Restaurant', 'Laptop', 'Financial']:
            for base_method in ['llama', 'other']:
                print(f"\n\n*********{start_name} -- {src_name} -- {base_method}\n\n")
                # Detectar bases e métricas
                bases = set()
                metricas = ['lexical_vocab_jaccard','lexical_ngram_jsd','syntactic_pos_jsd','syntactic_dep_jsd', 'semantic_mmd', 'std']
                metricas_text = [ 'Vocab Jacc','2-gram JSD','POS JSD','Dep JSD', 'MMD']
                for metodo, conteudo in data.items():
                    if metodo.startswith(f'{start_name}__{src_name.lower()}') and ((base_method=='llama' and base_method in metodo) or (base_method!='llama' and base_method not in metodo)):
                        bases.add("base_referencia")
                        if "outros_resultados" in conteudo:
                            for base in conteudo["outros_resultados"]:
                                bases.add(base)
                    # if "resultados2" in conteudo:
                    #     metricas.update(conteudo["resultados2"].keys())
                    # if "outros_resultados" in conteudo:
                    #     for base in conteudo["outros_resultados"].values():
                    #         metricas.update(base.keys())

                # Inicializar somas e contagens
                somas = {base: {m: 0.0 for m in metricas} for base in bases}
                contagem = {base: 0 for base in bases}

                # Somar valores
                for metodo, conteudo in data.items():
                    if metodo.startswith(f'{start_name}__{src_name.lower()}') and ((base_method=='llama' and base_method in metodo) or (base_method!='llama' and base_method not in metodo)):
                        if "resultados2" in conteudo:
                            for m in metricas:
                                val = conteudo["resultados2"].get(m)
                                if val is not None:
                                    somas["base_referencia"][m] += val
                            contagem["base_referencia"] += 1
                        if "outros_resultados" in conteudo:
                            for base, valores in conteudo["outros_resultados"].items():
                                for m in metricas:
                                    val = valores.get(m)
                                    if val is not None:
                                        somas[base][m] += val
                                contagem[base] += 1

                # Calcular médias
                medias = {}
                for base in bases:
                    medias[base] = {}
                    for m in metricas:
                        medias[base][m] = somas[base][m] / contagem[base] if contagem[base] > 0 else None

                # Gerar tabela LaTeX transposta:
                # metricas_ordenadas = sorted(metricas)
                bases_ordenadas = sorted(bases)

                # Cabeçalho
                header = "Métrica & " + " & ".join(
                    [f"Destiny ({src_name})" if b == "base_referencia" else b.capitalize() for b in bases_ordenadas]
                ) + " \\\\ \\midrule"

                linhas = []
                for m,t in zip(metricas, metricas_text):
                    valores = []
                    for b in bases_ordenadas:
                        val = medias[b][m]
                        valores.append(f"{val:.3f}" if val is not None else "-")
                    linhas.append(f"{t} & \\textbf{valores[0]} & " + " & ".join(valores[1:]) + " \\\\")

                tabela_latex = "\\begin{tabular}{l" + "r" * len(bases_ordenadas) + "}\n"
                tabela_latex += "\\toprule\n"
                tabela_latex += header + "\n"
                tabela_latex += "\n".join(linhas) + "\n"
                tabela_latex += "\\bottomrule\n"
                tabela_latex += "\\end{tabular}"

                print(tabela_latex)



# === EXEMPLO ===
if __name__ == "__main__":

    bases_src = ['restaurant', 'laptop', 'financial']

    bases_dst = ['financial_pt','reddit_games']
    # bases_dst = ['reddit_games']
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
        'reddit_games':'../bases_dados/reddits/posts_unicos_contents.txt',
        'laptop':'../bases_dados/uabsa/cross_domain/laptop_test.txt',
        'restaurant':'../bases_dados/uabsa/cross_domain/rest_test.txt',
        'financial':'../bases_dados/uabsa/cross_domain/SEnFIN11_test.txt',
        'pt_laptop':'../bases_dados/uabsa/cross_domain/laptop_test_pt_frases.txt',
        'pt_restaurant':'../bases_dados/uabsa/cross_domain/rest_test_pt_frases.txt',
        'pt_financial':'../bases_dados/uabsa/cross_domain/SEnFIN11_test_pt_frases.txt',
    }

    beautiful_text = {
        'financial_pt':'Financial in Portuguese',
        'reddit_games':'Games in Portuguese',
        'laptop':'Laptop in English',
        'restaurant':'Restaurant in English',
        'financial':'Financial in English',
        'pt_laptop':'Translated Laptop to Portuguese',
        'pt_restaurant':'Translated Restaurant to Portuguese',
        'pt_financial':'Translated Financial to Portuguese',
    }

    beautiful_text_short = {
        'financial_pt':'Financial in Portuguese',
        'reddit_games':'Games in Portuguese',
        'laptop':'Laptop',
        'restaurant':'Restaurant',
        'financial':'Financial',
        'pt_laptop':'Translated Laptop to Portuguese',
        'pt_restaurant':'Translated Restaurant to Portuguese',
        'pt_financial':'Translated Financial to Portuguese',
    }

    beautiful_text_domains = {
        'sport in english':'Sport in English', 
        'sport in english-celebrity in german-movie in russian':'Serveral domains',
        'sport in german':'Sport in German',
        'financial in portuguese':'Financial in Portuguese',
        'games in portuguese':'Games in Portuguese',
        'games':'Games',
        'financial':'Financial',
        'movie':'Movie'}


    # results = {}
    # for base_dst in bases_dst:
    #     print("Análise da base:", base_dst)
    #     print("*********************************")
    #     print("*********************************")
    #     ref_file = bases[base_dst]
    #     # sentences_ref = load_sentences(ref_file)

    #     seeds = [42]

    #     for s in seeds:
    #         for src_name in bases_src:
    #             src_file_pt = bases['pt_'+src_name]
    #             other_bases = [x for x in bases_src if x!=src_name]
    #             # src_file_sentences = load_sentences(src_file)
    #             for model in modelos_bases[base_dst]:
    #                 # target_file = bases[model]
    #                 intermediate_domains = intermediate_domains_map[model]
    #                 for intermediate_domain in intermediate_domains:
    #                     final_domains = final_domains_map[model]
    #                     for final_domain in final_domains:
    #                         if intermediate_domain != final_domain or final_domain is None:
    #                             if intermediate_domain:
    #                                 current_file = f"results/bases_geradas/convert_{src_name}_to_{base_dst}__intermediate_{remove_bar_list(intermediate_domain)}__final_{final_domain.replace('/','_').replace(' ','_')}__seed_{s}.txt"
    #                             else:
    #                                 target = target_map[base_dst]
    #                                 current_file = f"results/bases_geradas/convert_{src_name}_to_{base_dst}__llama__final_{target.replace('/','_').replace(' ','_')}__seed_{s}.txt"
    #                             # sentences = load_sentences(current_file)

    #                             chave = f"{base_dst}__{src_name}__{model}__{'-'.join(intermediate_domain) if intermediate_domain else 'None'}__{final_domain}"

    #                             print("\nComparação entre conjuntos de frases:\n", flush=True)
    #                             print( ref_file )
    #                             print( current_file )
    #                             resultados = compare_sets(ref_file, current_file, ref_file)
    #                             for k, v in resultados.items():
    #                                 print(f"{k}: {v:.4f}")

    #                             print("\nEm contraposição ao conjunto de frases:\n", flush=True)
    #                             print( src_file_pt )
    #                             print( current_file )
    #                             resultados2 = compare_sets(current_file, src_file_pt, ref_file)
    #                             for k, v in resultados2.items():
    #                                 print(f"{k}: {v:.4f}")

                                
    #                             print("\nOu outros datasets:\n", flush=True)
    #                             all_other_results = {}
    #                             for x in other_bases:
    #                                 other_file_pt = bases['pt_'+x]
    #                                 resultados3 =  compare_sets(current_file, other_file_pt, ref_file)
    #                                 all_other_results[x] = resultados3
    #                                 for k, v in resultados3.items():
    #                                     print(f"{k}: {v:.4f}")
    #                             results[chave] = {'resultados':resultados,
    #                                               'resultados2':resultados2,
    #                                               'outros_resultados':all_other_results}
    #     # === Salva em JSON ===
    # with open("resultados_crazy.json", "w", encoding="utf-8") as f:
    #     json.dump(convert_numpy(results), f, indent=4, ensure_ascii=False)

    with open("resultados_crazy.json", 'r', encoding='utf-8') as f:
        results = json.load(f)

    print('======================')
    print('====================== RESULTADOS')
    latex_table( results, subset ='resultados' )

    print('======================')
    print('====================== RESULTADOS 2')
    latex_table( results, subset ='resultados2', distance_original=True )

    print('======================')
    print('====================== RESULTADOS 3')
    latex_table_v3(results)