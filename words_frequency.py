from collections import Counter, defaultdict
import spacy

nlp = spacy.load("pt_core_news_sm")
bases_src = ['restaurant', 'laptop', 'financial']
target_map = {'financial_pt':'financial in portuguese',
            'reddit_games':'games in portuguese'}


s = 42
# financial
base_dst = 'financial_pt'
arquivo_base = "../bases_dados/financeiro_pt/frases_financeiro_pt.txt"
intermediate_domains = [['sport in english', 'celebrity in german', 'movie in russian'],['sport in english'],['sport in german']]
final_domains = ['movie','portuguese','financial','financial in portuguese']

#games
# base_dst = 'reddit_games'
# arquivo_base = '../bases_dados/reddits/posts_unicos_contents.txt'
# intermediate_domains = [['sport in english', 'celebrity in german', 'movie in russian'],['sport in english'],['sport in german']]
# final_domains = ['movie','portuguese','games','games in portuguese']


bases = {
    'financial_pt':'../datasets/dst/frases_financeiro_pt.txt',
    'reddit_games':'../datasets/dst/frases_reddit_games.txt',
    'laptop':'../datasets/src/laptop_test.txt',
    'restaurant':'../datasets/src/rest_test.txt',
    'financial':'../datasets/src/SEnFIN11_test.txt',
    'pt_laptop':'../datasets/src/laptop_test_pt_frases.txt',
    'pt_restaurant':'../datasets/src/rest_test_pt_frases.txt',
    'pt_financial':'../datasets/src/SEnFIN11_test_pt_frases.txt',
}


def escape_latex( word ):
    return word.replace('#','\\#').replace('_','\\_')
def remove_bar_list( lista ):
    return '_'.join([ l.replace('/','_') for l in lista ]).replace(' ','_')

def bases_list_as_string( lista ):
    if len(lista) == 1:
        return lista[0]
    else:
        return '-'.join( lista )

dados = []
total_frases = []
base_names = []

with open(arquivo_base, 'r', encoding='utf-8') as f:
    frases = f.readlines()
    total_frases.append(len(frases))
    contagem = Counter()
    for frase in frases:
        doc = nlp(frase.strip())
        # extrai substantivos únicos por frase
        substantivos = set(token.text.lower() for token in doc if token.pos_ == "NOUN")
        contagem.update(substantivos)
    dados.append(contagem)
    base_names.append('Original')


# Processamento
contagem = Counter()
c = 0
target = target_map[base_dst]
for src_name in bases_src:
    arquivo = f"results/bases_geradas/convert_{src_name}_to_{base_dst}__llama__final_{target.replace('/','_').replace(' ','_')}__seed_{s}.txt"
    with open(arquivo, 'r', encoding='utf-8') as f:
        frases = f.readlines()
        c += len(frases)
        
        for frase in frases:
            doc = nlp(frase.strip())
            # extrai substantivos únicos por frase
            substantivos = set(token.text.lower() for token in doc if token.pos_ == "NOUN")
            contagem.update(substantivos)
base_names.append('LLAMA 8B')
dados.append(contagem)
total_frases.append(c)


# Processamento
contagem = Counter()
c = 0
for src_name in bases_src:
    for interm in intermediate_domains:
        for final_domain in final_domains:
            arquivo = f"results/bases_geradas/convert_{src_name}_to_{base_dst}__intermediate_{remove_bar_list(interm)}__final_{final_domain.replace('/','_').replace(' ','_')}__seed_{s}.txt"
            with open(arquivo, 'r', encoding='utf-8') as f:
                frases = f.readlines()
                c += len(frases)
                
                for frase in frases:
                    doc = nlp(frase.strip())
                    # extrai substantivos únicos por frase
                    substantivos = set(token.text.lower() for token in doc if token.pos_ == "NOUN")
                    contagem.update(substantivos)
total_frases.append(c)
dados.append(contagem)
base_names.append( 'Learned Domain' )



# Top 10 de cada arquivo (unido e sem repetições)
top_palavras = set()
for contagem in dados:
    top_palavras.update([p for p, _ in contagem.most_common(10)])

# Calcular as frequências
frequencias = defaultdict(list)
for palavra in top_palavras:
    for i in range(3):
        freq = (dados[i][palavra] / total_frases[i]) * 100
        frequencias[palavra].append(freq)

ordenado = sorted(frequencias.items(), key=lambda x: x[1][0], reverse=True)
palavras_ordenadas = [palavra for palavra, _ in ordenado]

# Gerar código LaTeX (pgfplots)
print(r"\begin{tikzpicture}")
print(r"\begin{axis}[ybar, bar width=8pt, width=\textwidth, height=0.5\textwidth, enlarge x limits=0.05, bar width=3pt,")
print(r"legend style={at={(0.2,0.92)}, anchor=north, legend columns=3},")
print(r"ylabel={Frequência (\%)}, xtick=data, symbolic x coords={")

# Lista de palavras para o eixo x
# palavras = list(frequencias.keys())
print(",".join([escape_latex(x) for x in palavras_ordenadas]), end="},\n")

print(r"x tick label style={rotate=45, anchor=east}]")

# Adiciona os dados de cada arquivo
for i in range(3):
    print(f"\\addplot+[xshift={'+'if i <=1 else ''}{1-i}pt] coordinates {{")
    for palavra in palavras_ordenadas:
        print(f"({escape_latex( palavra )},{frequencias[palavra][i]:.2f})", end=' ')
    print("};")
    print(f"\\addlegendentry{{{base_names[i]}}}")

print(r"\end{axis}")
print(r"\end{tikzpicture}")
