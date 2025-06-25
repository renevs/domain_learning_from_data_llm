import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import os
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from torch.distributed import init_process_group, destroy_process_group
from transformers import default_data_collator
import torch.nn.functional as F
import gc
class ModelData:
   def __init__(self) -> None:
        self.model_name_or_path = os.environ.get( "MODEL_NAME", "./Meta-Llama-3-8B-Instruct/" )
        self.tokenizer_name_or_path = self.model_name_or_path
        self.max_length = 48

def freeze_model( model ):
    for param in model.parameters():
        param.requires_grad = False

def read_file( file_path ):
    with open( file_path, 'r') as f:
        lines = f.readlines()
    return lines

class TransformData(object):
  def __init__( self, tokenizer ):
    self.tokenizer = tokenizer
  def __call__(self, sample):
    return {'input_ids':self.tokenizer.encode(sample, add_special_tokens=False)}


class CustomDataset(Dataset):
    def __init__(self, data, transform = None ):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        if self.transform:
            sample = self.transform( sample )
        return sample
    
def _load_checkpoint( model_file, phrase_model ):
    PATH = os.path.join( os.path.normpath( os.environ["WORK_SPACE"] ), model_file )
    checkpoint = torch.load( PATH, weights_only=True )
    phrase_model.load_state_dict(checkpoint['model_back_state_dict'])
    print(f'Modelo lido de {PATH}')
    return checkpoint['epoch']
    
def load_train_objs( model_data, source_file ):
    print("Loading objs", flush=True)

    all_source_data = read_file( source_file)
    print( 'Examples:\n', all_source_data[:3] )
 
    attn_implementation = "eager"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    print("1>>>>> torch.cuda.memory_stats (global free/total gpu): ", torch.cuda.mem_get_info(0) )

    model = AutoModelForCausalLM.from_pretrained(
        model_data.model_name_or_path,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    freeze_model( model )
    print('Base model loaded', flush=True)



    print("Model path", model_data.model_name_or_path )
    print("Tokenizer path", model_data.tokenizer_name_or_path, flush=True )
    tokenizer = AutoTokenizer.from_pretrained(model_data.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    transformData = TransformData( tokenizer )
    current_dataset = CustomDataset(all_source_data, transform=transformData)

    print("Load ended", flush=True)
    return current_dataset, tokenizer, model


class PhraseGeneratorModel( nn.Module ):
    def generate_domain_weights( self, tokenizer, initial_domain, requires_grad=True ):
        
        initial_weights_domains = self.weights_values[ tokenizer.encode(initial_domain) ].clone()
        # initial_weights_domains = torch.full( (len(self.tokens_usados),), -5, dtype=self.weights_values.dtype, device=self.weights_values.device )

        learnable_domain_weights = nn.Parameter( initial_weights_domains, requires_grad=requires_grad )
        
        return learnable_domain_weights
    def __init__(self, llm_model, tokenizer, adapter_name, complete_token_id=-1, fixed_output_size = -1, initial_domain='financial', anchor=False ):
        super(PhraseGeneratorModel, self).__init__()
        self.anchor = anchor
        self.llm_model = llm_model

        self.adapter_name = adapter_name

        self.tokenizer = tokenizer

        self.terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        for name, mod in llm_model.named_modules(): 
            if name=='model.embed_tokens':
                print('****************')
                print( mod )
                print( type(mod) )
                print( mod.weight.shape )
                self.weights_values = mod.weight
                break
        print( self.weights_values )

        self.fixed_output_size = fixed_output_size
        self.complete_token_id = complete_token_id

        
        self.embedding_padding = self.weights_values[ self.tokenizer.eos_token_id ]

        if anchor:
            self.nouns = ['celebrities']
            self.idx_noun = -1
        else:
            self.learnable_domain_weights = self.generate_domain_weights( tokenizer, initial_domain, requires_grad=True )
        self.embedding_complete_token = self.weights_values[ self.complete_token_id ]
        self.prompts = {}
    def add_prompt( self, prompt_values, name = "DEFAULT" ):
        messages_user_part1 = prompt_values['messages_user_part1']
        # messages_user_part2 = "financial - - - -"
        messages_user_part2 = prompt_values['messages_user_part2']

        prompt_convert_domain_text_part1 = prompt_values['prompt_convert_domain_text_part1']
        prompt_convert_domain_text_part2 = prompt_values['prompt_convert_domain_text_part2']


        prompt_ids_part1 = self.tokenizer.encode( prompt_convert_domain_text_part1 ) # Synthetic data generation
        inputs_embeds_part1 = self.weights_values[ prompt_ids_part1 ] # Synthetic data generation
        prompt_ids_part2 = self.tokenizer.encode( prompt_convert_domain_text_part2 ) # assistant...
        inputs_embeds_part2 = self.weights_values[ prompt_ids_part2 ] # assistant...

        messages_user_part1_ids = self.tokenizer.encode( messages_user_part1 )   # Adapt the text above.. to domanin:
        inputs_embeds_user_part1 = self.weights_values[ messages_user_part1_ids ]   # Adapt the text above.. to domanin:

        messages_user_part2_ids = self.tokenizer.encode( messages_user_part2 )   # . Show me only the new text
        inputs_embeds_user_part2 = self.weights_values[ messages_user_part2_ids ]  # . Show me only the new text
        input_size_without_phrase = inputs_embeds_part1.shape[0] + inputs_embeds_user_part1.shape[0] + \
                                inputs_embeds_user_part2.shape[0] + \
                                inputs_embeds_part2.shape[0]

        self.prompts[ name ] = {
            'inputs_embeds_part1': inputs_embeds_part1,
            'inputs_embeds_part2':inputs_embeds_part2,
            'inputs_embeds_user_part1':inputs_embeds_user_part1,
            'inputs_embeds_user_part2':inputs_embeds_user_part2,
            'input_size_without_phrase':input_size_without_phrase
        }

    def _next_noun( self ):
        self.idx_noun+=1
        if self.idx_noun >= len( self.nouns ):
            self.idx_noun = 0
        self.learnable_domain_weights = self.weights_values[ self.tokenizer.encode( self.nouns[self.idx_noun] ) ].clone()
    def forward(self, inputs, return_output_embeddings=False, output_size=-1, goal_phrases = None, freeze = False, prompt='DEFAULT', clean_input=False ): #output_size é o tamanho sem o token de fim de frase

        if self.anchor:
            self._next_noun()
            freeze = False ### sempre é freeze

        current_device = self.llm_model.device

        max_phrase_length=48
        max_frase_size = 0

        if freeze:
            current_learnable_domain_weights = self.learnable_domain_weights.detach()
        else:
            current_learnable_domain_weights = self.learnable_domain_weights

        if 'input_ids' in inputs:
            x=inputs['input_ids']
            total_frases=x.shape[0]
            inputs_embeds_phrases = []
            for i in range(total_frases):
                f = x[i].squeeze(0)[:max_phrase_length]
                if clean_input:
                    mask = f == self.complete_token_id
                    
                    if sum(mask) > 0:
                        pos_init = mask.nonzero(as_tuple=True)[0][0].item()

                        phrase = f[:pos_init]
                    else:
                        phrase = f
                else:
                    phrase = f
                inputs_embeds_phrase = F.embedding(
                    phrase, weight=self.weights_values, padding_idx=None, max_norm=None,
                    norm_type=2, scale_grad_by_freq=False, sparse=False)

                inputs_embeds_phrases.append( inputs_embeds_phrase )
                max_frase_size = max(len(phrase), max_frase_size)
        else:
            inputs_embeds_phrases = inputs['input_embeds']
            total_frases=len(inputs_embeds_phrases)
            for i in range(len(inputs_embeds_phrases)):
                max_frase_size = max(inputs_embeds_phrases[i].shape[0], max_frase_size)

        if goal_phrases is None:
            model_mode = self.llm_model.training
            if self.llm_model.training:
                print("O modelo está em modo de treinamento, colocando em eval.")
                self.llm_model.eval()

            ### 1a. etapa: gera as frases
            with torch.no_grad():
                
                
                padded_input_embeds = []
                padded_attention_mask = []
                # padded_position_ids = []
                
                for i in range(total_frases):
                    inputs_embeds_frase = inputs_embeds_phrases[i]
                    tam_frase = len(inputs_embeds_frase)
                    repeat_factors = (max_frase_size - tam_frase,) + (-1,)* len(self.embedding_padding.shape)
                    input_embeds_cat = torch.cat( (
                                            # ----> ver onde fica melhor o padding
                                            self.embedding_padding.unsqueeze(0).expand( repeat_factors ),
                                            self.prompts[prompt]['inputs_embeds_part1'], 
                                            inputs_embeds_frase, 
                                            self.prompts[prompt]['inputs_embeds_user_part1'],
                                            current_learnable_domain_weights, 
                                            self.prompts[prompt]['inputs_embeds_user_part2'],
                                            self.prompts[prompt]['inputs_embeds_part2'],
                                        ), dim=0 )

                    attention_mask = torch.cat([
                            torch.zeros( max_frase_size - tam_frase, device=current_device ),
                            torch.ones( self.prompts[prompt]['input_size_without_phrase'] + current_learnable_domain_weights.shape[0] + tam_frase, device=current_device ),
                            ]).long()
                    
                    padded_input_embeds.append( input_embeds_cat )

                    padded_attention_mask.append( attention_mask )

                padded_input_embeds =torch.stack( padded_input_embeds )
                padded_attention_mask = torch.stack( padded_attention_mask ).long()
                
                outputs = self.llm_model.generate( 
                    inputs_embeds=padded_input_embeds, 
                    attention_mask=padded_attention_mask, 
                    max_new_tokens=max_phrase_length+2, eos_token_id=self.terminators,
                    pad_token_id=self.tokenizer.pad_token_id
                        #eos_token_id=3
                )

                outputs_ids = outputs.detach()


            if model_mode:
                print("Retornando para treinamento.")
                self.llm_model.train()

        if return_output_embeddings:
            padded_input_embeds = []
            padded_attention_mask = []

            if goal_phrases is not None:
                outputs_ids = goal_phrases
            for i in range(total_frases):
                inputs_embeds_frase = inputs_embeds_phrases[i]
                phrase_answer = outputs_ids[i][:max_phrase_length]
                size_answer = min( max_phrase_length, len(phrase_answer) )
                inputs_embeds_answer = self.weights_values[ phrase_answer ]

                repeat_factors_answer = (max_phrase_length - size_answer,) + (-1,)* len(self.embedding_padding.shape)
                
                input_embeds_cat = torch.cat( (
                                    self.prompts[prompt]['inputs_embeds_part1'], 
                                    inputs_embeds_frase, 
                                    self.prompts[prompt]['inputs_embeds_user_part1'],
                                    current_learnable_domain_weights, 
                                    self.prompts[prompt]['inputs_embeds_user_part2'],
                                    self.prompts[prompt]['inputs_embeds_part2'],
                                    inputs_embeds_answer,
                                    self.embedding_complete_token.unsqueeze(0).expand( repeat_factors_answer ),
                                    ), dim=0 )

                attention_mask = torch.cat([
                        torch.ones( self.prompts[prompt]['input_size_without_phrase'] + current_learnable_domain_weights.shape[0] + max_phrase_length*2, device=current_device ), # O token final não entra no input
                        ]).long()


                padded_input_embeds.append( input_embeds_cat )
                padded_attention_mask.append( attention_mask )

            padded_input_embeds =torch.stack( padded_input_embeds )
            padded_attention_mask = torch.stack( padded_attention_mask ).long()

            position_ids = padded_attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(padded_attention_mask == 0, 1)
            outputs = self.llm_model(
                input_ids=None,
                attention_mask=padded_attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=padded_input_embeds,
                use_cache=False, #True
                output_attentions=False,
                output_hidden_states=False, # False
                return_dict=True,
                cache_position=None, #cache_position
            )

            outputs_ids = []
            for i in range(total_frases):
                generate_phrase = outputs.logits[i,-max_phrase_length-1:-1,:].to(self.weights_values.dtype) #-1 é pq o último token de entrada gerara o primeiro de saída (incluir eos token)
                
                _,uai = torch.max(generate_phrase.detach(), dim=-1) 
                uai = uai.cpu().numpy()
                frase_em_texto = self.tokenizer.decode(uai, skip_special_tokens=False)
                print(f'Frase output ({i}): ', frase_em_texto, flush=True )

                outputs_ids.append( generate_phrase )

        return outputs_ids
    
def get_phrase_generator_model( model, model_data, tokenizer, complete_token_id=-1, initial_domain='' ):
    model_back = PhraseGeneratorModel( model, tokenizer, 'model_back', complete_token_id=complete_token_id, 
                                      fixed_output_size=model_data.max_length, initial_domain=initial_domain )

    message_system = """Synthetic data generation and data adaptation activity for text between brackets."""
    messages_user_part1 = """]
Adapt the text above between brackets to generate a new text to the domain: """
        # messages_user_part2 = "financial - - - -"
    messages_user_part2 = """. Show me only the new phrase generated to the new domain (without the brackets) and put infinite section signs ('§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§') after the phrase. NONE OTHER TEXT!!!"""

    prompt_convert_domain_text_part1 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{message_system}<|eot_id|><|start_header_id|>user<|end_header_id|>

["""
    prompt_convert_domain_text_part2 = f"""<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    model_back.add_prompt( prompt_values = {
            'messages_user_part1':messages_user_part1,
            'messages_user_part2':messages_user_part2,
            'prompt_convert_domain_text_part1':prompt_convert_domain_text_part1,
            'prompt_convert_domain_text_part2':prompt_convert_domain_text_part2 } )
    return model_back
    


def ddp_setup(global_rank, local_rank, world_size, multiple_gpu=False):
    print(f"Global rank = {global_rank} Local rank = {local_rank}")
    init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

class CollatePadding:
    def __init__(self, pad_token_id, max_length, complete_token_id ) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.complete_token_id = complete_token_id
    def collate_tokenize(self, data):
        # print( 'MAx_size:', max_size )
        for element in data:
            complete_size = self.max_length - len(element["input_ids"])
            if complete_size > 0:
                element["input_ids"] = torch.tensor(element["input_ids"] + [self.complete_token_id]*complete_size)
            else:
                element["input_ids"] = torch.tensor(element["input_ids"][-self.max_length:])

        return default_data_collator( data )


def prepare_dataloader(g:torch.Generator, data_set: Dataset, batch_size: int, max_length: int, tokenizer = None, complete_token_id:int = -1 ):
    colate_padding = CollatePadding( tokenizer.pad_token_id, max_length, complete_token_id ).collate_tokenize
    return DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=colate_padding,
    )

def generate_target_file( current_dataset, phrase_gen, tokenizer, target_file, special_char='§' ):
    print('Gerando arquivo de saída', flush=True)
    j=0
    with open( target_file, 'w') as f:
        for i, data in enumerate(current_dataset):
            print( f'Processando batch {i}', flush=True )
            data = {k: v.to(phrase_gen.llm_model.device) for k, v in data.items()}
            outputs_ids = phrase_gen( data, return_output_embeddings=False, freeze = True, prompt='DEFAULT', clean_input=True )
            frase_geradas = tokenizer.batch_decode(outputs_ids.cpu().numpy(), skip_special_tokens=True)
            for idx in range(len(frase_geradas)):
                frase = frase_geradas[idx].replace("\n", "").rstrip( special_char ) 
                f.write( frase+ '\n' )
                print( j, ')', frase, flush=True )
                j+=1
            f.flush()

def main( local_rank, global_rank, world_size, batch_size, random_seed, model_file, source_file, target_file, final_domain ):
    print('Entrou no main', flush=True)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    g = torch.Generator()
    g.manual_seed(random_seed)

    if global_rank == -1:
        global_rank = local_rank
        print(f"Global rank ajustado: {global_rank}", flush=True)
    multiple_gpu = torch.cuda.device_count() > 1
    ddp_setup(global_rank, local_rank, world_size, multiple_gpu)

    model_data = ModelData()
    current_dataset, tokenizer, model = load_train_objs( model_data, source_file )

    special_char = '§'
    complete_token_id = tokenizer.encode( special_char )[0]

    phrase_gen = get_phrase_generator_model( model, model_data, tokenizer, complete_token_id=complete_token_id, initial_domain=final_domain )

    if model_file:
        _load_checkpoint( model_file, phrase_gen )

    source_data = prepare_dataloader(g, current_dataset, batch_size, model_data.max_length, tokenizer, complete_token_id=complete_token_id )

    generate_target_file( source_data, phrase_gen, tokenizer, target_file, special_char=special_char )

    print('****************************** The end ****************************')
    destroy_process_group()

def remove_bar_list( lista ):
    return '_'.join([ l.replace('/','_') for l in lista ]).replace(' ','_')
if __name__ == "__main__":
    print('Started', flush=True)

    os.environ["WORLD_SIZE"]="1"
    os.environ["RANK"]="0"
    os.environ["LOCAL_RANK"]="0"
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="12121"
    os.environ["WORK_SPACE"]="results"
    os.environ["MODEL_NAME"]="./Meta-Llama-3-8B-Instruct/"

    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--batch_size', default=18, type=int, help='Input batch size on each device (default: 8)')
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for torch and numpy")
    parser.add_argument("--model_file", '-m', type=str, help="model file", default='checkpoint_states.pt')
    args = parser.parse_args()
    print( 'Argumentos', args )
    # world_size = torch.cuda.device_count()
    gpus_por_task = int( os.environ.get( "gpus-per-task", "1" ) )
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS'))) * gpus_por_task
    print(f"Valores gpus_task = {gpus_por_task} ntasks = {os.environ.get('SLURM_NTASKS')} world_size={world_size}")
    global_rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    # main( global_rank, local_rank, world_size, args.save_every, args.total_epochs, args.batch_size )
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)

    print( f"Total Devices available: {torch.cuda.device_count()}", flush=True )
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    bases = {
        'laptop':'./datasets/src/laptop_test_frases.txt',
        'restaurant':'./datasets/src/rest_test_frases.txt',
        'financial':'./datasets/src/SEnFIN11_test_frases.txt',
    }
    bases_src = ['laptop','restaurant','financial']
    modelos = ['financial_pt','reddit_games']

    intermediate_domains_map = {'financial_pt':[['sport in english', 'celebrity in german', 'movie in russian'],['sport in english'],['sport in german'],['financial in portuguese']],
                            'reddit_games':[['sport in english', 'celebrity in german', 'movie in russian'],['sport in english'],['sport in german'],['games in portuguese']]}


    final_domains_map = {'financial_pt':['financial in portuguese'],
                    'reddit_games': ['games in portuguese']}
    seeds = [42]
    for s in seeds:
        for src_name in bases_src:
            source_file = bases[src_name]
            for model in modelos:
                final_domains = final_domains_map[model]
                for final_domain in final_domains:
                    gc.collect()
                    torch.cuda.empty_cache()
                    target_file = f"results/generated_datasets/convert_{src_name}_to_{model}__llama__final_{final_domain.replace('/','_').replace(' ','_')}__seed_{s}.txt"
                    print("*****************************************")
                    print(f"Arquivo: {target_file}")
                    main( local_rank, global_rank, world_size, args.batch_size, args.random_seed, None, source_file, target_file, final_domain )


    final_domains_map = {'financial_pt':['movie','portuguese','financial','financial in portuguese'],
                     'reddit_games': ['movie','portuguese','games','games in portuguese']
                     }
    seeds = [42]
    for s in seeds:
        for src_name in bases_src:
            source_file = bases[src_name]
            for model in modelos:
                # target_file = bases[model]
                intermediate_domains = intermediate_domains_map[model]
                for intermediate_domain in intermediate_domains:
                    final_domains = final_domains_map[model]
                    for final_domain in final_domains:
                        if intermediate_domain != final_domain:
                            gc.collect()
                            torch.cuda.empty_cache()
                            model_file = f"final_models/convert_laptop_to_{model}__intermediate_{remove_bar_list(intermediate_domain)}__final_{final_domain.replace('/','_').replace(' ','_')}__seed_{s}.pt"
                            target_file = f"results/generated_datasets/convert_{src_name}_to_{model}__intermediate_{remove_bar_list(intermediate_domain)}__final_{final_domain.replace('/','_').replace(' ','_')}__seed_{s}.txt"
                            print("*****************************************")
                            print( f"Modelo: {model_file}")
                            print(f"Arquivo: {target_file}")
                            if os.path.isfile( target_file ):
                                print( f"File {target_file} already exists, skipping...")
                            else:
                                main( local_rank, global_rank, world_size, args.batch_size, args.random_seed, model_file, source_file, target_file, final_domain )

