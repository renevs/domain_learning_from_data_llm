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
    
def load_train_objs( model_data ):
    print("Loading objs", flush=True)

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


    print("Load ended", flush=True)
    return tokenizer, model


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

    message_system = """Marketing campaign adaptation activity for text between brackets."""
    messages_user_part1 = """]
Above is a text about a new product. We want to talk about this in a forum. Adapt the text above between brackets to generate a new text to the domain / language: """
        # messages_user_part2 = "financial - - - -"
    # messages_user_part2 = """. The text must be adapted to simulate a user commenting about it. None new information must be generate."""
    messages_user_part2 = """. VERY IMPORTANT: Don´t create any new information to adapt. Only adapt."""

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

def main( local_rank, global_rank, world_size, random_seed, model_file, final_domain ):
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
    tokenizer, model = load_train_objs( model_data )

    special_char = '§'
    complete_token_id = tokenizer.encode( special_char )[0]

    phrase_gen = get_phrase_generator_model( model, model_data, tokenizer, complete_token_id=complete_token_id, initial_domain=final_domain )

    if model_file:
        _load_checkpoint( model_file, phrase_gen )

#     message_part1 = """Rewriteen the command below to be more legible. The command is between brackets.

# ************** START OF THE COMMAND **************
# [Adapt the text below to the domain / language: .
# This pizza is wonderful!"""

#     message_part1 = """Rewrite the command below to be more legible and interpretable. Be your best to adapt the embedding representing the domain and language. The command is between brackets.

# ************** START OF THE COMMAND **************
# [Adapt and translate the text below to the domain / language: """
#     message_part2 = """
# This pizza is wonderful!]
# ************** END OF THE COMMAND **************"""


#     message_part1 = """Rewrite the the command below to be more legible and interpretable. Be your best to adapt the embedding representing the domain and language. You can use many words to describe the domain and language. The command is between brackets.

# ************** START OF THE COMMAND **************
# [Adapt and translate the text to the domain / language: """
#     message_part2 = """]
# ************** END OF THE COMMAND **************"""


#     message_part1 = """Rewrite the the command below to be more legible and interpretable. Be your best to adapt the embedding representing the register, syntax, function, topic and language. You can use many words to describe the register, syntax, function, topic and language. The command is between brackets.

# ************** START OF THE COMMAND **************
# [Adapt and translate the text to the register, syntax, function, topic and language: """
#     message_part2 = """]
# ************** END OF THE COMMAND **************"""

#     message_part1 = """Rewrite the the command below to be more legible and interpretable. Be your best to adapt the embedding representing the register, syntax, function, topic and language. You can use many words to describe the register, syntax, function, topic and language.

# ************** START OF THE COMMAND **************
# \"\"\"Adapt and translate the text to the register, syntax, function, topic and language: """
#     message_part2 = """\"\"\""
# ************** END OF THE COMMAND **************"""


    message_part1 = """Rewrite the the command below, which is between brackets, to be more legible and interpretable. Be your best to adapt the embedding representing the register, syntax, function, topic and target language. You can use many words to describe the register, syntax, function, topic and target language. The command is between brackets. BE VERY CAREFUL AND VERY, VERY, VERY DETAILED WITH DOMAIN.

************** START OF THE COMMAND **************
\"\"\"[Adapt (register, syntax, function and topic) and translate the text using the domain/language: """
    message_part2 = """]\"\"\""
************** END OF THE COMMAND **************"""


#     message_part1 = """Interpretable command activity. I am trying to interpretate the command. Rewrite the command below to be more legible. Do your best. The command is between brackets.

# ************** START OF THE COMMAND **************
# [Adapt and translate the text below to the domain / language: .
# This pizza is wonderful!"""

#     message_part2 = """]
# ************** END OF THE COMMAND **************"""

    message_part1_ids = phrase_gen.tokenizer.encode( message_part1 )   
    inputs_embeds_part1 = phrase_gen.weights_values[ message_part1_ids ]  

    message_part2_ids = phrase_gen.tokenizer.encode( message_part2 )   
    inputs_embeds_part2 = phrase_gen.weights_values[ message_part2_ids ]

    input_embeds_cat = torch.cat( (
                                        inputs_embeds_part1, 
                                        phrase_gen.learnable_domain_weights.detach(),
                                        inputs_embeds_part2
                                  ), dim=0 )

    attention_mask = torch.ones( input_embeds_cat.shape[0], device=input_embeds_cat.device )
                
    outputs = phrase_gen.llm_model.generate( 
        inputs_embeds=input_embeds_cat.unsqueeze(0), 
        attention_mask=attention_mask.unsqueeze(0), 
        max_new_tokens=1024, eos_token_id=phrase_gen.terminators,
        pad_token_id=phrase_gen.tokenizer.pad_token_id
            #eos_token_id=3
    )

    output_ids = outputs.detach()


    print(tokenizer.decode(output_ids[0]))

    # output_ids = phrase_gen( {'input_ids':torch.tensor(tokenizer.encode('KJC 5000 gaming keyboard at unbeatable prices. Contact us!')).reshape(1,-1).to(local_rank)} )
    # print(tokenizer.decode(output_ids[0]))

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

    # model_file = f"final_models/convert_laptop_to_financial_pt__intermediate_sport_in_german__final_financial__seed_13.pt"
    model_file = f"final_models/convert_laptop_to_financial_pt__intermediate_sport_in_german__final_movie__seed_13.pt"
    # model_file = f"../../interpretable_prompt/results/final_models/convert_laptop_to_reddit_games__intermediate_sport_in_german__final_games_in_portuguese__seed_42.pt"
    # model_file = f"../../interpretable_prompt/results/final_models/convert_laptop_to_reddit_games__intermediate_sport_in_german__final_movie__seed_42.pt"
    # final_domain='financial'
    # final_domain='games in portuguese'
    final_domain='movie'
    main( local_rank, global_rank, world_size, args.random_seed, model_file, final_domain )

