import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.distributed import init_process_group, destroy_process_group
import os
import numpy as np
import random

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import zipfile
import xml.etree.ElementTree as ET
import gc
import copy
import re

print('terceiro print', flush=True)


def ddp_setup(global_rank, local_rank, world_size, multiple_gpu=False):
    print(f"Global rank = {global_rank} Local rank = {local_rank}")
    init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def print_norm_params(param):
    # Cálculo da norma L2 (euclidiana) dos gradientes
    print(f'Norma L2 dos gradientes dos parämetros: {param.grad.data.norm(2)}')
def print_norm(model):
    # Cálculo da norma L2 (euclidiana) dos gradientes
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None and param.requires_grad:
            param_norm = param.grad.data.norm(2)  # Norma L2 dos gradientes de um parâmetro
            total_norm += param_norm.item() ** 2  # Somando o quadrado da norma

    total_norm = total_norm ** 0.5  # Raiz quadrada da soma das normas

    print(f'Norma L2 dos gradientes: {total_norm}')
def check_for_nan(model):
    parar = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Parameter {name} contains NaN.")
            parar = True
        elif torch.isinf(param).any():
            print(f"Parameter {name} contain infinite value.")
            parar = True
    return parar

def entropy_loss(logits, temperature=1.0):
    # Ajusta os logits usando a temperatura (escalonamento)
    scaled_logits = logits / temperature

    # Aplica softmax nos logits escalonados
    probs = F.softmax(scaled_logits, dim=-1)

    # Calcula a entropia
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    # Retorna a média da entropia
    return torch.mean(entropy)

def print_embeddings( tokenizer, vocab_embeddings, your_embeddings, message="Phrase input to confirm", distancia = False, similaridades_cos = False ):
    # Calcula a similaridade entre suas embeddings e as embeddings do vocabulário
    for sentence_embeddings in your_embeddings:
        # decoded_sentence = []
        ids_sentence = []
        distancias = []
        if similaridades_cos:
            ids_sentence_similaridade = []
            distancias_similaridade = []

        print( 'Sentence embeddings:', sentence_embeddings.shape[0], flush=True )
        for idx_word_embedding in range(sentence_embeddings.shape[0]):
            word_embedding = sentence_embeddings[idx_word_embedding]
            if similaridades_cos:
                # Calcular similaridade (usando dot product ou cosine similarity)
                similarities = torch.nn.functional.cosine_similarity(word_embedding.unsqueeze(0), vocab_embeddings)
                best_token_id_similarity = torch.argmax(similarities).item()
                ids_sentence_similaridade.append(best_token_id_similarity)
                if distancia:
                    distancias_similaridade += [ similarities[best_token_id_similarity] ]
            distances = torch.norm(vocab_embeddings - word_embedding, dim=1)
            best_token_id = torch.argmin(distances).item()
            ids_sentence.append(best_token_id)
            if distancia:
                distancias += [ distances[best_token_id] ]

        if distancia:
            print(f'{message}: {tokenizer.decode(ids_sentence, skip_special_tokens=False)} - distance: {distancias}', flush=True )
        else:
            print(f'{message}: {tokenizer.decode(ids_sentence, skip_special_tokens=False)}', flush=True )
        if similaridades_cos:
            if distancia:
                print(f'{message} cos: {tokenizer.decode(ids_sentence_similaridade, skip_special_tokens=False)} - distancia: {distancias_similaridade}', flush=True )
            else:
                print(f'{message} cos: {tokenizer.decode(ids_sentence_similaridade, skip_special_tokens=False)}', flush=True )

def print_phrases_ids( tokenizer, batch_input ):
    # batch_input = _clean_phrase( batch_input )
    for phrase_id in range(len(batch_input)):
        phrase = batch_input[phrase_id].cpu().numpy()
        if len( phrase ) > 0:
            phrase = tokenizer.decode(phrase, skip_special_tokens=True)
        else:
            phrase = "<<empty>>"
        print(f'Frase in ({phrase_id}): ', phrase, flush=True )
class Trainer:
    def __init__(
        self,
        model_back: nn.Parameter,
        model_anchor: nn.Parameter,
        source_train_data: DataLoader,
        source_eval_data: DataLoader,
        target_train_data: DataLoader,
        target_eval_data: DataLoader,
        optimizer_generator: torch.optim.Optimizer,
        local_rank: int,
        global_rank: int,
        save_every: int,
        output_size: int,
        complete_token_id: str,
        model_name:str,
    ) -> None:
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.model_back = model_back
        self.model_anchor = model_anchor
        self.train_data_source = source_train_data
        self.train_data_target = target_train_data
        self.eval_data_source = source_eval_data
        self.eval_data_target = target_eval_data
        self.optimizer_generator = optimizer_generator
        self.save_every = save_every
        self.output_size = output_size
        self.complete_token_id = complete_token_id
        self.model_name = model_name
        print("Device ids", self.local_rank)
        self.criterion = nn.BCEWithLogitsLoss()
        self.verify_phrase_loss = nn.CrossEntropyLoss( reduction = 'mean')

        self.weights_values=self.model_back.weights_values # pode ser do in ou do out
        self.tokenizer = self.model_back.tokenizer
        self.terminators = torch.tensor(self.model_back.terminators, device = self.local_rank )

    
    def _remove_terminator_token( self, embeddings_phrase_generate ):
        if len( embeddings_phrase_generate ) > 1:
            return embeddings_phrase_generate[:-1]
        return embeddings_phrase_generate
    
    def _run_one_direction_phrase( self, source_data, generator, prompt='DEFAULT', clean_input = False ):
        outputs_ids = generator( source_data, return_output_embeddings=False, freeze = True, prompt = prompt, clean_input=clean_input)
        return outputs_ids
        

    def _run_one_direction_cycle( self, data, all_generated_embbedding_phrases, generator ):

        source_input_ids = data['input_ids'] #eh o destino
        total_frases=source_input_ids.shape[0]

        outputs_ids_return = generator( {'input_ids':all_generated_embbedding_phrases}, return_output_embeddings=True, output_size=self.output_size, goal_phrases = source_input_ids, freeze = False )



        all_loss_phrases = 0.0
        phrases_considerated = 0
        for i in range(total_frases):
            if torch.any(all_generated_embbedding_phrases[i] != self.complete_token_id):
                f = source_input_ids[i].squeeze(0)
                f_output = outputs_ids_return[i]
                all_loss_phrases += self.verify_phrase_loss( f_output, f )
                phrases_considerated+=1
            else:
                print(f'Warning: Phrase {i} is empty. Ignoring.')
        all_loss_phrases /= max( 1, phrases_considerated )

        

        return all_loss_phrases

    def _run_batch(self, batch_generator, target_domain_input, lr_scheduler_generator=None ):
        self.step_number += 1

        print( f'Step: {self.step_number}')

        print('Ids do source:')
        print_phrases_ids( self.tokenizer, batch_generator['input_ids'] )

        print('\nIds do target:')
        print_phrases_ids( self.tokenizer, target_domain_input['input_ids'] )

        # print('Gerador frase: laptop->financeiro')
        # print('================================')
        # all_generated_embbedding_phrases_go = self._run_one_direction_phrase( batch_generator, self.model_in )

        # print('Gerador frase: laptop->service (anchor)')
        # print('================================')
        # all_generated_embbedding_phrases_go_anchor = self._run_one_direction_phrase( batch_generator, self.model_anchor )

        if self.step_number%30 == 0:
            print('Generator phrase: source->target')
            print('================================', flush=True)
            _ = self._run_one_direction_phrase( batch_generator, self.model_back, prompt='ASPECTS', clean_input=True )

        print('Gerador frase: target->intermediate (anchor)')
        print('================================')
        all_generated_embbedding_phrases_target_anchor = self._run_one_direction_phrase( target_domain_input, self.model_anchor )

        ############# go
        print('Autoencoder:')
        print('===================')
        loss_cycle_go = self._run_one_direction_cycle( target_domain_input, all_generated_embbedding_phrases_target_anchor, self.model_back )

        loss = 1 * loss_cycle_go

        print(f'Loss cycle go: {loss_cycle_go.detach().float()}')

        loss_value = loss.detach().float()

        loss.backward()

        self.optimizer_generator.step()
        self.optimizer_generator.zero_grad()
        loss_cycle_go = None
        lr_scheduler_generator.step()

        parar = check_for_nan(self.model_back)
        if parar:
            print("nan. Stopping...")
            exit(1)

        # print('PArams do modelo')
        # for name, param in self.model_domain.named_parameters():
        #     print(f"{name} :\n{param}")
        return loss_value


    def _run_epoch(self, epoch, lr_scheduler_generator ):
        b_sz = len(next( iter( next(iter(self.train_data_source) ).values() ) ) )
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data_target)}")
        self.model_back.train()
        total_loss = 0


        for target_batch in tqdm(self.train_data_target):
            target_batch = {k: v.to(self.local_rank) for k, v in target_batch.items()}
            batch = next(self.train_data_source)
            batch = {k: v.to(self.local_rank) for k, v in batch.items()}
            loss_value = self._run_batch( batch, target_batch, lr_scheduler_generator )
            total_loss += loss_value
        train_epoch_loss = total_loss / len(self.train_data_target)
        train_ppl = torch.exp(train_epoch_loss)

        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
    def _save_checkpoint(self, epoch, lr_scheduler_generator, model_name=None):
        checkpoint = {
            'epoch': epoch,
            'model_back_state_dict': self.model_back.state_dict(),
            'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
            'scheduler_generator_state_dict': lr_scheduler_generator.state_dict(),
        }
        if model_name:
            PATH = os.path.join( os.path.normpath( os.environ["WORK_SPACE"] ), model_name )
            torch.save(checkpoint, PATH)
            print(f"Final model checkpoint saved at {PATH}")
        else:
            PATH = os.path.join( os.path.normpath( os.environ["WORK_SPACE"] ), "checkpoint_states.pt" )
            torch.save(checkpoint, PATH)
            PATH = os.path.join( os.path.normpath( os.environ["WORK_SPACE"] ), f"checkpoint_states_epoch_{epoch}.pt" )
            torch.save(checkpoint, PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    def _load_checkpoint( self, lr_scheduler_generator ):
        PATH = os.path.join( os.path.normpath( os.environ["WORK_SPACE"] ), "checkpoint_states.pt" )
        checkpoint = torch.load( PATH, weights_only=True )
        self.model_back.load_state_dict(checkpoint['model_back_state_dict'])
        self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator_state_dict'])
        lr_scheduler_generator.load_state_dict(checkpoint['scheduler_generator_state_dict'])
        print(f'Modelo lido de {PATH}')
        return checkpoint['epoch']
    def train(self, max_epochs: int):
        print("Training started")
        self.step_number = 0
        lr_scheduler_generator = get_linear_schedule_with_warmup(
            optimizer=self.optimizer_generator,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_data_target) * max_epochs),
        )

        PATH = os.path.join( os.path.normpath( os.environ["WORK_SPACE"] ), self.model_name )
        if os.path.exists( PATH ):
            # print( "Arquivo encontrado. Lendo...")
            # epoch_initial = self._load_checkpoint( lr_scheduler_generator )
            print('model already exists... skipping...')
        else:
            epoch_initial = 0
            for epoch in range(epoch_initial, max_epochs):
                self._run_epoch(epoch, lr_scheduler_generator)
                if self.local_rank == 0 and (epoch + 1) % self.save_every == 0:
                    self._save_checkpoint(epoch, lr_scheduler_generator)
            self._save_checkpoint(epoch, lr_scheduler_generator, model_name=self.model_name)
            print("Training end")


def ot2bieos_absa(absa_tag_sequence):
    """
    ot2bieos function for end-to-end aspect-based sentiment analysis task
    """
    n_tags = len(absa_tag_sequence)
    #new_ts_sequence = []
    new_absa_sequence = []
    prev_pos = '$$$'

    for i in range(n_tags):
        cur_absa_tag = absa_tag_sequence[i]
        if cur_absa_tag == 'O' or cur_absa_tag == 'EQ':
            # when meet the EQ tag, regard it as O
            new_absa_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_absa_tag.split('-')
            # cur_pos is T
            if cur_pos != prev_pos:
                # prev_pos is O and new_cur_pos can only be B or S
                if i == n_tags - 1:
                    new_absa_sequence.append('S-%s' % cur_sentiment)
                else:
                    next_absa_tag = absa_tag_sequence[i + 1]
                    if next_absa_tag == 'O':
                        new_absa_sequence.append('S-%s' % cur_sentiment)
                    else:
                        new_absa_sequence.append('B-%s' % cur_sentiment)
            else:
                # prev_pos is T and new_cur_pos can only be I or E
                if i == n_tags - 1:
                    new_absa_sequence.append('E-%s' % cur_sentiment)
                else:
                    next_absa_tag = absa_tag_sequence[i + 1]
                    if next_absa_tag == 'O':
                        new_absa_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_absa_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_absa_sequence



def read_by_bieos(file_path):
    sents, labels  = [], []
    word_parts = []
    with open(file_path, 'r', encoding='UTF-8') as fp:
        words, tags = [], []
        for line in fp:
            if "####" not in line:
                word_part = line.strip()
                words = word_part.split(" ")
                tags = ["O"] * len(words)
            else:
                word_part, label_part = line.strip().split("####")
                # I=O love=O apple=T-POS
                tokens = label_part.split(" ")
                # remove some period "."
                tokens = [t for t in tokens if "=" in t]
                # sometimes there are multiple =, such as ==O
                words = ["".join(i.split("=")[:-1]) for i in tokens]
                tags = [i.split("=")[-1] for i in tokens]
                tags = ot2bieos_absa(tags)
            sents.append(words)
            labels.append(tags)
            word_parts.append( word_part )
            words, tags = [], []
    return word_parts, sents, labels


def bieos2generation(word_parts, sents, labels, keep_polarity = True):
    final_sents = []

    for si, s in enumerate(sents):
        pairs = []
        aspect_idx = []
        for wi, w in enumerate(s):
            tag = labels[si][wi]
            if tag == "O":
                aspect_idx = []
                continue

            label, polarity = labels[si][wi].split('-')
            if label in ["B", "I"]:
                aspect_idx.append(wi)
            elif label in ["E", "S"]:
                aspect_idx.append(wi)
                if keep_polarity:
                    aspect_tuple = (
                        " ".join([sents[si][i] for i in aspect_idx]), polarity)
                else:
                    aspect_tuple = " ".join([sents[si][i] for i in aspect_idx])
                pairs.append(aspect_tuple)
                aspect_idx = []
        final_sents.append((word_parts[si], pairs))

    return final_sents
def read_generation_uabsa(file_path, keep_polarity = True):
    # ["I love apple .####[([0, "POS"])]"]
    word_parts, sents, labels = read_by_bieos(file_path)
    final_sents = bieos2generation(word_parts, sents, labels, keep_polarity)
    return final_sents

class CustomDataset(Dataset):
    def __init__(self, data, text_column, label_column, transform = None):
        self.data = data
        self.transform = transform
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_value, label = self.data[index]

        # Pré-processamento do valor de entrada e rótulo
        # input_value = preprocess_input(input_value)
        # label = preprocess_label(label)
        sample = {self.text_column:input_value, self.label_column:label}
        if self.transform:
            sample = self.transform( sample )

        return sample
    
class TransformTrain(object):
  def __init__( self, tokenizer, text_column, label_column, aspects_treatment = None ):
    self.tokenizer = tokenizer
    self.text_column = text_column
    self.label_column = label_column
    self.aspects_treatment = aspects_treatment
  def __call__(self, sample):
    text = sample[self.text_column]
    if self.aspects_treatment=='put':
        target_value = ""
        sample_value = sample[self.label_column]
        for t in range( len(sample_value) ):
            target_value = target_value + sample_value[t]
            if t < len(sample_value) - 1:
                target_value += ','
        if target_value == '':
            target_value = '-'
        text += ' -> ASPECTS: ' + target_value
    elif self.aspects_treatment=='generate':
        text += ' | ASPECTS: -'
    model_input = {'input_ids':self.tokenizer(text, add_special_tokens=False)["input_ids"]}
    return model_input
  

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

  
def load_train_objs( model_data, source_file, target_file ):
    print("Loading train objs", flush=True)

    all_source_data = read_generation_uabsa( source_file, keep_polarity = False)
    print( 'Examples:\n', all_source_data[:3] )
    random.shuffle(all_source_data)
    position_break = int(len(all_source_data)*0.99)
    source_train_data = all_source_data[:position_break]
    source_eval_data = all_source_data[position_break:]

    all_target_data = read_generation_uabsa( target_file, keep_polarity = False)
    position_break = int(len(all_target_data)*0.99)
    target_train_data = all_target_data[:position_break]
    target_eval_data = all_target_data[position_break:]

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
    model = prepare_model_for_kbit_training( model )
    freeze_model( model )
    print('Base model loaded')



    print("Model path", model_data.model_name_or_path )
    print("Tokenizer path", model_data.tokenizer_name_or_path, flush=True )
    tokenizer = AutoTokenizer.from_pretrained(model_data.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    transformTrain = TransformTrain( tokenizer, model_data.text_column, model_data.label_column )
    transformTrain_with_aspects = TransformTrain( tokenizer, model_data.text_column, model_data.label_column, aspects_treatment = 'put' )
    train_dataset = CustomDataset(source_train_data, model_data.text_column, model_data.label_column, transform=transformTrain_with_aspects)
    source_eval_dataset = CustomDataset(source_eval_data, model_data.text_column, model_data.label_column, transform=transformTrain_with_aspects)
    target_dataset = CustomDataset(target_train_data, model_data.text_column, model_data.label_column, transform=transformTrain)
    target_eval_dataset = CustomDataset(target_eval_data, model_data.text_column, model_data.label_column, transform=transformTrain)

    print("Load train ended", flush=True)
    return train_dataset, source_eval_dataset, target_dataset, target_eval_dataset, tokenizer, model

def prepare_dataloader(g:torch.Generator, data_set: Dataset, batch_size: int, max_length: int, tokenizer = None, complete_token_id:int = -1 ):
    colate_padding = CollatePadding( tokenizer.pad_token_id, max_length, complete_token_id ).collate_tokenize
    return DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=colate_padding,
    )

class ModelData:
   def __init__(self) -> None:
        self.model_name_or_path = os.environ.get( "MODEL_NAME", "./Meta-Llama-3-8B-Instruct/" )
        self.tokenizer_name_or_path = self.model_name_or_path
        self.text_column = "Sentiment text"
        self.label_column = "text_label"
        self.max_length = 48


def freeze_model( model ):
    for param in model.parameters():
        param.requires_grad = False

class PhraseGeneratorModel( nn.Module ):
    def generate_domain_weights( self, tokenizer, initial_domain, requires_grad=True ):
        
        initial_weights_domains = self.weights_values[ tokenizer.encode(initial_domain) ].clone()
        # initial_weights_domains = torch.full( (len(self.tokens_usados),), -5, dtype=self.weights_values.dtype, device=self.weights_values.device )

        learnable_domain_weights = nn.Parameter( initial_weights_domains, requires_grad=requires_grad )
        
        return learnable_domain_weights

    def __init__(self, llm_model, tokenizer, adapter_name, complete_token_id=-1, fixed_output_size = -1, initial_domain='restaurant', anchor=False ):
        super(PhraseGeneratorModel, self).__init__()
        self.anchor = anchor
        self.llm_model = llm_model

        self.adapter_name = adapter_name

        self.tokenizer = tokenizer

        self.terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Este usa a matriz do LLAMA
        for name, mod in llm_model.named_modules(): 
            if name=='model.embed_tokens':
                self.weights_values = mod.weight
                break

        self.fixed_output_size = fixed_output_size
        self.complete_token_id = complete_token_id

        
        self.embedding_padding = self.weights_values[ self.tokenizer.eos_token_id ]

        if anchor:
            self.nouns = initial_domain
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
        print_embeddings( self.tokenizer, self.weights_values, [self.learnable_domain_weights.detach()], message="Token mais próximo: ", distancia=True, similaridades_cos=True )

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
                print("Model is in training, switching to eval.")
                self.llm_model.eval()

            ### 1st step: generate phrases
            with torch.no_grad():
                
                
                padded_input_embeds = []
                padded_attention_mask = []
                
                for i in range(total_frases):
                    inputs_embeds_frase = inputs_embeds_phrases[i]
                    tam_frase = len(inputs_embeds_frase)
                    repeat_factors = (max_frase_size - tam_frase,) + (-1,)* len(self.embedding_padding.shape)
                    input_embeds_cat = torch.cat( (
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
                
                # print_embeddings( self.tokenizer, self.weights_values, padded_input_embeds, message="Prompt 1a. chamada: " )


                outputs = self.llm_model.generate( 
                    inputs_embeds=padded_input_embeds, 
                    attention_mask=padded_attention_mask, 
                    max_new_tokens=max_phrase_length+2, eos_token_id=self.terminators, #o +2 max_phrase is fot token eot_id + end_of_text_id
                    pad_token_id=self.tokenizer.pad_token_id
                )

                outputs_ids = outputs.detach()

                frase_gerada = self.tokenizer.batch_decode(outputs_ids.cpu().numpy(), skip_special_tokens=True)
                for idx in range(len(frase_gerada)):
                    f = frase_gerada[idx]
                    print(f'Frase gerada ({idx}): {f} ({len(f)})', flush=True )

            if model_mode:
                print("Retornando para treinamento.")
                self.llm_model.train()



        ### 2nd. step: use the generated phrases
        # TODO: treat variable phrase size
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
                        torch.ones( self.prompts[prompt]['input_size_without_phrase'] + current_learnable_domain_weights.shape[0] + max_phrase_length*2, device=current_device ), # The final token must be removed
                        ]).long()


                padded_input_embeds.append( input_embeds_cat )
                padded_attention_mask.append( attention_mask )

            padded_input_embeds =torch.stack( padded_input_embeds )
            padded_attention_mask = torch.stack( padded_attention_mask ).long()

            # print_embeddings( self.tokenizer, self.weights_values, padded_input_embeds, message="Prompt: " )

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

            ### 3rd step
            outputs_ids = []
            for i in range(total_frases):
                generate_phrase = outputs.logits[i,-max_phrase_length-1:-1,:].to(self.weights_values.dtype) #-1 é pq o último token de entrada gerara o primeiro de saída (incluir eos token)
                
                _,uai = torch.max(generate_phrase.detach(), dim=-1) 
                uai = uai.cpu().numpy()
                frase_em_texto = self.tokenizer.decode(uai, skip_special_tokens=False)
                print(f'Frase output ({i}): ', frase_em_texto, flush=True )

                outputs_ids.append( generate_phrase )
        return outputs_ids
    
def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def trainable_params( params ):
    return [
        p for p in params if p.requires_grad
    ]


def main(local_rank: int,global_rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, random_seed: int, source_file:str, target_file:str, intermediate_domain:str, final_domain:str, model_name:str):
    print('\n*****************************')
    print(f'Iniciando global_rank={global_rank} local_rank={local_rank} world_size={world_size} save_every={save_every} total_epochs={total_epochs} batch_size={batch_size} seed={random_seed}')
    print(f'Source file: {source_file}')
    print(f'Target file: {target_file}')
    print('*********************************************************************************', flush=True)

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
    print("Vai chamar o load", flush=True)
    
    train_dataset, source_eval_dataset, target_dataset, target_eval_dataset, tokenizer, model = load_train_objs( model_data, source_file, target_file )

    g = torch.Generator()
    g.manual_seed(random_seed)
    complete_token_id = tokenizer.encode('§')[0]

    source_train_data = infinite_loader( prepare_dataloader(g, train_dataset, batch_size, model_data.max_length, tokenizer, complete_token_id=complete_token_id ) )
    source_eval_data = prepare_dataloader(g, source_eval_dataset, batch_size, model_data.max_length, tokenizer, complete_token_id=complete_token_id)
    target_train_data = prepare_dataloader(g, target_dataset, batch_size, model_data.max_length, tokenizer, complete_token_id=complete_token_id)
    target_eval_data = prepare_dataloader(g, target_eval_dataset, batch_size, model_data.max_length, tokenizer, complete_token_id=complete_token_id)

    train_features = next(iter(source_train_data))
    print(f"Feature batch shape: {len(train_features['input_ids'])}")
    model_back = PhraseGeneratorModel( model, tokenizer, 'model_back', complete_token_id=complete_token_id, fixed_output_size=model_data.max_length, initial_domain=final_domain)

    message_system = """Synthetic data generation and data adaptation activity for text between brackets."""
    messages_user_part1 = """]
Adapt the text above between brackets to generate a new text to the language/domain: """
        # messages_user_part2 = "financial - - - -"
    messages_user_part2 = """. Show me only the new phrase generated (without the brackets) and put infinite section signs ('§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§') after the phrase. NONE OTHER TEXT!!!"""

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
    
    message_system_aspects = """Synthetic data generation and data adaptation activity for Aspect Based Sentiment Analysis (ABSA)."""
    model_back.add_prompt( prompt_values = {
            'messages_user_part1':"""]
Above there is a content between brackets. Adapt ALL the content (This single example) between brackets to the language/domain: """,
            'messages_user_part2':""". Generate a new one with the format: [<phrase> -> ASPECTS: <aspects>]. SHOW ONLY IT AND NONE OTHER TEXT. 
""",
            'prompt_convert_domain_text_part1':f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{message_system_aspects}<|eot_id|><|start_header_id|>user<|end_header_id|>

Examples:
[All the money went into the interior decoration, none of it went to the chefs. -> ASPECTS: interior decoration, chefs]
===> [All profits went into the CEO's pocket, none of it went to the employees -> ASPECTS: CEO, employees]

[I will be going back very soon. -> ASPECTS: -]
===> [I will be investing in stocks soon. -> ASPECTS: -]

Now, do the same to the content below:

[""",
            'prompt_convert_domain_text_part2':f"""<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
}, name='ASPECTS' )
 
    model_anchor = PhraseGeneratorModel( model, tokenizer, 'model_anchor', complete_token_id=complete_token_id, fixed_output_size=model_data.max_length, initial_domain=intermediate_domain, anchor = True)
    model_anchor.add_prompt( prompt_values = {
            'messages_user_part1':messages_user_part1,
            'messages_user_part2':messages_user_part2,
            'prompt_convert_domain_text_part1':prompt_convert_domain_text_part1,
            'prompt_convert_domain_text_part2':prompt_convert_domain_text_part2 } )


    optimizer_generator = torch.optim.Adam([
                                 {'params':model_back.learnable_domain_weights,'lr':0.0001}], lr=0.0001)
    

    trainer = Trainer(model_back, model_anchor, source_train_data, source_eval_data, target_train_data, target_eval_data, optimizer_generator, local_rank, global_rank, save_every, output_size=model_data.max_length, complete_token_id = complete_token_id, model_name=model_name )
    trainer.train(total_epochs)

    trainer = None # to clean memory
    model_back =  None
    model_anchor = None
    source_train_data = None
    source_eval_data = None
    target_train_data = None
    target_eval_data = None
    optimizer_generator = None
    model = None
    tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

    print('****************************** Fim ****************************')
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
    parser.add_argument('--total_epochs', default=5, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=10, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=18, type=int, help='Input batch size on each device (default: 18)')

    args = parser.parse_args()
    print( 'Argumentos', args )
    gpus_por_task = int( os.environ.get( "gpus-per-task", "1" ) )
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS'))) * gpus_por_task
    print(f"Valores gpus_task = {gpus_por_task} ntasks = {os.environ.get('SLURM_NTASKS')} world_size={world_size}")
    global_rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))

    print( f"Total Devices available: {torch.cuda.device_count()}", flush=True )
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    bases = {
        'laptop':'./datasets/src/laptop_test_frases.txt',
        # 'restaurant':'./datasets/src/rest_test_frases.txt',
        # 'financial':'./datasets/src/SEnFIN11_test_frases.txt',
        'financial_pt':'./datasets/dst/frases_financeiro_pt.txt',
        'reddit_games':'./datasets/dst/frases_reddit_games.txt'
    }
    bases_src = ['laptop']
    bases_dst = ['reddit_games']
    intermediate_domains = [['sport in english', 'celebrity in german', 'movie in russian'],['sport in english'],['sport in german'],['games in portuguese']]
    final_domains = ['movie','portuguese','games','games in portuguese']


    seeds = [13]
    for s in seeds:
        for src_name in bases_src:
            source_file = bases[src_name]
            for dst_name in bases_dst:
                target_file = bases[dst_name]
                if src_name != dst_name:
                    for intermediate_domain in intermediate_domains:
                        for final_domain in final_domains:
                            if intermediate_domain != final_domain:
                                gc.collect()
                                torch.cuda.empty_cache()

                                print(f"Vai chamar main com seed={s} source={src_name} target={dst_name} intermediate={intermediate_domain} final={final_domain}")
                                model_name = f"final_models/convert_{src_name}_to_{dst_name}__intermediate_{remove_bar_list(intermediate_domain)}__final_{final_domain.replace('/','_').replace(' ','_')}__seed_{s}.pt"
                                print(f"Model name: {model_name}")
                                main( local_rank, global_rank, world_size, args.save_every, args.total_epochs, args.batch_size, s, source_file, target_file, intermediate_domain, final_domain, model_name )
                                print(f"Terminou main com seed={s} source={source_file} target={target_file} intermediate={intermediate_domain} final={final_domain}")


    bases_src = ['laptop']
    bases_dst = ['financial_pt']
    intermediate_domains = [['sport in english', 'celebrity in german', 'movie in russian'],['sport in english'],['sport in german'],['financial in portuguese']]
    final_domains = ['movie','portuguese','financial','financial in portuguese']


    seeds = [13]  #### IMPORTANT: MUST PROVIDE SEEDS
    for s in seeds:
        for src_name in bases_src:
            source_file = bases[src_name]
            for dst_name in bases_dst:
                target_file = bases[dst_name]
                if src_name != dst_name:
                    for intermediate_domain in intermediate_domains:
                        for final_domain in final_domains:
                            if intermediate_domain != final_domain:
                                gc.collect()
                                torch.cuda.empty_cache()

                                print(f"Vai chamar main com seed={s} source={src_name} target={dst_name} intermediate={intermediate_domain} final={final_domain}")
                                model_name = f"final_models/convert_{src_name}_to_{dst_name}__intermediate_{remove_bar_list(intermediate_domain)}__final_{final_domain.replace('/','_').replace(' ','_')}__seed_{s}.pt"
                                print(f"Model name: {model_name}")
                                main( local_rank, global_rank, world_size, args.save_every, args.total_epochs, args.batch_size, s, source_file, target_file, intermediate_domain, final_domain, model_name )
                                print(f"Terminou main com seed={s} source={source_file} target={target_file} intermediate={intermediate_domain} final={final_domain}")

