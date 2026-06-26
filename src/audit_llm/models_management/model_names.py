import os

LLama_HPC = [
    "codellama/CodeLlama-70b-Instruct-hf",
    "codellama/CodeLlama-70b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-Guard-2-8B"
]
GPT = [
    'aubmindlab/aragpt2-base',
    'aubmindlab/aragpt2-large',
    'aubmindlab/aragpt2-medium',
    'aubmindlab/aragpt2-mega',
    'gpt2',
    'gpt2-large',
    'gpt2-medium',
    'gpt2-xl',
]
DeBerta = [
'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
'MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary',
'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',
'cross-encoder/nli-deberta-base',
'cross-encoder/nli-deberta-v3-base',
'cross-encoder/nli-deberta-v3-large',
'cross-encoder/nli-distilroberta-base',
'cross-encoder/nli-roberta-base',
]
BLOOM_HPC = [
'bigscience/bloom',
'bigscience/bloom-1b1',
'bigscience/bloom-1b7',
'bigscience/bloom-3b',
'bigscience/bloom-560m',
'bigscience/bloom-7b1',
'bigscience/bloom-optimizer-states',
'bigscience/bloomz',
'bigscience/bloomz-7b1',
'bigscience/bloomz-mt',
]
Falcon_HPC = [
'tiiuae/falcon-180b-chat',
'tiiuae/falcon-40b-instruct',
'tiiuae/falcon-7b-instruct',
]
Microsoft_HPC= [
'microsoft/phi-2',
'microsoft/DialoGPT-large',
'microsoft/DialoGPT-medium',
'microsoft/DialoGPT-small',
]
Facebook_HPC = [
'facebook/opt-13b',
'facebook/opt-66b',
'facebook/opt-iml-max-30b',
]
Mistral_HPC = [
'mistralai/Mistral-7B-Instruct-v0.1',
'mistralai/Mistral-7B-Instruct-v0.2',
'mistralai/Mistral-7B-v0.1',
'mistralai/Mixtral-8x22B-Instruct-v0.1',
'mistralai/Mixtral-8x22B-v0.1',
'mistralai/Mixtral-8x7B-Instruct-v0.1',
'mistralai/Mixtral-8x7B-v0.1',
]

Microsoft = ['microsoft/phi-2', 'microsoft/phi-1_5']

# Fr Model
Camembert = ['almanach/camembert-base', 'camembert-base', 'dangvantuan/sentence-camembert-base', 'dangvantuan/sentence-camembert-large']
Croissant = ['croissantllm/CroissantLLMBase', 'croissantllm/CroissantLLMChat-v0.1']
FlauBERT = ['flaubert/flaubert_base_uncased', 'hugorosen/flaubert_base_uncased-xnli-sts']


# Surely not supported by vllm :
Unicorn = ['Unicorn 11B'] #wtfpl --> "Do what the fuck you want" so ok.

# These lists are consumed by model selection.
T5 = ['google/flan-t5-large', 'google/flan-t5-xl',
       'google-t5/t5-3b', 'google-t5/t5-large',
     'google/flan-t5-large', 'google/flan-t5-xxl'] #xl : 2.85b # Use 26% and 8% of memory with bs =30
XLMRobertaModel = ['intfloat/multilingual-e5-large'] # License ok.
BertForMaskedLM = ['johngiorgi/declutr-sci-base'] # License ok.

# Selection 2025
GOOGLE = [
    'google/recurrentgemma-2b-it', # 30% memory with bs = 30
    #'google/gemma-7b',
    #'google/gemma-2-2b',
    'google/flan-t5-xl',]

#'google/flan-t5-xxl',# 167Go
#'google/flan-t5-large', # 3Go

# WORKING
BLOOM_LAB =[
    #'bigscience/bloom-560m',
    #'bigscience/bloom-1b7',
    'bigscience/bloom-7b1', ]


ELEUTHER_AI = [
                    #'EleutherAI/pythia-1.4b', "Invalide device ordinal"
                  # 'EleutherAI/gpt-j-6b',
                  #'EleutherAI/pythia-12b', # cf files not_working_on_lab
                  'EleutherAI/gpt-neox-20b' # 
                  ]

MISTRAL = [#'mistralai/Mistral-7B-Instruct-v0.1',
            #'mistralai/Mistral-7B-Instruct-v0.2',
             # 'mistralai/Mistral-7B-Instruct-v0.3'
             ]
# WORKING
OPT = [ 'facebook/opt-6.7b', 'facebook/opt-13b'] 

# WORKING
QWEN = ['Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-7B',]

SELECTION_2025= GOOGLE + BLOOM_LAB + MISTRAL + ELEUTHER_AI + OPT + QWEN

OpenAI = [ ] 
Microsoft_LAB = ['microsoft/phi-1_5', 'microsoft/Phi-3-mini-128k-instruct']

# More than 14b models
CommandR = ['CohereForAI/c4ai-command-r-plus', 'CohereForAI/c4ai-command-r-v01'] # 104b and 34b Creative Commons Attribution Non Commercial 4.0 seems ok.
DBRX = ['databricks/dbrx-instruct'] # 132b

# Finetuned models:

LLama = ['meta-llama/Llama-3.3-70B-Instruct', 'decapoda-research/llama-13b-hf', 'decapoda-research/llama-7b-hf'] 
# OSError: decapoda-research/llama-13b-hf is not a local folder or a valid repository name on 'https://hf.co'
# for 7b-hf: TypeError: not a string

DeciLM = ['Deci/DeciLM-7B-instruct', 'Deci/DeciLM-6B-instruct']


InternLM2 = ['internlm/internlm2-chat-7b']

# Not with right license
Aquila = [  'BAAI/AquilaChat-7B', 'BAAI/AquilaChat2-7B' ]
BigAquila = ['BAAI/AquilaChat2-34B']
Baichuan = ['baichuan-inc/Baichuan2-13B-Chat', 'baichuan-inc/Baichuan2-7B-Chat',
            'baichuan-inc/Baichuan-13B-Chat'] 
ChatGLM = [ 'THUDM/chatglm2-6b', 'THUDM/chatglm3-6b'] 


# Model_dictionnaries
model_to_add = {'Mistral_HPC': Mistral_HPC,
                'Microsoft_HPC': Microsoft_HPC,
                'Facebook_HPC': Facebook_HPC,
                #'Falcon_HPC': Falcon_HPC,
                #'BLOOM_HPC': BLOOM_HPC,
                #'DeBerta': DeBerta,
                #'GPT': GPT_HPC,
                           }
model_to_add_lab = {'BLOOM_LAB': BLOOM_LAB,
                    'Microsoft_LAB': Microsoft_LAB,
                    'EleutherAI_LAB': ELEUTHER_AI,
                           }

model_families_fr = { 'Camembert': Camembert,
                    'Croissant': Croissant,
                    'FlauBERT': FlauBERT,
                    }

model_families_not_vllm = { 
                            'T5': T5,                     
                            'XLMRobertaModel': XLMRobertaModel,
                            'BertForMaskedLM': BertForMaskedLM,
                            'Google_Gemma': GOOGLE,
                            'LLama': LLama,
                            'OpenAI': OpenAI
                            }   

model_families_more_14b = {
                           'CommandR': CommandR,
                            'DBRX': DBRX,
                            }

model_families_vllm ={              
                    'DeciLM': DeciLM,
                    #'LLama': LLama,
                    'LLama': LLama_HPC,
                    'Mistral': MISTRAL,
                    'OPT': OPT,
                    'Qwen': QWEN,
                    'InternLM2': InternLM2,
                    'Microsoft': Microsoft,
                    }

model_families_no_license ={
                    'BigAquila': BigAquila,
                    'Aquila': Aquila,
                    'Baichuan': Baichuan,
                    'ChatGLM': ChatGLM,
                    }

# Debugging 
# Does not seem working with vllm:
# sacremoses library missing

# model_name = 'decapoda-research--llama-7b-hf' config.json missing
# model_name = 'decapoda-research/llama-13b-hf' ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported

""" 
llama 13b has a super weird behaviour :'The future of AI is', Generated text: ' титуakoiryajaxxftoberhalbś共ヨDelta souventViewHolder Bet rac exper'
"""
NotFinetuned_GPT = ['gpt2']
NotFinetuned_Asi = ['asi/gpt-fr-cased-base',] # 30% memory when bs = 30 

NotFinetuned_InternLM2 = ['internlm/internlm2-7b', ] # requires a dedicated HF cache entry; incompatible with a shared hub cache.

NotFinetuned_LLaMA = ['huggyllama/llama-7b', 'huggyllama/llama-13b', ] # ~80% GPU memory at batch_size=30 on a single A100-40GB.

NotFinetuned_Qwen = ['Qwen/Qwen-7B']
NotFinetuned_Qwen2 =['Qwen/Qwen2-beta-7B']
NotFinetuned_Mistral = []
NotFinetuned_StableLM = ['stabilityai/stablelm-3b-4e1t']

NotFinetuned_OPT = ['facebook/opt-125m']

model_families_not_finetuned={
    'NotFinetuned_Asi': NotFinetuned_Asi,
    'NotFinetuned_GPT': NotFinetuned_GPT,
    'NotFinetuned_LLaMA': NotFinetuned_LLaMA,
    #'NotFinetuned_StableLM': NotFinetuned_StableLM,
}

def model_name_to_model_path(model_name):
    """
    Transforms names into their corresponding paths.
    Examples : 
    'mistralai/Mistral-7B-Instruct-v0.2'->'/datasets/huggingface_hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6'
    '01-ai/Yi-6B' -> '/datasets/huggingface_hub/models--01-ai--Yi-6B/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6'

    Update(08/01/25): not useful anymore ?
    """
    model_name = model_name.replace('/', '--')
    model_path = '/datasets/huggingface_hub/models--' + model_name + '/snapshots/'
    
    dirs = os.listdir(model_path)
    if len(dirs) == 1 :
        return os.path.join(model_path, dirs[0]) 
    else :
        raise ValueError(f"There is more than one file in the {model_path}")

def model_family_selection(family_names: list[str]):
    """
    Obscelete. 
    Available families: 'InternLM2',
                        'Microsoft', 'Mistral',
                        'OPT',                         
                        'Qwen2',
                        'StableLM'
    """    
    path_model_map = {}

    for family_name in family_names:
        for model_name in model_families_vllm[family_name]:
            path_model_map[model_name] = model_name_to_model_path(model_name) # Version <= 0.4.0
            # path_model_map[model_name] = model_name
    return path_model_map

def vllm_model_names_to_path(model_names: list[str]):
    """ 
    Useful in Version <= 0.4.0
    """    
    path_model_map = {}

    for model_name in model_names:
        path_model_map[model_name] = model_name_to_model_path(model_name) # Version <= 0.4.0
    return path_model_map


# Returns errors (see reasons on the sheet Model_inventory) :
'stabilityai/stablelm-base-alpha-7b-v2'

OLMo = ['allenai/OLMo-1B', 'allenai/OLMo-7B']



# Notes on models

##Troubleshooting
# intfloat/multilingual-e5-large:
"""If you want to use `XLMRobertaLMHeadModel` as a standalone, add `is_decoder=True.`
Some weights of XLMRobertaForCausalLM were not initialized from the model checkpoint at intfloat/multilingual-e5-large and are newly initialized: ['lm_head.bias', 'lm_head.decoder.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""

# johngiorgi/declutr-sci-base 
""" 
eos_token is None for this model, which propagates to pad_token and causes downstream errors.

"""

# intern_lm 
"""FileNotFoundError: [Errno 2] No such file or directory: 'snapshots/4275caa205dbb8ff83930e2c1ce6bc62ec49329c/tokenization_internlm2.py"""

#llama 13b 
"""OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like huggyllama/llama-13b is not the path to a directory containing a file named pytorch_model-00002-of-00003.bin.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'"""

# LLama = ['decapoda-research/llama-13b-hf', 'decapoda-research/llama-7b-hf'] 

# OSError: decapoda-research/llama-13b-hf is not a local folder or a valid repository name on 'https://hf.co'

# for 7b-hf: TypeError: not a string

# for asi at the 60th batch
"""RuntimeError: The size of tensor a (1024) must
 match the size of tensor b (1058) at non-singleton
  dimension 3"""
# Others


# ── PRNG / Abliterated model constants (moved from Bits_Generation.bits_tools) ──

PRNG_MODELS = [
    'numpy_default',
    # 'numpy_mt19937', 'numpy_pcg64', 'numpy_sfc64',
    # 'python_random', 'secrets', 'xor_shift',
    # 'lcg'  # excluded: degenerate output (1010... repeating)
]

ABLITERATED_MODELS = [
    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
    "failspy/llama-3-70B-Instruct-abliterated",  # actually not tested in original way
    "failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3",
    "huihui-ai/Qwen2.5-72B-Instruct-abliterated",  # actually not tested in original way
    "natong19/Qwen2-7B-Instruct-abliterated",
    "failspy/Phi-3-medium-4k-instruct-abliterated-v3",
    "dphn/dolphin-2.9.2-Phi-3-Medium-abliterated",
    "failspy/Phi-3-mini-128k-instruct-abliterated-v3",
]

ABLITERATED_MODELS_MAP_TO_ORIGINAL = {
    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3": 'meta-llama/Llama-3.1-8B-Instruct',
    # "failspy/llama-3-70B-Instruct-abliterated", actually not tested in original way
    "failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3": 'abacusai/Smaug-Llama-3-70B-Instruct',
    # "huihui-ai/Qwen2.5-72B-Instruct-abliterated":,
    "natong19/Qwen2-7B-Instruct-abliterated": 'Qwen/Qwen2-7B-Instruct',
    "failspy/Phi-3-medium-4k-instruct-abliterated-v3": 'microsoft/Phi-3-medium-4k-instruct',
    "dphn/dolphin-2.9.2-Phi-3-Medium-abliterated": 'microsoft/Phi-3-medium-128k-instruct',
    "failspy/Phi-3-mini-128k-instruct-abliterated-v3": 'microsoft/Phi-3-mini-128k-instruct',
}