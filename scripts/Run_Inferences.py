# python scripts/Run_Inferences.py --dataset Toy_ds --model mistral_small_instruct_v0.1 --sub_run test_code_chat_temp
# python scripts/Run_Inferences.py --dataset SPImpact --model Qwen/Qwen2.5-7B-Instruct --sub_run QWEN_test1
# mistralai/Mistral-7B-Instruct-v0.3
# meta-llama/Llama-3.1-8B-Instruct
if __name__=="__main__":
    
    import os
    import shutil
    from audit_llm.system_utils import argparsing, vllm_version_import_manager

    # Define the argument configuration
    args = {
        'device': {
            'action': '--device',
            'type': str,
            'default': '0',
        },
        'dataset': {
            'action': '--dataset',
            'type': str,
            'default': 'Toy_example', 
        },
        # If test is false, the logs will go in a separate file and an email will be sent to assess the success of the run.
        'test': { 
            'action': '--test',
            'type': bool,
            'default': False, 
        },
        # Delays the experiment start.
        'hours_delay': { 
            'action': '--hours_delay',
            'type': float,
            'default': 0., 
        },
        # Without, running the code will try to continue the run at a checkpoint if any.
        # With, it will begin the run from scratch and erase previous run data if any.
        'erase_previous_run': { 
            'action': '--erase_previous_run',
            'type': bool,
            'default': False, 
        },
        'gpu':{
            'action': '--gpu',
            'type': int,
            'default': 1, 
        },
        'parse_gen': { 
            'action': '--parse_gen',
            'type': bool,
            'default': False, 
        },
        'model': {
            'action': '--model',
            'type': str,
            'default': '', 
        },
        # Name of the run of a certain set of model. For one run you have as many sub runs as there are model sets.
        'sub_run': {
            'action': '--sub_run',
            'type': str,
            'default': 'default', 
        },
        'bs':{
            'action': '--bs',
            'type': int,
            'default': 20, 
        },
        'openrouter': { 
            'action': '--openrouter',
            'type': bool,
            'default': False, 
        },
        'seed': {
            'action': '--seed',
            'type': int,
            'default': 42,
        },
    }

    # Parse arguments
    parsed_args = argparsing(args)
    print("Here are the parsed_args:\n", parsed_args, "\n")
    device = parsed_args['device']

    # These two os.environ lines must be set before importing torch/vllm; otherwise GPU assignment errors occur.
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    from audit_llm.LLM_Classes.inference_runner import multi_model_infer, run_inferences
    from audit_llm.LLM_Classes.generation_parser import parsing_generations
    from audit_llm.LLM_Classes.run_config import make_run_config
    from audit_llm.models_management.model_names import *
    from audit_llm.path_utils import get_repository_level_path

    ### Tunable arguments ### 
    # Careful to not change this script location or the paths solving won't work.
    
    # Inference configuration
    ## Opening yaml file to get the inference configurations in get_repository_level_path()/scripts/Inference_configs.yaml
    with open( os.path.join( get_repository_level_path(), 'scripts', 'Inference_configs.yaml' ), 'r') as file:
        import yaml
        Inf_configs_dict = yaml.safe_load(file)

    Inf_config = Inf_configs_dict[parsed_args['dataset']]
    
    local_cache=False
    ###############
    # TODO: load model groups (with quantization variants) from config/models.yaml

    if parsed_args['model']:
        model_names=[parsed_args['model']]
        working_lib=None
        quantization_map = {mn: 'no_quantized' for mn in model_names}
    else:
        model_names = OR_4 # OpenRouterCheapest # OR_selection_2  #+ OPT #+ ELEUTHER_AI
        working_lib=None
        quantization_map = {mn: 'no_quantized' for mn in model_names}
    
    Logits: bool = False # TODO: logit post-processing not yet implemented.

    #########################

    if parsed_args['test']:
        Production_folder="Test_Runs"

        mail=False
    else:
        Production_folder="Normal_Runs"
        mail=False  # email notifications disabled

    if parsed_args['dataset']=='Toy_example':
        Production_folder='Toy_Runs'
        mail=False

    model_cache_path = os.environ.get("AUDIT_LLM_MODEL_CACHE")

    # Construct the dataset path
    Dataset_relative_path=os.path.join("datasets", "Bits_Datasets", f"{parsed_args['dataset']}.csv")

    Productions_relative_path = f'Productions/Graph_Productions/{Production_folder}'
    hours_delay = parsed_args['hours_delay'] 

    # max_seq_len_to_capture for vllm > 0.4.0
    max_context = vllm_version_import_manager('max_context')
    vllm_model_config = {
        'model_only_config': {
            'dtype': "bfloat16",
            'gpu_memory_utilization': 0.9, # 0.9 is a solid default for vLLM GPU memory utilization on A100.
            'trust_remote_code': True,
            max_context : 2500,  # controls max input token length only.
            'max_model_len': Inf_config['max_model_len'],
            'tensor_parallel_size': parsed_args['gpu'],        
        },
        'sampling_config': {
            #'temperature': Inf_config['TEMPERATURES'],
            'top_k': Inf_config['TOP_K'],
            'max_tokens': Inf_config['MAX_TOKENS'], # Maximum number of tokens to generate per output sequence. (from vllm doc)
            'logprobs': Inf_config.get('logprobs', None),
        }
    }

    hf_model_config = {
            "device": f"cuda:{device}",
            #  bs: batch_size, contrary to vllm, you have to choose manually the batch_size,
            #  when set too high it yields errors, and if too low, your inferences will go slowly.
            #  batch size 5 is a conservative default; raise it for higher-memory GPUs.
            "bs":parsed_args['bs'],
            "gpu": parsed_args['gpu'],

            'model_only_config' : {
                "trust_remote_code": True,
                "device_map" : 'auto',
            },

            'sampling_config' : {
                "max_new_tokens": Inf_config['MAX_TOKENS'], # maximum number of generated tokens.
                "do_sample": True,
                #"temperature": Inf_config['TEMPERATURES'],
                "top_k": Inf_config['TOP_K'],
                "repetition_penalty": 1.05, # default value.
                "output_logits": True if Logits else False,  # Set to True if you need to have the logits in the output
                "return_dict_in_generate": True if Logits else False, # Set to True if you need to have the logits in the output
            }        
        }
    
    open_router_config={
        'sampling_config': {
            'max_completion_tokens': Inf_config['MAX_TOKENS'],
            #"temperature": Inf_config['TEMPERATURES'],
        }
    }

    scrapping_rule = 'Graph_Numeric' # necessary for the code, but currently the code use dataset_to_scrapper in parsing_graph_tools.py, regardless the scrapping_rule name


    run_name=parsed_args['dataset'] + f"_run/{parsed_args['sub_run']}"

    repository_level = get_repository_level_path()
    Productions_path= os.path.join(repository_level, Productions_relative_path)

    run_path=os.path.join(Productions_path, run_name)


    if os.path.exists(run_path) and parsed_args['erase_previous_run']:  # Using .get() to avoid KeyError
            shutil.rmtree(run_path)  # Deletes the folder and its contents

    os.makedirs(run_path, exist_ok=True)  # Recreate the directory if erased.

    # Snapshot configurations for reproducibility
    shutil.copy(
        os.path.join(get_repository_level_path(), 'scripts', 'Inference_configs.yaml'),
        os.path.join(run_path, 'Inference_configs.yaml')
    )
    make_run_config(scrapping_rule, run_name, Inf_config, models_configs,
                    vllm_model_config, hf_model_config, open_router_config,
                    Dataset_relative_path, Productions_relative_path, hours_delay,
                    model_names=model_names, model_cache_path=model_cache_path,
                    outlines=False, working_lib=working_lib, ALL_OPEN_ROUTER=parsed_args['openrouter'],
                    quantization_map=quantization_map,
                    ) 

    def run():
        multi_model_infer(run_name, Productions_path)

    run_inferences(run_name, Productions_path, run)

    if parsed_args['parse_gen']:
        parsing_generations(run_name, Productions_path)


