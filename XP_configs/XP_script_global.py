
import json
import logging
import os
# os.environ['HF_HUB_OFFLINE'] = '0'

# os.environ['TRANSFORMERS_OFFLINE'] = '0' 
from pathlib import Path
# # Set cache directory - use absolute path
# CACHE_DIR = os.path.abspath('embedding_model_llmmap')

# # Create cache directory if it doesn't exist
# Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# # Set environment variables BEFORE importing transformers components
# os.environ['HF_HOME'] = CACHE_DIR
# os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
# os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

from audit_llm.system_utils import argparsing
from audit_llm.path_utils import get_repository_level_path
from audit_llm.experiment_config_schema import load_experiment_config
from audit_llm.Analysis_Classes import AuditionsAnalysis

logger = logging.getLogger(__name__)

if __name__=="__main__":

    args = { 
        'run_name': {
            'action': '--run_name',
            'type': str,
            'default': '',
        },
        'xp_suffix': {
            'action': '--xp_suffix',
            'type': str,
            'default': '',
        },
        'xp_config_path': {
            'action': '--xp_config_path',
            'type': str,
            'default': '0',
        },
    }
    
    parsed_args = argparsing(args)
    logger.debug("Here are the parsed_args:\n%s", parsed_args)

    xp_suffix=parsed_args['xp_suffix']
    RUN_NAME = parsed_args['run_name']
    Repo_level_path = get_repository_level_path()
    PRODUCTIONS_PATH = os.path.join(Repo_level_path, 'Productions/')
    
    run_path = os.path.join(PRODUCTIONS_PATH, RUN_NAME)
    Audition_analysis = AuditionsAnalysis(run_path=run_path)

    yaml_abs_path = Path(Repo_level_path) / parsed_args['xp_config_path']
    xp_config = load_experiment_config(yaml_abs_path)
    xp_config['_yaml_source_path'] = str(yaml_abs_path)

    # Suppressing models that did not work well for the run
    with open(Path(Repo_level_path) / 'Productions/Black_list_models_in_runs.json', 'r') as f:
        black_list_models_map = json.load(f)
    logger.info("Black list models are: %s", black_list_models_map.get(RUN_NAME.split('/')[0], []))
    
    config_name= parsed_args['xp_config_path'].split('/')[-1].split('.')[0]        
    xp_config['models_to_remove'] = black_list_models_map.get(RUN_NAME.split('/')[0], []) # Adding black listed models
    xp_config['xp_name'] = f"{config_name}_{xp_suffix}"

    
    Audition_analysis.run_xp(xp_config)