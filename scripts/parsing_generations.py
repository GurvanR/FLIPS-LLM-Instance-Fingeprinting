import os, time
from audit_llm.path_utils import get_repository_level_path
from audit_llm.system_utils import argparsing, vllm_version_import_manager

args = { 
    'run_name': {
        'action': '--run_name',
        'type': str,
        'default': '', 
    },
    'hours_delay': { 
        'action': '--hours_delay',
        'type': float,
        'default': 0., 
    },
    'erase_previous_run': { 
        'action': '--erase_previous_run',
        'type': bool,
        'default': False, 
    },
    # if you're on the HPC cluster, changes the path of models.
    'cluster_mirror': {
        'action': '--cluster_mirror',
        'type': bool,
        'default': False, 
    },
    'merge_sub_run': { 
        'action': '--merge_sub_run',
        'type': bool,
        'default': False, 
    },
}

# Parse arguments
parsed_args = argparsing(args)
print("Here are the parsed_args:\n", parsed_args, "\n")
Repo_level_path=get_repository_level_path()

if parsed_args["hours_delay"] > 0:
        print(f"Delaying run by {parsed_args['hours_delay']} hours.")
        time.sleep(parsed_args["hours_delay"] * 3600)

# Merge sub_runs
if not parsed_args['run_name']:
    raise ValueError("need --run_name")

RUN_NAME = parsed_args['run_name'] # e.g. FLiPS-Monochar-0-1
PRODUCTIONS_PATH = os.path.join(Repo_level_path, 'Productions/')

from audit_llm.LLM_Classes.generation_parser import parsing_generations
parsing_generations(RUN_NAME, PRODUCTIONS_PATH, merge_sub_runs=parsed_args["merge_sub_run"], use_cluster_mirror=parsed_args['cluster_mirror'])