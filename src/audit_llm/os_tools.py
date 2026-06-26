# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Deprecated module — functions have been split into focused submodules.

Use the canonical submodules directly:
  - file_io      : write_data, erase_file, open_pickle_file, save_pickle_file,
                   load_json, write_json, write_dict_on_file, ini_to_dict,
                   compute_file_hash, write_per_model_parquet, ...
  - path_utils   : make_dir, make_exp_dir, make_path_and_create_folder,
                   get_repository_level_path, get_answers_dir, ...
  - data_transforms : revert_dictionary, preprocess_token_ids_col_of_answers,
                      nested_loop, sort_by_key, ...
  - system_utils : vllm_version_import_manager, argparsing, ...

Backward-compatibility re-exports are kept here for Fingerprinting_methods consumers
that cannot be updated in this phase.
"""

# Backward-compat re-exports for off-limits consumers (Fingerprinting_methods)
from audit_llm.data_transforms import *  # noqa: F401, F403
from audit_llm.file_io import *  # noqa: F401, F403
from audit_llm.path_utils import *  # noqa: F401, F403
from audit_llm.system_utils import *  # noqa: F401, F403
