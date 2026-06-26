"""High-level analysis entry point.

Exposes :class:`AuditionsAnalysis`, which loads a production run's ``run_config``
and drives the two analysis stages: parsing raw generations into per-model
Parquet (:meth:`model_analysis`) and running a classification experiment over
the parsed answers (:meth:`run_xp`).
"""

import logging
import os
from pathlib import Path
from typing import List

from tqdm.auto import tqdm

from audit_llm.Bits_Generation.Token_Pairs_Sets import TOKEN_PAIRS_SETS_DICT

logger = logging.getLogger(__name__)
from audit_llm.data_loader import _analysis_mode
from audit_llm.experiment_runner import run_xp
from audit_llm.file_io import load_json
from audit_llm.path_utils import get_repository_level_path


class AuditionsAnalysis:
    """High-level analysis driver for a production run.

    Loads the run's ``run_config`` and exposes the two analysis stages:
    :meth:`model_analysis` (parse raw generations into per-model Parquet) and
    :meth:`run_xp` (run a classification experiment over the parsed answers).
    """

    def __init__(self, run_path: str, show_config_infos: bool = False) -> None:
        """Initialize analysis from run_path.

        Args:
            run_path: Path to the production run directory
            show_config_infos: Whether to print config information
        """
        # Load run_config (JSON-first with pickle fallback)
        run_config_json_path = Path(run_path) / "run_config.json"
        run_config_pickle_path = Path(run_path) / "run_config.pickle"

        if run_config_json_path.exists():
            run_config = load_json(path=run_config_json_path)
        elif run_config_pickle_path.exists():
            import pickle
            with open(str(run_config_pickle_path), "rb") as f:
                run_config = pickle.load(f)
        else:
            raise FileNotFoundError(f"No run_config found in {run_path}")

        logger.debug("run_config = %s", run_config)
        self.use_cluster_mirror = False

        if show_config_infos:
            logger.debug("VLLM models:")
            for vllm_model, is_done in run_config["vllm_models"].items():
                logger.debug("%s, is done: %s", vllm_model, is_done)

            logger.debug("HF models:")
            for hf_model, is_done in run_config["hf_models"].items():
                logger.debug("%s, is done: %s", hf_model, is_done)

            logger.debug("OpenRouter models:")
            for openrouter_model, is_done in run_config.get("openrouter_models", {}).items():
                logger.debug("%s, is done: %s", openrouter_model, is_done)

        # For data retrieving. We take models that have been processed.
        self.scrapping_rule = run_config["scrapping_rule"]

        self.vllm_models = [model_name for model_name, is_done in run_config["vllm_models"].items() if is_done]
        self.vllm_model_path = run_config["vllm_model_path"]
        self.vllm_model_config = run_config["vllm_model_config"]

        self.hf_models = [model_name for model_name, is_done in run_config["hf_models"].items() if is_done]
        self.hf_model_path = run_config["hf_model_path"]
        self.hf_model_config = run_config["hf_model_config"]

        self.openrouter_models = [
            model_name for model_name, is_done in run_config.get("openrouter_models", {}).items() if is_done
        ]
        self.openrouter_model_path = run_config.get("openrouter_model_path", "")
        self.openrouter_model_config = run_config.get("openrouter_model_config", {})

        self.min_seq_length_inf = run_config["min_seq_length"]
        self.max_tokens = run_config["MAX_TOKENS"]

        self.Dataset_path = os.path.join(get_repository_level_path(), run_config["Dataset_relative_path"])
        self.token_pairs_set: List[str] = TOKEN_PAIRS_SETS_DICT[run_config["TOKEN_PAIRS_SET"]]

        # For Analysis
        self.audited_models = self.vllm_models + self.hf_models + self.openrouter_models

        self.Analysis_save_path = os.path.join(run_path, "Analysis/")
        self.Generations_save_path = os.path.join(run_path, "Generations/")
        self.Experiments_path = os.path.join(run_path, "Experiments/")

        os.makedirs(self.Analysis_save_path, exist_ok=True)

        self.save_fig_path = ""

        # These will be set by _analysis_mode()
        self.Answers_df = None
        self.TokenIDs_df = None
        self.MainDataset_df = None
        self.Answers_df_path = None

    def run_xp(self, xp_config: dict, hard_datasets: list[str] = []):
        """Run experiment with given configuration.

        This method wraps the refactored run_xp() function from experiment_runner.py.

        Args:
            xp_config: Experiment configuration dictionary
            hard_datasets: List of hard datasets (optional)
        """
        # Call refactored _analysis_mode to load data
        logger.info("doing analysis")
        self.Answers_df, self.TokenIDs_df, self.MainDataset_df, self.Answers_df_path = _analysis_mode(
            Analysis_save_path=self.Analysis_save_path,
            scrapping_rule=self.scrapping_rule,
            Dataset_path=self.Dataset_path,
            seed=None,
        )
        logger.info("analysis done")

        # Call the refactored standalone run_xp function from experiment_runner.py
        run_xp(
            xp_config=xp_config,
            Answers_df=self.Answers_df,
            TokenIDs_df=self.TokenIDs_df,
            MainDataset_df=self.MainDataset_df,
            token_pairs_set=self.token_pairs_set,
            Dataset_path=self.Dataset_path,
            Experiments_path=self.Experiments_path,
            hard_datasets=hard_datasets,
        )

    def model_analysis(self):
        """Parse generations into per-model Parquet files.

        Depends on the scrapping_rule used to decide to which answer
        corresponds the generated texts.
        """
        from audit_llm.data_loader import _set_environment
        from audit_llm.file_io import sanitize_model_name
        from audit_llm.model_analysis import SingleModelAuditionAnalysis
        from audit_llm.path_utils import _is_cluster_mirror_path

        Analysis_save_path = os.path.join(self.Analysis_save_path, self.scrapping_rule)
        _set_environment(Analysis_save_path, self.Experiments_path)

        answers_dir = Path(Analysis_save_path) / "answers"

        # --- HF models ---
        for model_name in tqdm(self.hf_models, desc="hf_models", leave=True):
            model_path = self.hf_model_path[model_name]
            if (_is_cluster_mirror_path(model_path) and self.use_cluster_mirror) or (not _is_cluster_mirror_path(model_path) and not self.use_cluster_mirror):
                raise NotImplementedError("HF model analysis not implemented yet.")

        # --- vLLM models ---
        for model_name in tqdm(self.vllm_models, desc="vllm_models", leave=True):
            logger.info("Parsing model: %s", model_name)

            # Skip check: per-model file existence
            sanitized = sanitize_model_name(model_name)
            model_parquet = answers_dir / f"{sanitized}.parquet"
            if model_parquet.exists():
                logger.info("Model %s already parsed, skipping.", model_name)
                continue

            model_path = self.vllm_model_path[model_name]
            audition = SingleModelAuditionAnalysis(
                model_name=model_name,
                model_path=model_path,
                Dataset_path=self.Dataset_path,
                model_config=self.vllm_model_config,
                Generations_save_path=self.Generations_save_path,
                Analysis_save_path=Analysis_save_path,
                scrapping_rule=self.scrapping_rule,
                min_seq_length_inf=self.min_seq_length_inf,
                token_pairs_set=self.token_pairs_set,
                mode="vllm",
            )
            audition.analysis()

        # --- OpenRouter models ---
        for model_name in tqdm(self.openrouter_models, desc="openrouter_models", leave=True):
            raise NotImplementedError("OpenRouter model analysis not implemented yet.")


__all__ = ["AuditionsAnalysis"]
