"""Single model audition analysis (extracted from Analysis_Classes.py).

This module contains the SingleModelAuditionAnalysis class which parses
LLM generations and saves them to parquet files.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Callable, List

import pandas as pd
from tqdm.auto import tqdm

from audit_llm.Bits_Generation.parsing_bits_tools import bits_token_pair_to_scrapper
from audit_llm.file_io import open_pickle_file, write_per_model_parquet
from audit_llm.LLM_Classes.General_LLM_Class import LLM_Generation
from audit_llm.LLM_Classes.model_tokenizer_map import model_tokenizer_map
from audit_llm.path_utils import make_path_and_create_folder


# Scrapper map constant
SCRAPPER_MAP = {
    "Bits_Datasets": bits_token_pair_to_scrapper,
}


class SingleModelAuditionAnalysis:
    """Analyzes generations from a single model and saves parsed results.
    
    This class handles the parsing of LLM output text into structured data
    (answers, token IDs, logprobs) and saves them to parquet files.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        Dataset_path: str,
        model_config: dict,
        Generations_save_path: str,
        Analysis_save_path: str,
        scrapping_rule: str,
        min_seq_length_inf: int,
        token_pairs_set: List[str],
        save_pro_and_improper: bool = True,
        mode: str = "vllm",
    ) -> None:
        """Initialize single model analysis.
        
        Args:
            model_name: Name of the model
            model_path: Path to model files
            Dataset_path: Path to the dataset CSV
            model_config: Model configuration dictionary
            Generations_save_path: Path to save generations
            Analysis_save_path: Path to save analysis results
            scrapping_rule: Rule for scrapping answers
            min_seq_length_inf: Minimum sequence length for inference
            token_pairs_set: List of token pair names
            save_pro_and_improper: Whether to save proper/improper generations
            mode: Backend mode ('vllm', 'HF', or 'openrouter')
        """
        self.save_pro_and_improper = save_pro_and_improper
        self.scrapping_rule = scrapping_rule
        self.min_seq_length_inf = min_seq_length_inf

        self.model_name = model_name
        self.Dataset_path = Dataset_path
        self.scrapper: Callable[[str, str], str] = SCRAPPER_MAP[self.Dataset_path.split("/")[-2]](
            min_seq_length=self.min_seq_length_inf
        )

        self.mode = mode
        if mode == "vllm":
            # Getting parameters of the vllm_model_config we want
            self.max_tokens = model_config["sampling_config"]["max_tokens"]
            # logprobs for vLLM
            self.logprobs_on: bool = model_config["sampling_config"]["logprobs"] is not None
        elif mode == "HF":
            # CRITICAL: AutoTokenizer import moved inside method body for lazy loading
            # to avoid crashing without transformers package (generation extras)
            from transformers import AutoTokenizer
            
            self.max_tokens = model_config["sampling_config"]["max_new_tokens"]
            # logprobs for HF
            self.tokenizer = model_tokenizer_map.get(model_name, AutoTokenizer).from_pretrained(
                model_path
            )
        elif mode == "openrouter":
            self.max_tokens = model_config["sampling_config"]["max_completion_tokens"]
        else:
            raise ValueError("Unknown mode")
            
        self.generations_path = make_path_and_create_folder(Generations_save_path, model_name)

        self.global_table_path = os.path.join(Analysis_save_path, "global_table.csv")
        self.pro_and_improper_generations_path = make_path_and_create_folder(
            os.path.join(Analysis_save_path, "Pro_and_improper_generations"), model_name
        )

        self.Analysis_save_path = Analysis_save_path
        self.pro_and_improper_header = [
            "QUESTION",
            "LLM GENERATED ANSWER",
            "IDENTIFIED ANSWER",
            "TEMPERATURE",
            "frequency_penalty",
            "system_prompt_idx",
        ]
        self.answers_header = [
            "Token_pair",
            "Model",
            "Dataset_Question Index",
            "gen_fail",
            "gen_counter",
            "Answer",
            "prompt_idx",
            "temperature",
            "frequency_penalty",
            "system_prompt_idx",
        ]

        self.token_pairs_set: List[str] = token_pairs_set
        self.dataset_name = "NO DATASET"
        self.execution_time = 1.0
        self.total_nb_inf = 1
        self.total_char_length = 1

    def analysis(self):
        """Analyze all terminated generation files for this model.

        Adds model to Answers.parquet with columns: Dataset, Model,
        Dataset_Question Index, Answer, temperature, Token_IDs,
        frequency_penalty, system_prompt_idx, token_pair
        """
        self.dataframe = pd.read_csv(self.Dataset_path)
        all_generations: List[LLM_Generation] = []
        for generations_path in Path(self.generations_path).iterdir():
            if "generations_terminated" in generations_path.name and generations_path.suffix in (
                ".parquet",
                ".pickle",
            ):
                self.single_generations_extraction(generations_path)
                all_generations.extend(self.generations)
        self.generations = all_generations
        if self.generations:
            self._save()

    def single_generations_extraction(self, generations_path: Path):
        """Extract and save answers from a single generation file.

        Args:
            generations_path: Path to the generation file (Parquet or pickle).
        """
        if generations_path.suffix == ".parquet":
            df = pd.read_parquet(generations_path)
            self.generations: List[LLM_Generation] = [
                LLM_Generation.from_row_dict(row) for row in df.to_dict(orient="records")
            ]
        else:
            self.generations = open_pickle_file(generations_path)

        logger.debug("(in single_generations_extraction) model_name=%s", self.model_name)

        if self.mode == "vllm":
            self.success_gen = [gen for gen in self.generations if gen.success]
            self.fail_gen = [gen for gen in self.generations if gen.fail]
            assert len(self.success_gen) + len(self.fail_gen) == len(self.generations)
            for LLM_gen in self.generations:
                output_text = LLM_gen.output_text
                if LLM_gen.token_pair == "no_token_pairs":
                    LLM_gen.scrapped_output = output_text
                else:
                    LLM_gen.scrapped_output = self.scrapper(output_text, LLM_gen.token_pair)
                LLM_gen.output_text = ""  # Free memory

        elif self.mode == "HF":  # Transformers
            raise NotImplementedError("HF analysis not implemented yet.")

        elif self.mode == "openrouter":
            raise NotImplementedError("openrouter analysis not implemented yet.")
        else:
            raise ValueError("Unknown mode")

    def _save(self):
        """Save answer map and optionally proper/improper generations."""
        self._save_answer_map()

        if self.save_pro_and_improper:
            # Saving **some** proper and improper answers
            pass

    def _save_answer_map(self):
        """Save parsed answers, token IDs, and logprobs as per-model Parquet."""
        records = []
        token_records = []
        logprobs_records = []

        df = self.dataframe
        cols_to_fetch = ["prompt_idx", "temperature", "frequency_penalty", "system_prompt_idx"]

        for gen in self.generations:
            # Get metadata as a dictionary
            metadata = {col: df.loc[gen.dataset_idx, col] for col in cols_to_fetch}

            base_fields = [gen.token_pair, self.model_name, gen.dataset_idx, gen.fail, gen.gen_counter]
            row_values = list(metadata.values())

            records.append(base_fields + [gen.scrapped_output, *row_values])
            token_records.append(base_fields + [gen.output_token_ids, *row_values])

            if self.logprobs_on:
                logprob_entry = {
                    "token_pair": gen.token_pair,
                    "model_name": self.model_name,
                    "dataset_idx": gen.dataset_idx,
                    "output_logprobs": gen.output_logprobs,
                    **metadata,
                }
                logprobs_records.append(logprob_entry)

        # Build DataFrames
        answers_header = self.answers_header
        token_ids_header = ["Token_IDs" if col_name == "Answer" else col_name for col_name in answers_header]

        self.single_model_answers_df = pd.DataFrame(records, columns=answers_header)
        token_ids_df = pd.DataFrame(token_records, columns=token_ids_header)

        # Write per-model Parquet files (no read-concat-overwrite)
        base = Path(self.Analysis_save_path)
        write_per_model_parquet(self.single_model_answers_df, base / "answers", self.model_name)
        write_per_model_parquet(token_ids_df, base / "token_ids", self.model_name)

        if self.logprobs_on and logprobs_records:
            logprobs_df = pd.DataFrame(logprobs_records)
            write_per_model_parquet(logprobs_df, base / "logprobs", self.model_name)
