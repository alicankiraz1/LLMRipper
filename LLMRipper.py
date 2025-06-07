#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLMRipper v2 – Interactive LLM Fine‑Tuner
----------------------------------------
New in this release (2025‑06‑07)
✓ Asks whether the user wants **LoRA‑based** (parameter‑efficient) or **Full** fine‑tuning *before* any other training hyper‑parameters are chosen. The rest of the workflow adapts automatically.
✓ Warns that datasets **must** follow the *System/User/Assistant* schema and verifies these columns exist.
✓ Prompts for dataset file‑format – **csv, json, jsonl, parquet** – and loads the file with the correct HuggingFace dataset loader.
✓ Improved validation & helpful error messages.
✓ Conditional questions (e.g. LoRA weight merge) only appear when relevant.
"""

import os
import sys
from huggingface_hub import login
from datasets import load_dataset, DatasetDict

from src.utils import print_banner
from src.dataset_helpers import load_local_dataset, ensure_columns
from src.config_handler import (
    get_tuning_strategy,
    get_model_info,
    get_dataset_info,
    get_hyperparameters,
    get_quantization_config,
    get_hub_push_config,
)
from src.model_loader import load_model_and_tokenizer
from src.processing import Processor
from src.trainer import train_model, save_and_merge_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Main interactive script
# ---------------------------------------------------------------------------

def main():
    print_banner()

    # --- Configuration State Machine ---
    config = {}
    steps = [
        'tuning_strategy', 
        'model_info', 
        'dataset_info', 
        'hyperparameters', 
        'quantization_config'
    ]
    current_step_index = 0

    while current_step_index < len(steps):
        current_step = steps[current_step_index]

        if current_step == 'tuning_strategy':
            tuning_type = get_tuning_strategy()
            config['tuning_type'] = tuning_type
            config['is_lora'] = (tuning_type == "lora")

        elif current_step == 'model_info':
            model_privacy, hf_token, base_model_name = get_model_info()
            if model_privacy == "BACK":
                current_step_index -= 1
                continue
            config.update({'model_privacy': model_privacy, 'hf_token': hf_token, 'base_model_name': base_model_name})
            if hf_token:
                login(token=hf_token, add_to_git_credential=True)

        elif current_step == 'dataset_info':
            dataset_source, dataset_format, dataset_path, dataset_repo, hf_token = get_dataset_info(config.get('hf_token'))
            if dataset_source == "BACK":
                current_step_index -= 1
                continue
            config.update({
                'dataset_source': dataset_source, 'dataset_format': dataset_format, 
                'dataset_path': dataset_path, 'dataset_repo': dataset_repo
            })
            if hf_token:
                config['hf_token'] = hf_token

        elif current_step == 'hyperparameters':
            max_length, grad_acc_steps, per_device_bs, num_epochs = get_hyperparameters()
            if max_length == "BACK":
                current_step_index -= 1
                continue
            config.update({
                'max_length': max_length, 'grad_acc_steps': grad_acc_steps, 
                'per_device_bs': per_device_bs, 'num_epochs': num_epochs
            })

        elif current_step == 'quantization_config':
            quantize_choice, bit_choice, precision_choice = get_quantization_config(config['tuning_type'])
            if quantize_choice == "BACK":
                current_step_index -= 1
                continue
            config.update({
                'quantize_choice': quantize_choice, 'bit_choice': bit_choice, 
                'precision_choice': precision_choice
            })

        current_step_index += 1

    # --- Load Dataset ---
    try:
        if config['dataset_source'] == "local":
            raw_datasets = load_local_dataset(config['dataset_path'], config['dataset_format'])
        else: # huggingface
            load_args = {"token": config.get('hf_token')} if config.get('hf_token') else {}
            raw_datasets = load_dataset(config['dataset_repo'], **load_args)
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if len(raw_datasets) == 1 and "train" in raw_datasets:
        split_ds = raw_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
        raw_datasets = DatasetDict({"train": split_ds["train"], "validation": split_ds["test"]})

    try:
        ensure_columns(raw_datasets)
    except ValueError as e:
        print(f"Dataset validation error: {e}")
        sys.exit(1)

    # --- Load Model and Tokenizer ---
    model, tokenizer, is_lora = load_model_and_tokenizer(
        config['base_model_name'], config['model_privacy'], config.get('hf_token'), 
        config['quantize_choice'], config['bit_choice'], config['precision_choice'], 
        config['is_lora']
    )
    config['is_lora'] = is_lora # Update in case it was changed during loading

    # --- Process Dataset ---
    print("Tokenising dataset…")
    processor = Processor(tokenizer, config['max_length'])
    tokenised_train = raw_datasets["train"].map(processor.preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)
    tokenised_val = raw_datasets["validation"].map(processor.preprocess_function, batched=True, remove_columns=raw_datasets["validation"].column_names)

    # --- Train Model ---
    trainer = train_model(
        model, tokenised_train, tokenised_val, 
        config['num_epochs'], config['per_device_bs'], 
        config['grad_acc_steps'], config['precision_choice']
    )

    # --- Save, Merge, and Push ---
    merge_weights, repo_id = get_hub_push_config(config['is_lora'], config.get('hf_token'))
    save_and_merge_model(trainer, tokenizer, config['is_lora'], config.get('hf_token'), merge_weights, repo_id)

    print("All done – happy prompting! ✨")


if __name__ == "__main__":
    main()
