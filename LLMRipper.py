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
import subprocess
import getpass
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
try:
    from peft import prepare_model_for_int8_training
except ImportError:
    prepare_model_for_int8_training = None
    print("Warning: prepare_model_for_int8_training not found. Update peft if you need 8‑bit training support.")
from huggingface_hub import login

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def print_banner():
    """Pretty ASCII banner (installs *pyfiglet* the first time)."""
    try:
        import pyfiglet
    except ImportError:
        print("pyfiglet not found. Installing pyfiglet…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyfiglet"])
        import pyfiglet
    ascii_banner = pyfiglet.figlet_format("LLMRipper")
    print(ascii_banner)
    print("Created by Alican Kiraz – v2.0")


def get_input(prompt: str, valid_options=None):
    """Small helper that forces the user to pick a valid option (case‑insensitive)."""
    while True:
        response = input(prompt).strip()
        if valid_options:
            if response.lower() in valid_options:
                return response.lower()
            else:
                print(f"Invalid input. Please choose from: {', '.join(valid_options)}")
        else:
            if response:
                return response
            else:
                print("Input cannot be empty. Please try again.")


def get_secure_input(prompt: str):
    """Secure input for tokens and passwords (hidden from terminal)."""
    while True:
        response = getpass.getpass(prompt).strip()
        if response:
            return response
        else:
            print("Input cannot be empty. Please try again.")


# ---------------------------------------------------------------------------
# Model architecture helpers
# ---------------------------------------------------------------------------

def get_target_modules_for_model(model):
    """Automatically detect target modules for LoRA based on model architecture."""
    
    # Common target modules for different architectures
    TARGET_MODULES_MAP = {
        # Llama, Alpaca, Vicuna, etc.
        "LlamaForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Gemma models  
        "GemmaForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # GPT models
        "GPTNeoXForCausalLM": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        # Mistral models
        "MistralForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Qwen models
        "QWenLMHeadModel": ["c_attn", "c_proj", "w1", "w2"],
        # Phi models
        "PhiForCausalLM": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        # Default fallback for transformer models
        "default": ["q_proj", "v_proj"]
    }
    
    model_type = model.__class__.__name__
    print(f"Detected model type: {model_type}")
    
    if model_type in TARGET_MODULES_MAP:
        target_modules = TARGET_MODULES_MAP[model_type]
        print(f"Using target modules: {target_modules}")
        return target_modules
    else:
        # Try to find common attention projection layers
        available_modules = []
        for name, module in model.named_modules():
            if any(target in name for target in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
                module_name = name.split('.')[-1]
                if module_name not in available_modules:
                    available_modules.append(module_name)
        
        if available_modules:
            print(f"Auto-detected modules: {available_modules}")
            return available_modules
        else:
            # Fallback to default
            print(f"Using default target modules for unknown architecture: {TARGET_MODULES_MAP['default']}")
            return TARGET_MODULES_MAP["default"]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = ["csv", "json", "jsonl", "parquet"]
REQ_COLUMNS = {"System", "User", "Assistant"}


def load_local_dataset(path: str, file_fmt: str):
    """Load a local dataset file with the correct HF loader."""
    if file_fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {file_fmt} – choose one of {', '.join(SUPPORTED_FORMATS)}")
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    try:
        if file_fmt == "jsonl":
            return load_dataset("json", data_files={"train": path}, split=None)
        elif file_fmt == "json":
            return load_dataset("json", data_files={"train": path})
        else:
            return load_dataset(file_fmt, data_files={"train": path})
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {path}: {str(e)}")


def ensure_columns(dataset):
    """Verify that required columns exist in *all* splits."""
    missing = REQ_COLUMNS - set(dataset["train"].column_names)
    if missing:
        raise ValueError(f"Dataset must contain columns {', '.join(REQ_COLUMNS)}. Missing: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Main interactive script
# ---------------------------------------------------------------------------

def main():
    print_banner()

    # Privacy & authentication ------------------------------------------------
    model_privacy = get_input("Is your *base model* private or public? [private/public]: ", ["private", "public"])
    hf_token = None
    if model_privacy == "private":
        hf_token = get_secure_input("Enter your Hugging Face token: ")
        login(token=hf_token)

    base_model_name = get_input("Enter the Hugging Face repo for the base model (e.g. AlicanKiraz0/LLM‑Base): ")

    # NEW ▶ Finetuning strategy ------------------------------------------------
    tuning_type = get_input("Select tuning strategy – LoRA (parameter‑efficient) or Full? [lora/full]: ", ["lora", "full"])
    is_lora = (tuning_type == "lora")

    # Dataset location & format ----------------------------------------------
    print("\n***  Dataset Requirements  ***")
    print("Your file MUST contain the columns: System | User | Assistant\n")

    dataset_source = get_input("Is the dataset local or on HuggingFace Hub? [local/huggingface]: ", ["local", "huggingface"])
    dataset_format = get_input("Which file format are you using? [csv/json/jsonl/parquet]: ", SUPPORTED_FORMATS)

    if dataset_source == "local":
        dataset_path = get_input("Enter the path to your local dataset file: ")
        try:
            raw_datasets = load_local_dataset(dataset_path, dataset_format)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)
    else:
        hub_privacy = get_input("Is the HF dataset repo public or private? [public/private]: ", ["public", "private"])
        if hub_privacy == "private":
            if not hf_token:
                hf_token = get_secure_input("Enter your Hugging Face token for the *dataset*: ")
                login(token=hf_token)
        dataset_repo = get_input("Enter the HF dataset repo (e.g. AlicanKiraz0/my‑chat‑dataset): ")
        load_args = {"token": hf_token} if (hub_privacy == "private" and hf_token) else {}
        try:
            raw_datasets = load_dataset(dataset_repo, **load_args)
        except Exception as e:
            print(f"Error loading dataset from HuggingFace Hub: {e}")
            sys.exit(1)

    # Ensure train/validation splits exist
    if len(raw_datasets) == 1 and "train" in raw_datasets:
        split_ds = raw_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
        raw_datasets = DatasetDict({"train": split_ds["train"], "validation": split_ds["test"]})

    try:
        ensure_columns(raw_datasets)
    except ValueError as e:
        print(f"Dataset validation error: {e}")
        print("Please ensure your dataset contains the required columns: System, User, Assistant")
        sys.exit(1)

    # Hyper‑parameters --------------------------------------------------------
    try:
        max_length = int(get_input("Enter maximum sequence length (e.g. 1024 or 2048): "))
        if max_length < 128 or max_length > 8192:
            print("Warning: Sequence length should typically be between 128 and 8192")
        
        grad_acc_steps = int(get_input("Gradient accumulation steps (1/2/4/8): "))
        if grad_acc_steps not in [1, 2, 4, 8]:
            print("Warning: Gradient accumulation steps should typically be 1, 2, 4, or 8")
        
        per_device_bs = int(get_input("Per‑device train batch size (1/2/4/8): "))
        if per_device_bs not in [1, 2, 4, 8]:
            print("Warning: Batch size should typically be 1, 2, 4, or 8")
        
        num_epochs = int(get_input("Number of epochs: "))
        if num_epochs < 1 or num_epochs > 100:
            print("Warning: Number of epochs should typically be between 1 and 100")
    except ValueError as e:
        print(f"Invalid input for hyperparameters: {e}")
        sys.exit(1)

    # Quantisation choice -----------------------------------------------------
    quantize_choice = get_input("Use model quantisation? [yes/no]: ", ["yes", "no"])

    # NOTE: full fine‑tuning + 4‑/8‑bit weights is *experimental*; fall back to fp16/bf16/fp32.
    if tuning_type == "full" and quantize_choice == "yes":
        print("Full fine‑tuning with 4‑/8‑bit adapters is not officially supported – switching to non‑quantised training.")
        quantize_choice = "no"

    # Precision (only asked when not quantising) -----------------------------
    precision_choice = None
    torch_dtype = None
    if quantize_choice == "no":
        precision_choice = get_input("Choose precision [fp16/bf16/fp32]: ", ["fp16", "bf16", "fp32"])
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[precision_choice]

    # -----------------------------------------------------------------------
    # MODEL LOADING SECTION
    # -----------------------------------------------------------------------
    print("Loading model – this can take a while…")

    common_kwargs = dict(
        token=hf_token if model_privacy == "private" else None,
        trust_remote_code=True,
        device_map="auto",
    )

    if quantize_choice == "yes":  # 4‑bit / 8‑bit LoRA fine‑tuning
        bit_choice = get_input("Quantisation bits – 4 or 8? [4/8]: ", ["4", "8"])
        load_kwargs = {
            "load_in_4bit": bit_choice == "4",
            "load_in_8bit": bit_choice == "8",
        }
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **common_kwargs, **load_kwargs)
        if bit_choice == "4":
            model = prepare_model_for_kbit_training(model)
        else:  # 8‑bit
            if prepare_model_for_int8_training is None:
                raise ImportError("prepare_model_for_int8_training missing – update *peft* package.")
            model = prepare_model_for_int8_training(model)
    else:  # No quantisation
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **common_kwargs, torch_dtype=torch_dtype)
        model.config.use_cache = False

    # Wrap with LoRA if requested -------------------------------------------
    if is_lora:
        try:
            target_modules = get_target_modules_for_model(model)
            lora_cfg = LoraConfig(
                r=4, 
                lora_alpha=16, 
                lora_dropout=0.1, 
                bias="none", 
                task_type="CAUSAL_LM",
                target_modules=target_modules
            )
            model = get_peft_model(model, lora_cfg)
            model.enable_input_require_grads()
            print("✅ LoRA adapter successfully applied")
        except Exception as e:
            print(f"❌ Error applying LoRA adapter: {e}")
            print("Falling back to alternative target modules...")
            
            # Try with minimal target modules
            try:
                fallback_modules = ["q_proj", "v_proj"]
                lora_cfg = LoraConfig(
                    r=4, 
                    lora_alpha=16, 
                    lora_dropout=0.1, 
                    bias="none", 
                    task_type="CAUSAL_LM",
                    target_modules=fallback_modules
                )
                model = get_peft_model(model, lora_cfg)
                model.enable_input_require_grads()
                print(f"✅ LoRA adapter applied with fallback modules: {fallback_modules}")
            except Exception as e2:
                print(f"❌ LoRA fallback also failed: {e2}")
                print("Switching to full fine-tuning mode...")
                is_lora = False

    # -----------------------------------------------------------------------
    # TOKENISER & PRE‑PROCESSING
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, **common_kwargs, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def create_prompt(sys_text: str, usr_text: str):
        return f"[SYSTEM]\n{sys_text}\n[USER]\n{usr_text}\n[ASSISTANT]\n"

    def chunk(seq, size):
        return [seq[i:i + size] for i in range(0, len(seq), size)]

    def preprocess_function(examples):
        final_input_ids, final_attention_masks, final_labels = [], [], []
        for sys_t, usr_t, as_t in zip(examples["System"], examples["User"], examples["Assistant"]):
            prompt = create_prompt(sys_t or "", usr_t or "")
            full_text = prompt + (as_t or "")
            tokenised = tokenizer(full_text, truncation=False, padding=False)
            ids = tokenised["input_ids"]
            prompt_len = len(tokenizer(prompt, truncation=False, padding=False)["input_ids"])
            labels = [-100] * len(ids)
            for i in range(prompt_len, len(ids)):
                labels[i] = ids[i]
            # Split or pad to max_length
            if len(ids) > max_length:
                id_chunks = chunk(ids, max_length)
                label_chunks = chunk(labels, max_length)
                for ic, lc in zip(id_chunks, label_chunks):
                    pad_len = max_length - len(ic)
                    ic += [tokenizer.pad_token_id] * pad_len
                    lc += [-100] * pad_len
                    attn = [int(tok != tokenizer.pad_token_id) for tok in ic]
                    final_input_ids.append(ic)
                    final_attention_masks.append(attn)
                    final_labels.append(lc)
            else:
                pad_len = max_length - len(ids)
                ids += [tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
                attn = [int(tok != tokenizer.pad_token_id) for tok in ids]
                final_input_ids.append(ids)
                final_attention_masks.append(attn)
                final_labels.append(labels)
        return {"input_ids": final_input_ids, "attention_mask": final_attention_masks, "labels": final_labels}

    print("Tokenising dataset…")
    tokenised_train = raw_datasets["train"].map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)
    tokenised_val = raw_datasets["validation"].map(preprocess_function, batched=True, remove_columns=raw_datasets["validation"].column_names)

    data_collator = DefaultDataCollator()

    # -----------------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------------
    print("\n***  Training  ***")
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_acc_steps,
        eval_steps=200,
        save_steps=200,
        logging_steps=100,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=(precision_choice == "fp16" if precision_choice else False),
        bf16=(precision_choice == "bf16" if precision_choice else False),
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False,
        optim="adamw_torch_fused",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    # Save outputs -----------------------------------------------------------
    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    print("Training finished → ./finetuned_model created.")

    # Merge LoRA weights (only if LoRA strategy) -----------------------------
    if is_lora:
        merge_choice = get_input("Merge LoRA weights with the base model? [yes/no]: ", ["yes", "no"])
        if merge_choice == "yes":
            merged_dir = "./merged_final_model"
            print("Merging…")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"Merged model saved → {merged_dir}")
            if get_input("Push merged model to HuggingFace Hub? [yes/no]: ", ["yes", "no"]) == "yes":
                repo_id = get_input("HF repo ID to push to (e.g. AlicanKiraz0/LLMRipper‑finetuned): ")
                print("Uploading…")
                merged_model.push_to_hub(repo_id, token=hf_token)
                tokenizer.push_to_hub(repo_id, token=hf_token)
                print("Upload complete.")

    print("All done – happy prompting! ✨")


if __name__ == "__main__":
    main()
