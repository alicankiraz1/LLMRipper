import os
import sys
import subprocess
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
try:
    from peft import prepare_model_for_int8_training
except ImportError:
    prepare_model_for_int8_training = None
    print("Warning: prepare_model_for_int8_training not found. Please update peft if you want to use 8-bit quantization.")
from huggingface_hub import login

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def print_banner():
    try:
        import pyfiglet
    except ImportError:
        print("pyfiglet not found. Installing pyfiglet...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyfiglet"])
        import pyfiglet
    ascii_banner = pyfiglet.figlet_format("LLMRipper")
    print(ascii_banner)
    print("Created by Alican Kiraz")  # Normal yazı şeklinde alt metin

def get_input(prompt, valid_options=None):
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

def main():
    print_banner()
    
    model_type = get_input("Please specify if your model is private or public [private/public]: ", valid_options=["private", "public"])
    hf_token = None
    if model_type == "private":
        hf_token = get_input("Please enter your Hugging Face token: ")
        login(token=hf_token)
    base_model_name = get_input("Enter the Hugging Face repository name for the model (e.g., AlicanKiraz0/SenecaLLM_x_Qwen2.5-7B-CyberSecurity): ")

    dataset_choice = get_input("Is the dataset local or from Hugging Face? [local/huggingface]: ", valid_options=["local", "huggingface"])
    if dataset_choice == "local":
        dataset_path = get_input("Enter the path to your local dataset: ")
        raw_datasets = load_dataset("json", data_files={"train": dataset_path})
        if len(raw_datasets) == 1 and "train" in raw_datasets:
            split_dataset = raw_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
            raw_datasets = DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})
    else:
        hf_dataset_privacy = get_input("Is the Hugging Face dataset public or private? [public/private]: ", valid_options=["public", "private"])
        if hf_dataset_privacy == "private":
            if model_type == "private":
                same_token = get_input("Can we use the same token you provided earlier for accessing the dataset? [yes/no]: ", valid_options=["yes", "no"])
                if same_token != "yes":
                    hf_token = get_input("Please enter a new Hugging Face token: ")
                    login(token=hf_token)
            else:
                hf_token = get_input("Please enter your Hugging Face token: ")
                login(token=hf_token)
        dataset_repo = get_input("Enter the Hugging Face repository name for the dataset (e.g., AlicanKiraz0/05-01-CyberSec-Cencored): ")
        if hf_dataset_privacy == "private" and hf_token is not None:
            raw_datasets = load_dataset(dataset_repo, token=hf_token)
        else:
            raw_datasets = load_dataset(dataset_repo)
        if len(raw_datasets) == 1 and "train" in raw_datasets:
            split_dataset = raw_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
            raw_datasets = DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})

    max_length = int(get_input("Enter maximum sequence length (e.g., 1024 or 2048): "))
    gradient_accumulation_steps = int(get_input("Enter gradient accumulation steps (e.g., 1, 2, 4, or 8): "))
    per_device_train_batch_size = int(get_input("Enter per-device training batch size (e.g., 1, 2, 4, or 8): "))
    num_epochs = int(get_input("Enter the number of epochs for training: "))

    quantize_choice = get_input("Would you like to use model quantization during training? [yes/no]: ", valid_options=["yes", "no"])
    if quantize_choice == "yes":
        bit_choice = get_input("Choose quantization bit - 4-bit or 8-bit? [4/8]: ", valid_options=["4", "8"])
        if bit_choice == "4":
            print("4-bit quantization selected. Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                token=hf_token if model_type == "private" else None,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            model = prepare_model_for_kbit_training(model)
        elif bit_choice == "8":
            print("8-bit quantization selected. Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                token=hf_token if model_type == "private" else None,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            if prepare_model_for_int8_training is None:
                raise ImportError("prepare_model_for_int8_training not found. Please update peft: pip install --upgrade peft")
            else:
                model = prepare_model_for_int8_training(model)
        lora_config = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
    else:
        precision_choice = get_input("Choose precision mode [fp16/bf16/fp32]: ", valid_options=["fp16", "bf16", "fp32"])
        torch_dtype = torch.float16 if precision_choice == "fp16" else torch.bfloat16 if precision_choice == "bf16" else torch.float32
        print(f"{precision_choice} precision selected. Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            token=hf_token if model_type == "private" else None,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        model.config.use_cache = False
        lora_config = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=hf_token if model_type == "private" else None,
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def create_prompt(system_text, user_text, assistant_text):
        prompt = f"[SYSTEM]\n{system_text}\n[USER]\n{user_text}\n[ASSISTANT]\n"
        return prompt, assistant_text

    def chunk_list(seq, chunk_size):
        return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]

    def preprocess_function(examples):
        system_texts = examples["System"]
        user_texts = examples["User"]
        assistant_texts = examples["Assistant"]
        final_input_ids = []
        final_attention_masks = []
        final_labels = []
        for sys_text, usr_text, asst_text in zip(system_texts, user_texts, assistant_texts):
            sys_text = sys_text or ""
            usr_text = usr_text or ""
            asst_text = asst_text or ""
            prompt, answer = create_prompt(sys_text.strip(), usr_text.strip(), asst_text.strip())
            full_text = prompt + answer
            tokenized = tokenizer(full_text, truncation=False, padding=False)
            input_ids = tokenized["input_ids"]
            prompt_tokenized = tokenizer(prompt, truncation=False, padding=False)
            prompt_length = len(prompt_tokenized["input_ids"])
            labels = [-100] * len(input_ids)
            for i in range(prompt_length, len(input_ids)):
                labels[i] = input_ids[i]
            if len(input_ids) > max_length:
                input_chunks = chunk_list(input_ids, max_length)
                label_chunks = chunk_list(labels, max_length)
                for inp_chunk, lbl_chunk in zip(input_chunks, label_chunks):
                    pad_len = max_length - len(inp_chunk)
                    if pad_len > 0:
                        inp_chunk += [tokenizer.pad_token_id] * pad_len
                        lbl_chunk += [-100] * pad_len
                    attn_mask = [1 if t != tokenizer.pad_token_id else 0 for t in inp_chunk]
                    final_input_ids.append(inp_chunk)
                    final_attention_masks.append(attn_mask)
                    final_labels.append(lbl_chunk)
            else:
                pad_len = max_length - len(input_ids)
                if pad_len > 0:
                    input_ids += [tokenizer.pad_token_id] * pad_len
                    labels += [-100] * pad_len
                attn_mask = [1 if t != tokenizer.pad_token_id else 0 for t in input_ids]
                final_input_ids.append(input_ids)
                final_attention_masks.append(attn_mask)
                final_labels.append(labels)
        return {"input_ids": final_input_ids, "attention_mask": final_attention_masks, "labels": final_labels}

    print("Tokenizing train and validation datasets...")
    tokenized_train = raw_datasets["train"].map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)
    tokenized_dev = raw_datasets["validation"].map(preprocess_function, batched=True, remove_columns=raw_datasets["validation"].column_names)
    data_collator = DefaultDataCollator()
    print("Training starts...")
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=200,
        save_steps=200,
        logging_steps=100,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=(quantize_choice == "no" and "fp16" in precision_choice),
        bf16=(quantize_choice == "no" and "bf16" in precision_choice),
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False,
        optim="adamw_torch_fused"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    print("Training completed -> './finetuned_model' folder created.")
    merge_choice = get_input("Would you like to merge the LoRA weights with the base model? [yes/no]: ", valid_options=["yes", "no"])
    final_model_dir = "./merged_final_model"
    if merge_choice == "yes":
        print("Merging LoRA weights with the base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        print(f"Merging completed. Merged model -> '{final_model_dir}'")
        push_choice = get_input("Would you like to push the merged model to Hugging Face? [yes/no]: ", valid_options=["yes", "no"])
        if push_choice == "yes":
            push_repo_id = get_input("Enter the Hugging Face repository ID to push the model to (e.g., AlicanKiraz0/My-Finetuned-Model): ")
            print("Pushing model to Hugging Face...")
            merged_model.push_to_hub(push_repo_id, token=hf_token)
            tokenizer.push_to_hub(push_repo_id, token=hf_token)
            print(f"Model successfully pushed to '{push_repo_id}'")
        else:
            print("Model was not pushed.")
    else:
        print("LoRA weights were not merged.")
    print("Process completed.")

if __name__ == "__main__":
    main()
