import questionary
import sys

def ask(question):
    """Wrapper for questionary prompts to handle global exit."""
    answer = question.ask()
    if answer is None:
        print("\nExiting...")
        sys.exit(0)
    return answer

def get_tuning_strategy():
    """Asks the user to select the fine-tuning strategy."""
    tuning_type = ask(questionary.select(
        "Select tuning strategy:",
        choices=["LoRA (parameter-efficient)", "Full"],
        default="LoRA (parameter-efficient)"
    ))
    return "lora" if "LoRA" in tuning_type else "full"

def get_model_info():
    """Gets model information from the user."""
    while True:
        model_privacy = ask(questionary.select(
            "Is your base model private or public?",
            choices=["public", "private", "Go Back"],
            default="public"
        ))
        if model_privacy == "Go Back":
            return "BACK", None, None

        hf_token = None
        if model_privacy == "private":
            hf_token = ask(questionary.password("Enter your Hugging Face token:"))

        while True:
            base_model_name = ask(questionary.text(
                "Enter the Hugging Face repo for the base model (e.g., AlicanKiraz0/LLM-Base).\n(Type 'BACK' to change model privacy)"
            ))
            if base_model_name.strip().upper() == "BACK":
                break
            if base_model_name.strip():
                return model_privacy, hf_token, base_model_name.strip()
            print("Model repo cannot be empty.")

def get_dataset_info(hf_token):
    """Gets dataset information from the user."""
    print("\n***  Dataset Requirements  ***")
    print("Your file MUST contain the columns: System | User | Assistant\n")

    while True:
        dataset_source = ask(questionary.select(
            "Is the dataset local or on HuggingFace Hub?",
            choices=["local", "huggingface", "Go Back"],
            default="local"
        ))
        if dataset_source == "Go Back":
            return "BACK", None, None, None, None

        while True:
            dataset_format = ask(questionary.select(
                "Which file format are you using?",
                choices=["csv", "json", "jsonl", "parquet", "Go Back"],
                default="jsonl"
            ))
            if dataset_format == "Go Back":
                break

            if dataset_source == "local":
                while True:
                    dataset_path = ask(questionary.text(
                        "Enter the path to your local dataset file.\n(Type 'BACK' to change format)"
                    ))
                    if dataset_path.strip().upper() == "BACK":
                        break
                    if dataset_path.strip():
                        return dataset_source, dataset_format, dataset_path.strip(), None, hf_token
                    print("Path cannot be empty.")
                if dataset_path.strip().upper() == "BACK":
                    continue

            else:  # huggingface
                while True:
                    hub_privacy = ask(questionary.select(
                        "Is the HF dataset repo public or private?",
                        choices=["public", "private", "Go Back"],
                        default="public"
                    ))
                    if hub_privacy == "Go Back":
                        break
                    
                    if hub_privacy == "private" and not hf_token:
                        hf_token = ask(questionary.password("Enter your Hugging Face token for the *dataset*:"))
                    
                    while True:
                        dataset_repo = ask(questionary.text(
                            "Enter the HF dataset repo (e.g. AlicanKiraz0/my-chat-dataset).\n(Type 'BACK' to change repo privacy)"
                        ))
                        if dataset_repo.strip().upper() == "BACK":
                            break
                        if dataset_repo.strip():
                            return dataset_source, dataset_format, None, dataset_repo.strip(), hf_token
                        print("Repo cannot be empty.")
                    if dataset_repo.strip().upper() == 'BACK':
                        continue
                if hub_privacy == 'Go Back':
                    continue

def get_hyperparameters():
    """Gets hyperparameters from the user."""
    while True:
        max_length_str = ask(questionary.text(
            "Enter maximum sequence length (e.g. 1024 or 2048):",
            default="1024",
            validate=lambda val: val.isdigit() and 128 <= int(val) <= 8192 or "Please enter a number between 128 and 8192"
        ))
        if max_length_str.upper() == "BACK": # User can't type back because of validator, but good practice
            return "BACK", None, None, None

        max_length = int(max_length_str)

        grad_acc_steps_str = ask(questionary.select(
            "Gradient accumulation steps:",
            choices=["1", "2", "4", "8", "Go Back"],
            default="2"
        ))
        if grad_acc_steps_str == "Go Back":
            continue
        grad_acc_steps = int(grad_acc_steps_str)
        
        per_device_bs_str = ask(questionary.select(
            "Per-device train batch size:",
            choices=["1", "2", "4", "8", "Go Back"],
            default="2"
        ))
        if per_device_bs_str == "Go Back":
            continue
        per_device_bs = int(per_device_bs_str)

        num_epochs_str = ask(questionary.text(
            "Number of epochs:",
            default="3",
            validate=lambda val: val.isdigit() and 1 <= int(val) <= 100 or "Please enter a number between 1 and 100"
        ))
        num_epochs = int(num_epochs_str)

        return max_length, grad_acc_steps, per_device_bs, num_epochs

def get_quantization_config(tuning_type):
    """Gets quantization and precision configuration."""
    while True:
        if tuning_type == "full":
            print("Full fine-tuning with 4-bit/8-bit adapters is not officially supported – switching to non-quantised training.")
            quantize_choice = "no"
        else:
            quantize_choice = ask(questionary.select(
                "Use model quantisation?",
                choices=["yes", "no", "Go Back"],
                default="yes"
            ))
        
        if quantize_choice == "Go Back":
            return "BACK", None, None

        bit_choice = None
        precision_choice = None

        if quantize_choice == "yes":
            bit_choice = ask(questionary.select(
                "Quantisation bits – 4 or 8?",
                choices=["4", "8", "Go Back"],
                default="4"
            ))
            if bit_choice == "Go Back":
                continue
        else:
            precision_choice = ask(questionary.select(
                "Choose precision:",
                choices=["fp16", "bf16", "fp32", "Go Back"],
                default="fp16"
            ))
            if precision_choice == "Go Back":
                continue
        
        return quantize_choice, bit_choice, precision_choice

def get_hub_push_config(is_lora, hf_token):
    """Asks if the user wants to merge LoRA and push to hub."""
    if not is_lora:
        return False, None

    while True:
        merge_choice_str = ask(questionary.select(
            "Merge LoRA weights with the base model?", 
            choices=["Yes", "No"], 
            default="Yes"
        ))
        merge_choice = (merge_choice_str == "Yes")
        if not merge_choice:
            return False, None

        push_to_hub_str = ask(questionary.select(
            "Push merged model to HuggingFace Hub?", 
            choices=["Yes", "No", "Go Back"], 
            default="No"
        ))
        if push_to_hub_str == "Go Back":
            continue
        
        push_to_hub = (push_to_hub_str == "Yes")
        if not push_to_hub:
            return True, None

        while True:
            repo_id = ask(questionary.text(
                "HF repo ID to push to (e.g. AlicanKiraz0/LLMRipper-finetuned):\n(Type 'BACK' to go back)",
                validate=lambda text: True if text.strip().upper() == 'BACK' else len(text.strip()) > 0 or "Cannot be empty"
            ))
            if repo_id.strip().upper() == "BACK":
                break
            
            return True, repo_id.strip() 