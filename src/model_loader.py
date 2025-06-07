import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
try:
    from peft import prepare_model_for_int8_training
except ImportError:
    prepare_model_for_int8_training = None
    print("Warning: prepare_model_for_int8_training not found. Update peft if you need 8-bit training support.")

from src.model_arch_helpers import get_target_modules_for_model

def load_model_and_tokenizer(
    base_model_name,
    model_privacy,
    hf_token,
    quantize_choice,
    bit_choice,
    precision_choice,
    is_lora
):
    print("Loading model – this can take a while…")

    common_kwargs = dict(
        token=hf_token if model_privacy == "private" else None,
        trust_remote_code=True,
        device_map="auto",
    )

    torch_dtype = None
    if precision_choice:
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[precision_choice]

    if quantize_choice == "yes":
        load_kwargs = {
            "load_in_4bit": bit_choice == "4",
            "load_in_8bit": bit_choice == "8",
        }
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **common_kwargs, **load_kwargs)
        if bit_choice == "4":
            model = prepare_model_for_kbit_training(model)
        else:
            if prepare_model_for_int8_training is None:
                raise ImportError("prepare_model_for_int8_training missing – update *peft* package.")
            model = prepare_model_for_int8_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **common_kwargs, torch_dtype=torch_dtype)
        model.config.use_cache = False

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
                print("Could not apply LoRA. Continuing with full fine-tuning.")
                is_lora = False
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, **common_kwargs, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer, is_lora 