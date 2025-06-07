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