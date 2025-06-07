# LLMRipper v2 - Interactive LLM Fine-Tuner

<div align="center">
  <img src="LLMRipper_Icon.png" alt="LLMRipper Logo" width="600"/>
  <br>
  <p><em>Fine-tune and merge LLMs with ease</em></p>
</div>

**LLMRipper** is an interactive tool for fine-tuning Large Language Models (LLMs) with both LoRA (Parameter-Efficient Fine-Tuning) and Full Fine-Tuning support.

## Features

✅ **LoRA & Full Fine-Tuning Support**: Choose between parameter-efficient LoRA or full model fine-tuning  
✅ **Multiple Dataset Formats**: Supports CSV, JSON, JSONL, and Parquet formats  
✅ **Dataset Validation**: Automatic validation for required System/User/Assistant columns  
✅ **Quantization Support**: 4-bit and 8-bit quantization for memory efficiency  
✅ **Interactive Workflow**: Step-by-step guided process  
✅ **HuggingFace Integration**: Seamless integration with HF Hub for models and datasets  
✅ **Error Handling**: Comprehensive error handling and validation  
✅ **Secure Token Input**: Hidden token input for better security  

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM (16GB+ recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch transformers datasets peft huggingface_hub pyfiglet accelerate
```

## Usage

### Quick Start

1. Run the script:
```bash
python3 LLMRipper.py
```

2. Follow the interactive prompts:
   - Choose model privacy (public/private)
   - Select fine-tuning strategy (LoRA/Full)
   - Specify dataset location and format
   - Configure training parameters
   - Start training!

### Dataset Requirements

Your dataset **MUST** contain these exact columns:
- `System`: System prompts/instructions
- `User`: User messages/questions
- `Assistant`: Assistant responses

### Supported Dataset Formats

- **CSV**: Comma-separated values
- **JSON**: Standard JSON array
- **JSONL**: JSON Lines (one JSON object per line)
- **Parquet**: Apache Parquet format

### Example Dataset (JSON)

```json
[
  {
    "System": "You are a helpful assistant.",
    "User": "What is the capital of France?",
    "Assistant": "The capital of France is Paris."
  },
  {
    "System": "You are a helpful assistant.",
    "User": "How do I cook pasta?",
    "Assistant": "To cook pasta: 1) Boil water, 2) Add pasta, 3) Cook and drain."
  }
]
```

## Configuration Options

### Fine-Tuning Strategies

- **LoRA (Recommended)**: Parameter-efficient, faster training, less memory usage
- **Full**: Complete model fine-tuning, more resource intensive

### Training Parameters

- **Sequence Length**: 128-8192 tokens (1024-2048 recommended)
- **Batch Size**: 1, 2, 4, or 8 (start with 1-2 for large models)
- **Gradient Accumulation**: 1, 2, 4, or 8 steps
- **Epochs**: Number of training epochs (1-10 recommended)
- **Quantization**: 4-bit or 8-bit for memory efficiency

### Precision Options

- **FP16**: Half-precision (faster, less memory)
- **BF16**: Brain Float 16 (better stability)
- **FP32**: Full precision (highest accuracy, more memory)

## Security Features

🔒 **Secure Token Input**: When entering HuggingFace tokens, your input is hidden from the terminal for security. You won't see the characters as you type - this is normal and intended behavior.

## Tips for Success

1. **Start Small**: Begin with a small dataset and short training to test
2. **Monitor Memory**: Use quantization if you encounter OOM errors
3. **Validation**: Always check your dataset format before training
4. **LoRA First**: Try LoRA before full fine-tuning for most use cases
5. **Save Frequently**: The tool auto-saves, but keep backups
6. **Token Security**: Your HF tokens are hidden during input for security

## Troubleshooting

### Common Issues

**"No module named 'torch'"**
```bash
pip install torch transformers datasets peft huggingface_hub
```

**"Dataset validation error"**
- Ensure your dataset has exactly these columns: System, User, Assistant
- Check column names are spelled correctly (case-sensitive)

**"CUDA out of memory"**
- Enable quantization (4-bit or 8-bit)
- Reduce batch size to 1
- Reduce sequence length
- Use LoRA instead of full fine-tuning

**"File not found"**
- Check the dataset file path
- Ensure the file format matches your selection

## Output Files

After training, you'll find:
- `./finetuned_model/`: Your fine-tuned model
- `./merged_final_model/`: Merged model (LoRA only, if chosen)

## Advanced Usage

### Using Private Models/Datasets

The tool supports private HuggingFace repositories. You'll need:
1. A HuggingFace account
2. An access token with appropriate permissions
3. Access to the private repository

### Pushing to HuggingFace Hub

After training, you can optionally push your model to HF Hub for sharing.

## Support

For issues or questions:
1. Check this README
2. Verify your dataset format
3. Try with a smaller dataset first
4. Check system requirements

## License

Created by Alican Kiraz - v2.0

---

Happy fine-tuning! 🚀 
