# LLMRipper

<div align="center">
  <img src="LLMRipper_Icon.png" alt="LLMRipper Logo" width="600"/>
  <br>
  <p><em>Fine-tune and merge LLMs with ease</em></p>
</div>

## 🚀 Overview

LLMRipper is a powerful Python tool designed to simplify the process of fine-tuning and merging Large Language Models (LLMs). It provides a streamlined interface for loading models, processing datasets, and creating merged models with enhanced capabilities.

## ✨ Features

- **Easy Model Loading**: Load models from Hugging Face Hub with a single line of code
- **Dataset Processing**: Efficient handling of training datasets
- **Model Merging**: Seamless merging of multiple models
- **Customizable Training**: Flexible training parameters
- **Progress Tracking**: Real-time progress monitoring
- **Error Handling**: Robust error management and logging

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Other dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/alicankiraz1/LLMRipper.git
cd LLMRipper
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
from llmripper import LLMRipper

# Initialize LLMRipper
ripper = LLMRipper(
    model_name="gpt2",
    dataset_name="wikitext",
    output_dir="./output"
)

# Load and process data
ripper.load_dataset()
ripper.preprocess_data()

# Load model and tokenizer
ripper.load_model()
ripper.load_tokenizer()

# Merge and save models
ripper.merge_and_save()
```

### Advanced Usage

```python
# Custom training configuration
ripper = LLMRipper(
    model_name="gpt2",
    dataset_name="wikitext",
    output_dir="./output",
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3
)

# Custom preprocessing
ripper.preprocess_data(
    max_length=512,
    truncation=True,
    padding="max_length"
)
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📫 Contact

For questions and support, please open an issue in the GitHub repository.

## 🙏 Acknowledgments

- Hugging Face for their amazing transformers library
- The open-source community for their continuous support

---

<div align="center">
  <p>Made with ❤️ by Alican Kiraz</p>
</div>
