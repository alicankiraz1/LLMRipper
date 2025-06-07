import os
import sys
import subprocess
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from tqdm import tqdm
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
    prepare_model_for_kbit_training
)
from transformers import BitsAndBytesConfig
try:
    from peft import prepare_model_for_int8_training
except ImportError:
    prepare_model_for_int8_training = None
    print("Warning: prepare_model_for_int8_training not found. Please update peft if you want to use 8-bit quantization.")
from huggingface_hub import login

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llmripper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMRipper:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.raw_datasets = None
        self._setup_environment()
    
    def _setup_environment(self):
        """Ortam değişkenlerini ve CUDA ayarlarını yapılandırır."""
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if torch.cuda.is_available():
            logger.info(f"CUDA kullanılabilir. GPU sayısı: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.warning("CUDA kullanılamıyor. CPU kullanılacak.")

    @staticmethod
    def print_banner():
        """ASCII banner'ı yazdırır."""
        try:
            import pyfiglet
        except ImportError:
            logger.info("pyfiglet yükleniyor...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyfiglet"])
            import pyfiglet
        ascii_banner = pyfiglet.figlet_format("LLMRipper")
        print(ascii_banner)
        print("Created by Alican Kiraz")

    @staticmethod
    def get_input(prompt: str, valid_options: Optional[list] = None) -> str:
        """Kullanıcıdan giriş alır ve doğrular."""
        while True:
            response = input(prompt).strip()
            if valid_options:
                if response.lower() in valid_options:
                    return response.lower()
                else:
                    print(f"Geçersiz giriş. Lütfen şunlardan birini seçin: {', '.join(valid_options)}")
            else:
                if response:
                    return response
                else:
                    print("Giriş boş olamaz. Lütfen tekrar deneyin.")

    def load_model(self, base_model_name: str, hf_token: Optional[str] = None):
        """Modeli yükler ve yapılandırır."""
        try:
            logger.info(f"Model yükleniyor: {base_model_name}")
            finetune_type = self.config.get("finetune_type", "lora")  # "lora" veya "full"
            if self.config.get("quantize", False):
                bit_choice = self.config.get("quantization_bits", 4)
                quant_config = None
                if bit_choice == 4:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        token=hf_token,
                        device_map="auto",
                        trust_remote_code=True,
                        quantization_config=quant_config
                    )
                    if finetune_type == "lora":
                        self.model = prepare_model_for_kbit_training(self.model)
                else:
                    quant_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        token=hf_token,
                        device_map="auto",
                        trust_remote_code=True,
                        quantization_config=quant_config
                    )
                    if finetune_type == "lora":
                        if prepare_model_for_int8_training is None:
                            raise ImportError("prepare_model_for_int8_training bulunamadı. Lütfen peft'i güncelleyin.")
                        self.model = prepare_model_for_int8_training(self.model)
            else:
                precision = self.config.get("precision", "fp16")
                torch_dtype = {
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                    "fp32": torch.float32
                }[precision]
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    token=hf_token,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.model.config.use_cache = False

            if finetune_type == "lora":
                lora_config = LoraConfig(
                    r=self.config.get("lora_r", 4),
                    lora_alpha=self.config.get("lora_alpha", 16),
                    lora_dropout=self.config.get("lora_dropout", 0.1),
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.enable_input_require_grads()
                logger.info("LoRA ile ince ayar için model hazırlandı.")
            else:
                logger.info("Full fine-tune için model hazırlandı (LoRA eklenmedi).")

        except Exception as e:
            logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
            raise

    def load_tokenizer(self, base_model_name: str, hf_token: Optional[str] = None):
        """Tokenizer'ı yükler ve yapılandırır."""
        try:
            logger.info("Tokenizer yükleniyor...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                token=hf_token,
                use_fast=False,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer başarıyla yüklendi.")
        except Exception as e:
            logger.error(f"Tokenizer yüklenirken hata oluştu: {str(e)}")
            raise

    def load_dataset(self, dataset_config: Dict[str, Any]):
        """Veri setini yükler ve hazırlar."""
        try:
            logging.info("Veri seti yükleniyor...")
            ds = load_dataset(
                dataset_config["name"],
                data_files=dataset_config["data_files"],
                split=dataset_config.get("split", "train")
            )
            # Eğer dönen veri Dataset ise, DatasetDict'e sar
            from datasets import Dataset, DatasetDict
            if isinstance(ds, Dataset):
                self.raw_datasets = DatasetDict({"train": ds})
            else:
                self.raw_datasets = ds
            logging.info("Veri seti başarıyla yüklendi!")
        except Exception as e:
            logging.error(f"Veri seti yüklenirken hata oluştu: {str(e)}")
            raise

    def preprocess_data(self):
        """Veri setini işler ve tokenleştirir."""
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer yüklenmemiş!")
            if not self.raw_datasets or "train" not in self.raw_datasets:
                raise ValueError("Veri seti yüklenmemiş veya 'train' split'i yok!")
            def tokenize_function(example):
                return self.tokenizer(
                    f"{example['System']}\nUser: {example['User']}\nAssistant: {example['Assistant']}",
                    truncation=True,
                    max_length=self.config["max_length"],
                    padding="max_length"
                )
            self.tokenized_train_dataset = self.raw_datasets["train"].map(tokenize_function, batched=False)
            # Eğer validation split varsa
            if "validation" in self.raw_datasets:
                self.tokenized_dev_dataset = self.raw_datasets["validation"].map(tokenize_function, batched=False)
            else:
                self.tokenized_dev_dataset = None
            logging.info("Veri seti başarıyla tokenleştirildi!")
        except Exception as e:
            logging.error(f"Veri ön işlenirken hata oluştu: {str(e)}")
            raise

    def train(self):
        """Modeli eğitir."""
        try:
            logger.info("Eğitim başlıyor...")
            training_args = TrainingArguments(
                output_dir="./finetuned_model",
                overwrite_output_dir=True,
                num_train_epochs=self.config.get("num_epochs", 3),
                per_device_train_batch_size=self.config.get("batch_size", 4),
                gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
                eval_steps=200,
                save_steps=200,
                logging_steps=100,
                learning_rate=self.config.get("learning_rate", 2e-5),
                warmup_steps=100,
                weight_decay=0.01,
                fp16=self.config.get("precision") == "fp16",
                bf16=self.config.get("precision") == "bf16",
                gradient_checkpointing=True,
                max_grad_norm=0.5,
                eval_strategy="steps",
                save_strategy="steps",
                save_total_limit=2,
                load_best_model_at_end=True,
                push_to_hub=False,
                optim="adamw_torch_fused"
            )

            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.tokenized_train_dataset,
                eval_dataset=self.tokenized_dev_dataset,
                data_collator=DefaultDataCollator(),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            self.trainer.train()
            self.trainer.save_model("./finetuned_model")
            self.tokenizer.save_pretrained("./finetuned_model")
            logger.info("Eğitim tamamlandı -> './finetuned_model' klasörü oluşturuldu.")
        except Exception as e:
            logger.error(f"Eğitim sırasında hata oluştu: {str(e)}")
            raise

    def merge_and_save(self, output_dir: str):
        """Modeli birleştirir ve kaydeder."""
        try:
            if not self.model:
                raise ValueError("Model yüklenmemiş!")
            if not self.tokenizer:
                raise ValueError("Tokenizer yüklenmemiş!")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            if self.config["finetune_type"] == "lora":
                logging.info("LoRA ağırlıkları birleştiriliyor...")
                self.model = self.model.merge_and_unload()
            logging.info(f"Model {output_dir} dizinine kaydediliyor...")
            self.model.save_pretrained(output_dir)
            # Tokenizer'ın tüm dosyalarını kaydet
            self.tokenizer.save_pretrained(output_dir)
            logging.info("Model başarıyla kaydedildi!")
        except Exception as e:
            logging.error(f"Model kaydedilirken hata oluştu: {str(e)}")
            raise