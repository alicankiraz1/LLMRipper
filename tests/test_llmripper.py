import unittest
import torch
import os
from pathlib import Path
from LLMRipper import LLMRipper

class TestLLMRipper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Test sınıfı başlatılmadan önce çalışır."""
        # Test veri dizinini oluştur
        os.makedirs("tests/data", exist_ok=True)

    def setUp(self):
        """Test öncesi hazırlık."""
        self.test_model_name = "gpt2"
        self.test_dataset_config = {
            "name": "json",
            "data_files": "tests/data/test_data.json",
            "split": "train"
        }
        self.config = {
            "quantize": False,
            "quantization_bits": 4,
            "max_length": 128,
            "num_epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-4,
            "finetune_type": "full"
        }
        self.ripper = LLMRipper(self.config)
        self.output_dir = "./test_merged_model"

    def test_initialization(self):
        """LLMRipper sınıfının başlatılmasını test eder."""
        self.assertEqual(self.ripper.config, self.config)
        self.assertIsNone(self.ripper.model)
        self.assertIsNone(self.ripper.tokenizer)
        self.assertIsNone(self.ripper.raw_datasets)

    def test_load_model(self):
        """Model yükleme fonksiyonunu test eder."""
        try:
            self.ripper.load_model(self.test_model_name)
            self.assertIsNotNone(self.ripper.model)
            self.assertTrue(hasattr(self.ripper.model, "forward"))
        except Exception as e:
            self.fail(f"Model yükleme hatası: {str(e)}")

    def test_load_tokenizer(self):
        """Tokenizer yükleme fonksiyonunu test eder."""
        try:
            self.ripper.load_tokenizer(self.test_model_name)
            self.assertIsNotNone(self.ripper.tokenizer)
            self.assertTrue(hasattr(self.ripper.tokenizer, "encode"))
        except Exception as e:
            self.fail(f"Tokenizer yükleme hatası: {str(e)}")

    def test_load_dataset(self):
        """Veri seti yükleme fonksiyonunu test eder."""
        try:
            config = self.test_dataset_config.copy()
            self.ripper.load_dataset(config)
            self.assertIsNotNone(self.ripper.raw_datasets)
            self.assertTrue("train" in self.ripper.raw_datasets)
        except Exception as e:
            self.fail(f"Veri seti yükleme hatası: {str(e)}")

    def test_preprocess_data(self):
        """Veri ön işleme fonksiyonunu test eder."""
        try:
            self.ripper.load_tokenizer(self.test_model_name)
            config = self.test_dataset_config.copy()
            self.ripper.load_dataset(config)
            self.ripper.preprocess_data()
            self.assertIsNotNone(self.ripper.tokenized_train_dataset)
            if "validation" in self.ripper.raw_datasets:
                self.assertIsNotNone(self.ripper.tokenized_dev_dataset)
            else:
                self.assertIsNone(self.ripper.tokenized_dev_dataset)
        except Exception as e:
            self.fail(f"Veri ön işleme hatası: {str(e)}")

    def test_merge_and_save(self):
        """Model birleştirme ve kaydetme fonksiyonunu test eder."""
        try:
            self.ripper.load_model(self.test_model_name)
            self.ripper.load_tokenizer(self.test_model_name)
            self.ripper.merge_and_save(self.output_dir)
            self.assertTrue(Path(self.output_dir).exists())
            self.assertTrue(Path(self.output_dir, "config.json").exists())
            tokenizer_files = list(Path(self.output_dir).glob("tokenizer.*")) + list(Path(self.output_dir).glob("vocab.*")) + list(Path(self.output_dir).glob("merges.*"))
            self.assertTrue(len(tokenizer_files) > 0)
            model_files = list(Path(self.output_dir).glob("pytorch_model.*")) + list(Path(self.output_dir).glob("model.*"))
            self.assertTrue(len(model_files) > 0)
        except Exception as e:
            self.fail(f"Model birleştirme ve kaydetme hatası: {str(e)}")

    def tearDown(self):
        """Test sonrası temizlik."""
        if Path(self.output_dir).exists():
            import shutil
            shutil.rmtree(self.output_dir)

    @classmethod
    def tearDownClass(cls):
        """Test sınıfı sonlandıktan sonra çalışır."""
        # Test veri dizinini temizle
        if Path("tests/data").exists():
            import shutil
            shutil.rmtree("tests/data")

if __name__ == "__main__":
    unittest.main() 