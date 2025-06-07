# LLMRipper

LLMRipper, Hugging Face LLM modellerini kolayca ince ayar yapmanızı sağlayan güçlü bir araçtır. Transformers mimarisini kullanarak, herhangi bir Hugging Face modelini kod yazmadan ince ayar yapabilirsiniz.

## Özellikler

- 🤖 Herhangi bir Hugging Face modelini destekler
- 🚀 LoRA ile verimli ince ayar
- 💾 4-bit ve 8-bit kuantizasyon desteği
- 🔄 Otomatik veri seti bölme ve ön işleme
- 📊 Detaylı eğitim metrikleri
- 🎯 Erken durdurma ve model kaydetme
- 🔒 Özel ve herkese açık modeller için destek

## Kurulum

```bash
# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

## Kullanım

```python
from LLMRipper import LLMRipper

# LLMRipper örneği oluştur
ripper = LLMRipper({
    "quantize": True,
    "quantization_bits": 4,
    "max_length": 1024,
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "precision": "fp16"
})

# Model ve tokenizer'ı yükle
ripper.load_model("model_name", hf_token)
ripper.load_tokenizer("model_name", hf_token)

# Veri setini yükle ve işle
ripper.load_dataset({
    "type": "huggingface",
    "name": "dataset_name",
    "private": False
})
ripper.preprocess_data()

# Eğitimi başlat
ripper.train()

# Modeli birleştir ve kaydet
ripper.merge_and_save()
```

## Konfigürasyon Seçenekleri

| Parametre | Açıklama | Varsayılan |
|-----------|-----------|------------|
| quantize | Kuantizasyon kullanılsın mı? | False |
| quantization_bits | Kuantizasyon biti (4 veya 8) | 4 |
| max_length | Maksimum dizi uzunluğu | 1024 |
| num_epochs | Eğitim epoch sayısı | 3 |
| batch_size | Batch boyutu | 4 |
| gradient_accumulation_steps | Gradient biriktirme adımları | 4 |
| learning_rate | Öğrenme oranı | 2e-5 |
| precision | Hassasiyet (fp16/bf16/fp32) | "fp16" |

## Test

```bash
# Tüm testleri çalıştır
python -m unittest discover tests

# Belirli bir test dosyasını çalıştır
python -m unittest tests/test_llmripper.py
```

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir özellik dalı oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Dalınıza push edin (`git push origin feature/amazing-feature`)
5. Bir Pull Request açın

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## İletişim

Alican Kiraz - [@AlicanKiraz](https://twitter.com/AlicanKiraz)

Proje Linki: [https://github.com/AlicanKiraz/LLMRipper](https://github.com/AlicanKiraz/LLMRipper)
