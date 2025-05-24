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
    print(
        "Uyarı: prepare_model_for_int8_training bulunamadı. 8-bit nicemlemeyi kullanmak istiyorsanız lütfen peft'i güncelleyin."
    )

from huggingface_hub import login

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def install_pyfiglet():
    """pyfiglet yüklü değilse yükler."""
    try:
        import pyfiglet  # noqa: F401
    except ImportError:
        print("pyfiglet bulunamadı. pyfiglet yükleniyor...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyfiglet"]
        )


def print_banner():
    """LLMRipper için ASCII banner yazdırır."""
    install_pyfiglet()
    import pyfiglet

    ascii_banner = pyfiglet.figlet_format("LLMRipper")
    print(ascii_banner)
    print("Oluşturan: Alican Kiraz")


def get_user_input(prompt, valid_options=None):
    """Kullanıcıdan giriş ister ve isteğe bağlı olarak geçerli seçenekleri doğrular."""
    while True:
        response = input(prompt).strip().lower()
        if valid_options and response not in valid_options:
            print(
                f"Geçersiz giriş. Lütfen şunlardan birini seçin: {', '.join(valid_options)}"
            )
        elif not response:
            print("Giriş boş olamaz. Lütfen tekrar deneyin.")
        else:
            return response


def load_model(model_name, hf_token, model_type, quantization_config=None, precision=None):
    """Modeli belirtilen konfigürasyonla yükler."""
    kwargs = {
        "pretrained_model_name_or_path": model_name,
        "token": hf_token if model_type == "özel" else None,
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if quantization_config:
        if quantization_config["bits"] == 4:
            print("4-bit nicemleme seçildi. Model yükleniyor...")
            kwargs["load_in_4bit"] = True
        elif quantization_config["bits"] == 8:
            print("8-bit nicemleme seçildi. Model yükleniyor...")
            kwargs["load_in_8bit"] = True
            if prepare_model_for_int8_training is None:
                raise ImportError(
                    "prepare_model_for_int8_training bulunamadı. Lütfen peft'i güncelleyin: pip install --upgrade peft"
                )

        model = AutoModelForCausalLM.from_pretrained(**kwargs)

        if quantization_config["bits"] == 4:
            model = prepare_model_for_kbit_training(model)
        elif quantization_config["bits"] == 8:
            model = prepare_model_for_int8_training(model)

    else:
        print(f"{precision} hassasiyet seçildi. Model yükleniyor...")
        if precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        model.config.use_cache = False

    return model


def configure_lora(model):
    """LoRA konfigürasyonunu ayarlar ve modeli LoRA ile sarar."""
    lora_config = LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    return model

def load_and_split_dataset(dataset_path, hf_token=None, is_private=False):
    """Veri kümesini yükler ve eğitim ve doğrulama bölümlerine ayırır."""
    try:
        if is_private and hf_token:
            raw_datasets = load_dataset(dataset_path, token=hf_token)
        else:
            raw_datasets = load_dataset(dataset_path)
    except Exception as e:
        raise ValueError(f"Veri kümesini yüklerken hata: {e}")

    if "train" not in raw_datasets:
        raise ValueError(
            "Veri kümesi 'train' bölümünü içermiyor. Lütfen veri kümenizin doğru olduğundan emin olun."
        )

    split_dataset = raw_datasets["train"].train_test_split(
        test_size=0.1, shuffle=True, seed=42
    )
    return DatasetDict(
        {"train": split_dataset["train"], "validation": split_dataset["test"]}
    )


def create_prompt(system_text, user_text, assistant_text):
    """Verilen metinlerle bir istem oluşturur."""
    prompt = f"[SYSTEM]\n{system_text}\n[USER]\n{user_text}\n[ASSISTANT]\n"
    return prompt, assistant_text


def chunk_list(seq, chunk_size):
    """Bir diziyi belirli bir boyutta parçalara ayırır."""
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


def preprocess_function(examples, tokenizer, max_length):
    """Veri kümesi örneklerini ön işler."""
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

        prompt, answer = create_prompt(
            sys_text.strip(), usr_text.strip(), asst_text.strip()
        )
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

                attn_mask = [
                    1 if t != tokenizer.pad_token_id else 0 for t in inp_chunk
                ]
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

    return {
        "input_ids": final_input_ids,
        "attention_mask": final_attention_masks,
        "labels": final_labels,
    }


def set_hf_token_environment():
    """Kullanıcıdan HF_TOKEN alır ve ortam değişkeni olarak ayarlar."""
    hf_token = get_user_input(
        "Lütfen Hugging Face tokeninizi girin ve ENTER'a basın: \n"
        "Tokeninizi buradan alabilirsiniz: https://huggingface.co/settings/tokens\n"
    )

    if os.environ["HF_TOKEN"] is None:
        os.environ["HF_TOKEN"] = hf_token
        print("HF_TOKEN env değişkeni ayarlandı.")
    elif "HF_TOKEN" in os.environ:
        print("HF_TOKEN env değişkeninden yüklendi.")
    else:
        print("HF_TOKEN sağlanmadı. Genel modellerle sınırlısınız.")


def main():
    """Ana eğitim süreci."""
    print_banner()
    set_hf_token_environment()  # HF_TOKEN'ı ayarla

    model_type = get_user_input(
        "Modeliniz genel mi yoksa özel mi [genel/özel]: ",
        valid_options=["genel", "özel"],
    )

    base_model_name = get_user_input(
        "Model için Hugging Face depo adını girin (örn. AlicanKiraz0/SenecaLLM_x_Qwen2.5-7B-CyberSecurity): "
    )

    dataset_choice = get_user_input(
        "Veri kümesi yerel mi yoksa Hugging Face'ten mi? [yerel/huggingface]: ",
        valid_options=["yerel", "huggingface"],
    )
    if dataset_choice == "yerel":
        dataset_path = get_user_input("Yerel veri kümenizin yolunu girin: ")
        try:
            raw_datasets = load_dataset("json", data_files={"train": dataset_path})
        except Exception as e:
            raise ValueError(f"Veri kümesini yüklerken hata: {e}")

        if "train" not in raw_datasets:
            raise ValueError(
                "Veri kümesi 'train' bölümünü içermiyor. Lütfen veri kümenizin doğru olduğundan emin olun."
            )

        split_dataset = raw_datasets["train"].train_test_split(
            test_size=0.1, shuffle=True, seed=42
        )
        raw_datasets = DatasetDict(
            {"train": split_dataset["train"], "validation": split_dataset["test"]}
        )
    else:
        hf_dataset_privacy = get_user_input(
            "Hugging Face veri kümesi genel mi yoksa özel mi? [genel/özel]: ",
            valid_options=["genel", "özel"],
        )

        hf_token = os.environ.get(
            "HF_TOKEN"
        )  # Ortam değişkeninden HF_TOKEN'ı al

        dataset_repo = get_user_input(
            "Veri kümesi için Hugging Face depo adını girin (örn. AlicanKiraz0/Test-CyberSec-dataset): "
        )
        try:
           if hf_dataset_privacy == "özel":
                raw_datasets = load_and_split_dataset(
                    dataset_repo, hf_token, is_private=True
                )
           else:
               raw_datasets = load_and_split_dataset(
                dataset_repo, hf_token, is_private=False
            )
        except ValueError as e:
            print(f"Hata: {e}")
            return

    max_length = int(
        get_user_input(
            "Maksimum sıra uzunluğunu girin (örn. 1024 veya 2048): "
        )
    )
    gradient_accumulation_steps = int(
        get_user_input(
            "Gradyan birikim adımlarını girin (örn. 1, 2, 4 veya 8): "
        )
    )
    per_device_train_batch_size = int(
        get_user_input(
            "Cihaz başına eğitim yığın boyutunu girin (örn. 1, 2, 4 veya 8): "
        )
    )
    num_epochs = int(get_user_input("Eğitim için epok sayısını girin: "))

    quantize_choice = get_user_input(
        "Eğitim sırasında model nicemlemesini kullanmak ister misiniz? [evet/hayır]: ",
        valid_options=["evet", "hayır"],
    )

    quantization_config = None
    precision_choice = None

    if quantize_choice == "evet":
        bit_choice = get_user_input(
            "Nicemleme bitini seçin - 4-bit veya 8-bit? [4/8]: ", valid_options=["4", "8"]
        )
        quantization_config = {"bits": int(bit_choice)}
    else:
        precision_choice = get_user_input(
            "Hassasiyet modunu seçin [fp16/bf16/fp32]: ",
            valid_options=["fp16", "bf16", "fp32"],
        )

    model = load_model(
        base_model_name, hf_token, model_type, quantization_config, precision_choice
    )
    model = configure_lora(model)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=hf_token if model_type == "özel" else None,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Eğitim ve doğrulama veri kümeleri tokenleştiriliyor...")
    tokenized_train = raw_datasets["train"].map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    tokenized_dev = raw_datasets["validation"].map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    data_collator = DefaultDataCollator()

    print("Eğitim başlıyor...")
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
        fp16=(quantize_choice == "hayır" and "fp16" in precision_choice),
        bf16=(quantize_choice == "hayır" and "bf16" in precision_choice),
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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    print("Eğitim tamamlandı -> './finetuned_model' klasörü oluşturuldu.")

    merge_choice = get_user_input(
        "LoRA ağırlıklarını temel modelle birleştirmek ister misiniz? [evet/hayır]: ",
        valid_options=["evet", "hayır"],
    )
    final_model_dir = "./merged_final_model"

    if merge_choice == "evet":
        print("LoRA ağırlıkları temel modelle birleştiriliyor...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        print(f"Birleştirme tamamlandı. Birleştirilmiş model -> '{final_model_dir}'")

        push_choice = get_user_input(
            "Birleştirilmiş modeli Hugging Face'e göndermek ister misiniz? [evet/hayır]: ",
            valid_options=["evet", "hayır"],
        )
        if push_choice == "evet":
            push_repo_id = get_user_input(
                "Modeli göndermek için Hugging Face depo kimliğini girin (örn. AlicanKiraz0/FinetunedModel): "
            )
            print("Model Hugging Face'e gönderiliyor...")
            merged_model.push_to_hub(push_repo_id, token=hf_token)
            tokenizer.push_to_hub(push_repo_id, token=hf_token)
            print(f"Model başarıyla '{push_repo_id}' adresine gönderildi")
        else:
            print("Model gönderilmedi.")
    else:
        print("LoRA ağırlıkları birleştirilmedi.")

    print("İşlem tamamlandı.")


if __name__ == "__main__":
    main()