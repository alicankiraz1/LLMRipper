from transformers import (
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
    EarlyStoppingCallback,
)

def train_model(
    model,
    tokenised_train,
    tokenised_val,
    num_epochs,
    per_device_bs,
    grad_acc_steps,
    precision_choice
):
    print("\n***  Training  ***")
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_acc_steps,
        eval_steps=200,
        save_steps=200,
        logging_steps=100,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=(precision_choice == "fp16" if precision_choice else False),
        bf16=(precision_choice == "bf16" if precision_choice else False),
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
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
        data_collator=DefaultDataCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    
    return trainer

def save_and_merge_model(trainer, tokenizer, is_lora, hf_token, merge_weights, repo_id):
    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    print("Training finished → ./finetuned_model created.")

    if is_lora and merge_weights:
        merged_dir = "./merged_final_model"
        print("Merging…")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved → {merged_dir}")
        if repo_id:
            print("Uploading…")
            merged_model.push_to_hub(repo_id, token=hf_token)
            tokenizer.push_to_hub(repo_id, token=hf_token)
            print("Upload complete.") 