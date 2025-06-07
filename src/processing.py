def create_prompt(sys_text: str, usr_text: str):
    return f"[SYSTEM]\n{sys_text}\n[USER]\n{usr_text}\n[ASSISTANT]\n"

def chunk(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]

class Processor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess_function(self, examples):
        final_input_ids, final_attention_masks, final_labels = [], [], []
        for sys_t, usr_t, as_t in zip(examples["System"], examples["User"], examples["Assistant"]):
            prompt = create_prompt(sys_t or "", usr_t or "")
            full_text = prompt + (as_t or "")
            tokenised = self.tokenizer(full_text, truncation=False, padding=False)
            ids = tokenised["input_ids"]
            prompt_len = len(self.tokenizer(prompt, truncation=False, padding=False)["input_ids"])
            labels = [-100] * len(ids)
            for i in range(prompt_len, len(ids)):
                labels[i] = ids[i]
            # Split or pad to max_length
            if len(ids) > self.max_length:
                id_chunks = chunk(ids, self.max_length)
                label_chunks = chunk(labels, self.max_length)
                for ic, lc in zip(id_chunks, label_chunks):
                    pad_len = self.max_length - len(ic)
                    ic += [self.tokenizer.pad_token_id] * pad_len
                    lc += [-100] * pad_len
                    attn = [int(tok != self.tokenizer.pad_token_id) for tok in ic]
                    final_input_ids.append(ic)
                    final_attention_masks.append(attn)
                    final_labels.append(lc)
            else:
                pad_len = self.max_length - len(ids)
                ids += [self.tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
                attn = [int(tok != self.tokenizer.pad_token_id) for tok in ids]
                final_input_ids.append(ids)
                final_attention_masks.append(attn)
                final_labels.append(labels)
        return {"input_ids": final_input_ids, "attention_mask": final_attention_masks, "labels": final_labels} 