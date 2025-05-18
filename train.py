import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def load_data():
    """Load a public dataset for language modeling."""
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return dataset

def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda e: tokenizer(e["text"]), batched=True, remove_columns=["text"])

def create_model(tokenizer):
    """Create a roughly 1B parameter GPT-NeoX model."""
    config = GPTNeoXConfig(
        vocab_size=len(tokenizer),
        hidden_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=16384,
    )
    model = GPTNeoXForCausalLM(config)
    return model

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_data()
    tokenized = tokenize_data(dataset, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=5000,
        logging_steps=500,
        report_to="none",
    )

    model = create_model(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./model")
    tokenizer.save_pretrained("./model")


if __name__ == "__main__":
    main()
