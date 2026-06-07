import numpy as np
import pandas as pd
import torch
import evaluate

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
ds = load_dataset("mteb/banking77")

train_df = pd.DataFrame(ds["train"])

num_labels = train_df["label"].nunique()
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

lora_config = LoraConfig(

    task_type=TaskType.SEQ_CLS,

    r=8,

    lora_alpha=16,

    lora_dropout=0.1,

    bias="none",

    target_modules=["q_lin", "v_lin"]
)

lora_model = get_peft_model(
    base_model,
    lora_config
)

def tokenize_function(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
  tokenized_ds = ds.map(
    tokenize_function,
    batched=True
)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

training_args = TrainingArguments(

    output_dir="./lora_results",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-4,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=1,

    logging_steps=100,

    report_to="none"
)

trainer = Trainer(

    model=lora_model,

    args=training_args,

    train_dataset=tokenized_ds["train"],

    eval_dataset=tokenized_ds["test"],

    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()





