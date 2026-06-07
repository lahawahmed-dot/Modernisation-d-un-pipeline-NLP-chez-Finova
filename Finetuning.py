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
ds = load_dataset("mteb/banking77")

print(ds)
train_df = pd.DataFrame(ds["train"])

num_labels = train_df["label"].nunique()

print("Nombre de labels :", num_labels)

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
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

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    return accuracy


training_args = TrainingArguments(

    output_dir="./results",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=1,

    weight_decay=0.01,

    logging_dir="./logs",

    logging_steps=100
)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_ds["train"],

    eval_dataset=tokenized_ds["test"],

    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()



