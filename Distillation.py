import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
ds = load_dataset("mteb/banking77")

train_df = pd.DataFrame(ds["train"])

num_labels = train_df["label"].nunique()

teacher_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

student_model_name = "google/bert_uncased_L-2_H-128_A-2"

tokenizer = AutoTokenizer.from_pretrained(
    student_model_name
)

student_model = AutoModelForSequenceClassification.from_pretrained(
    student_model_name,
    num_labels=num_labels
)

print(student_model)

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

def get_teacher_logits(texts):

    teacher_model.eval()

    all_logits = []

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():

            outputs = teacher_model(**inputs)

        logits = outputs.logits

        all_logits.append(
            logits.squeeze().numpy()
        )

    return np.array(all_logits)

sample_logits = get_teacher_logits(
    X_train[:5].tolist()
)

print(sample_logits.shape)

training_args = TrainingArguments(

    output_dir="./student_results",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=5e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=1,

    logging_steps=100,

    report_to="none"
)

trainer = Trainer(

    model=student_model,

    args=training_args,

    train_dataset=tokenized_ds["train"],

    eval_dataset=tokenized_ds["test"]
)

trainer.train()

trainer.evaluate()

