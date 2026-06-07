
import torch
import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModel
)

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)
print(tokenizer)

print(model)

def encode_texts(texts, batch_size=32):

    embeddings = []

    model.eval()

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:,0,:]

        embeddings.append(
            cls_embeddings.numpy()
        )

    return np.vstack(embeddings)

from datasets import load_dataset
import pandas as pd

ds = load_dataset("mteb/banking77")

train_df = pd.DataFrame(ds["train"])
test_df = pd.DataFrame(ds["test"])

X_train = train_df["text"]
X_test = test_df["text"]

y_train = train_df["label"]
y_test = test_df["label"]


X_train_embeddings = encode_texts(
    X_train.tolist()
)
X_test_embeddings = encode_texts(
    X_test.tolist()
)






