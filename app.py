from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel
)


# Chargement modèle

classifier = joblib.load(
    "intent_classifier.joblib"
)

model_info = joblib.load(
    "model_info.joblib"
)


# DistilBERT


MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

bert_model = AutoModel.from_pretrained(
    MODEL_NAME
)


# FastAPI


app = FastAPI(
    title="Finova Intent Classifier API"
)


# Input schema


class RequestBody(BaseModel):
    text: str


# Encode function


def encode_text(text):

    bert_model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = bert_model(**inputs)

    embedding = outputs.last_hidden_state[:,0,:]

    return embedding.detach().cpu().numpy()


# Health endpoint


@app.get("/health")

def health():

    return {
        "status": "ok"
    }


# Model info


@app.get("/model-info")

def get_model_info():

    return model_info


# Classification endpoint


@app.post("/classify")

def classify(request: RequestBody):

    embedding = encode_text(
        request.text
    )

    prediction = classifier.predict(
        embedding
    )[0]

    confidence = float(
        np.max(
            classifier.predict_proba(
                embedding
            )
        )
    )

    return {
        "intent": int(prediction),
        "confidence": confidence,
        "model": model_info["model"]
    }
