from datasets import load_dataset
import pandas as pd

ds = load_dataset("mteb/banking77")

train_df = pd.DataFrame(ds["train"])
test_df = pd.DataFrame(ds["test"])
X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1,2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.fit_transform(X_test)
print(X_train_tfidf.shape)
print(X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print(y_pred[:10])



