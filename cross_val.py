from sklearn.model_selection import cross_val_score
X = train_df["text"]

y = train_df["label"]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1,2)
)

X_tfidf = vectorizer.fit_transform(X)

print(X_tfidf.shape)
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_cv_accuracy = cross_val_score(
    log_model,
    X_tfidf,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
print("Scores Accuracy :", log_cv_accuracy)

print("Accuracy moyenne :", log_cv_accuracy.mean())

print("Écart-type :", log_cv_accuracy.std())

from sklearn.svm import LinearSVC

svc_model = LinearSVC(
    random_state=42
)

svc_cv_accuracy = cross_val_score(
    svc_model,
    X_tfidf,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

svc_cv_f1 = cross_val_score(
    svc_model,
    X_tfidf,
    y,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

print("===== LinearSVC =====")

print("Accuracy moyenne :", svc_cv_accuracy.mean())

print("F1 Macro moyen :", svc_cv_f1.mean())

rf_cv_accuracy = cross_val_score(
    rf_model,
    X_tfidf,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_cv_f1 = cross_val_score(
    rf_model,
    X_tfidf,
    y,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

print("===== RandomForest =====")

print("Accuracy moyenne :", rf_cv_accuracy.mean())

print("F1 Macro moyen :", rf_cv_f1.mean())


