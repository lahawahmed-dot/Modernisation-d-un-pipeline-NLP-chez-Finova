from sklearn.svm import LinearSVC
svc_model = LinearSVC(
    random_state=42
)
svc_model.fit(
    X_train_tfidf,
    y_train
)
y_pred_svc = svc_model.predict(
    X_test_tfidf
)
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

svc_accuracy = accuracy_score(
    y_test,
    y_pred_svc
)

svc_f1_macro = f1_score(
    y_test,
    y_pred_svc,
    average="macro"
)

svc_f1_weighted = f1_score(
    y_test,
    y_pred_svc,
    average="weighted"
)

print("===== LinearSVC =====")
print("Accuracy :", svc_accuracy)
print("F1 Macro :", svc_f1_macro)
print("F1 Weighted :", svc_f1_weighted)
