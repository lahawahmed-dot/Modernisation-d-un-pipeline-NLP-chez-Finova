from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(
    X_train_tfidf,
    y_train
)
y_pred_rf = rf_model.predict(
    X_test_tfidf
)
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

rf_f1_macro = f1_score(
    y_test,
    y_pred_rf,
    average="macro"
)

rf_f1_weighted = f1_score(
    y_test,
    y_pred_rf,
    average="weighted"
)

print("===== Random Forest =====")

print("Accuracy :", rf_accuracy)

print("F1 Macro :", rf_f1_macro)

print("F1 Weighted :", rf_f1_weighted)
