from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(y_test, y_pred)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro"
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("Accuracy :", accuracy)
print("F1 Macro :", f1_macro)
print("F1 Weighted :", f1_weighted)
