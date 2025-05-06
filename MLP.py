import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# print(X_train.head())
# print(y_train.head())


# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE to balance the classes
smote = SMOTE(k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)

hidden_configs = [
    (128,),
    (256,),
    (128, 64),
    (256, 128),
    (128, 64, 32),
    (256, 128, 64),
    (256, 128, 64, 32),
    (512, 512, 256, 128),
]

train_f1_scores = []
test_f1_scores = []
config_labels = []


print("MLP Configurations")
for layers in hidden_configs:
    mlp_model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation="relu",
        solver="adam",
        max_iter=10000,
        early_stopping=True,
    )

    mlp_model.fit(X_res, y_res)
    y_train_pred = mlp_model.predict(X_train)
    y_test_pred = mlp_model.predict(X_test)

    # Calculate F1 scores for training and testing sets.
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    train_f1_scores.append(train_f1)
    test_f1_scores.append(test_f1)
    config_labels.append(str(layers))

    print(
        f"Architecture {layers}: Training F1 = {train_f1:.4f}, Testing F1 = {test_f1:.4f}"
    )

    if layers == (128, 64):
        best_f1 = test_f1
        best_model_config = layers
        best_model_pred = y_test_pred


x = np.arange(len(hidden_configs))
width = 0.35

plt.figure()
plt.bar(x - width / 2, train_f1_scores, width, label="Training F1")
plt.bar(x + width / 2, test_f1_scores, width, label="Testing F1")
plt.xlabel("MLP Architecture")
plt.ylabel("F1 Score")
plt.title("MLP Performance With Different Architecture")
plt.xticks(x, config_labels)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Print out the classification report for the best model
print(f"Best Model test F1 score: {best_f1:.4f} and Configuration: (128, 64)")
print("\nClassification report:\n")

print(classification_report(y_test, best_model_pred))

# Save the best model and the classification report as CSVs
report_dict = classification_report(y_test, best_model_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).T
report_df.to_csv(
    "Best_MLP_classification_report_logistic-regression.csv", index=True
)
