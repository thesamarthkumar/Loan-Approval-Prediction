import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler


# Load the data

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


param_grid = {
    "penalty": ["l2", "l1"],
    "C": [0.01, 0.1, 1, 10, 100],
    "max_iter": [100, 500, 1000],
    "solver": ["liblinear"],
}


grid = GridSearchCV(
    LogisticRegression(
        class_weight="balanced",
    ),
    param_grid,
    scoring="f1",
    cv=5,
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best F1 score:", grid.best_score_)

results = pd.DataFrame(grid.cv_results_)

# select the columns you care about and sort by rank
display_cols = [
    "param_penalty",
    "param_C",
    "param_max_iter",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
]
results = results[display_cols].sort_values("rank_test_score")

# Print Results
print(results.to_string(index=False))
results.to_csv("gridsearchcv-results.csv", index=False)


# Get the classification report for the best model
best_model = grid.best_estimator_

y_test_pred = best_model.predict(X_test)

test_f1 = f1_score(y_test, y_test_pred)

print(f"Best Model test F1 score: {test_f1:.4f}")
print("\nClassification report:\n")

print(classification_report(y_test, y_test_pred))

# Save the best model and the classification report as CSVs
report_dict = classification_report(y_test, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).T
report_df.to_csv(
    "Best_LR_classification_report_logistic-regression.csv", index=True
)
