
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score
)

# --------------------------------------------------------
#                      LOAD DATA
# --------------------------------------------------------

data_2023 = pd.read_csv("data/training_2023.csv")
data_2024 = pd.read_csv("data/training_2024.csv")
data_2025 = pd.read_csv("data/training_2025.csv")

FEATURES = [
    "avg_exit_velocity",
    "hard_hit_rate",
    "strikeout_rate",
    "walk_rate",
    "plate_appearances",
    "batted_balls",
    "recent_ops"
]

print("--- DATASET SIZES ---")

for year, data in [
    (2023, data_2023),
    (2024, data_2024),
    (2025, data_2025)
]:
    print(f"{year}: {len(data):,} examples")


# --------------------------------------------------------
#                  MODEL FUNCTIONS
# --------------------------------------------------------

def create_model(features):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )


def evaluate(model, test_data, features, name):
    X_test = test_data[features]
    y_test = test_data["breakout"]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")

    print(
        f"Accuracy: {accuracy_score(y_test, predictions):.3f}"
    )

    print(
        f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}"
    )

    print(
        f"Brier: {brier_score_loss(y_test, probabilities):.3f}"
    )

    print(
        f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}"
    )

    print(
        f"Recall: {recall_score(y_test, predictions, zero_division=0):.3f}"
    )

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "brier": brier_score_loss(y_test, probabilities)
    }


# --------------------------------------------------------
#             VALIDATION ON THE 2024 SEASON
# --------------------------------------------------------

print("\n--- 2024 VALIDATION ---")

# Train on 2023 only
full_model = create_model(FEATURES)
full_model.fit(
    data_2023[FEATURES],
    data_2023["breakout"]
)

# Compare against the simpler OPS-only model
ops_features = ["recent_ops"]

ops_model = create_model(ops_features)
ops_model.fit(
    data_2023[ops_features],
    data_2023["breakout"]
)

evaluate(
    full_model, data_2024, FEATURES,
    "Full Model - 2024"
)

evaluate(
    ops_model, data_2024, ops_features,
    "OPS-Only Model - 2024"
)


# --------------------------------------------------------
#               FINAL 2025 EVALUATION
# --------------------------------------------------------

# Combine earlier seasons for training
training_data = pd.concat(
    [data_2023, data_2024],
    ignore_index=True
)

# Retrain on both historical seasons
full_model = create_model(FEATURES)
full_model.fit(
    training_data[FEATURES],
    training_data["breakout"]
)

ops_model = create_model(ops_features)
ops_model.fit(
    training_data[ops_features],
    training_data["breakout"]
)

print("\n--- 2025 TEST RESULTS ---")

evaluate(
    full_model, data_2025, FEATURES,
    "Full BreakoutIQ Model - 2025"
)

evaluate(
    ops_model, data_2025, ops_features,
    "OPS-Only Model - 2025"
)
