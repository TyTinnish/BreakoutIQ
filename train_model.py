import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    brier_score_loss
)

# --------------------------------------------------------
#                     LOAD DATA
# --------------------------------------------------------

data = pd.read_csv("data/training_2025.csv")

# Features available before the prediction window
FEATURES = [
    "avg_exit_velocity",
    "hard_hit_rate",
    "strikeout_rate",
    "walk_rate",
    "plate_appearances",
    "batted_balls",
    "recent_ops"
]

X = data[FEATURES]
y = data["breakout"]

# Keep each player's examples together
groups = data["batter"]

# --------------------------------------------------------
#                    SPLIT DATA
# --------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")

# Confirm no player is shared between groups
train_players = set(groups.iloc[train_idx])
test_players = set(groups.iloc[test_idx])

assert train_players.isdisjoint(test_players)

# --------------------------------------------------------
#                   TRAIN THE MODEL
# --------------------------------------------------------

# StandardScaler normalizes features using training data
# LogisticRegression learns relationships with breakout labels
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

# --------------------------------------------------------
#                    EVALUATE MODEL
# --------------------------------------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n--- MODEL RESULTS ---")

print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")

if y_test.nunique() == 2:
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

print(f"Brier Score: {brier_score_loss(y_test, probabilities):.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Compare against predicting no breakout for everyone
baseline_predictions = [0] * len(y_test)

print(
    "Always-no-breakout accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 3)
)

# --------------------------------------------------------
#                  FEATURE COEFFICIENTS
# --------------------------------------------------------

coefficients = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": model.named_steps[
        "logisticregression"
    ].coef_[0]
})

print("\nFeature Coefficients:")
print(
    coefficients.sort_values(
        "coefficient",
        ascending=False
    ).to_string(index=False)
)

# --------------------------------------------------------
#             COMPARE AGAINST OPS-ONLY MODEL
# --------------------------------------------------------

# Build a simpler model using only recent OPS
baseline_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# Train on the same players as the full model
baseline_model.fit(
    X_train[["recent_ops"]],
    y_train
)

# Predict using the OPS-only model
baseline_probs = baseline_model.predict_proba(
    X_test[["recent_ops"]]
)[:, 1]

baseline_preds = baseline_model.predict(
    X_test[["recent_ops"]]
)

print("\n--- MODEL COMPARISON ---")

print("Full BreakoutIQ model:")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
print(f"Brier: {brier_score_loss(y_test, probabilities):.3f}")

print("\nOPS-only model:")
print(f"Accuracy: {accuracy_score(y_test, baseline_preds):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, baseline_probs):.3f}")
print(f"Brier: {brier_score_loss(y_test, baseline_probs):.3f}")
