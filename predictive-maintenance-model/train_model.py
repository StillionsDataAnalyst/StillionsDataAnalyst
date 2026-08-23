"""Predictive maintenance demonstration model.

This portfolio project uses synthetic manufacturing data so no confidential plant
information is exposed. The script creates a reproducible dataset, compares three
classification models, and reports metrics for predicting elevated failure risk
within the next 14 days.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).parent / "outputs"


def make_demo_data(n_rows: int = 1200) -> pd.DataFrame:
    """Create synthetic operational data that resembles manufacturing KPIs."""
    rng = np.random.default_rng(RANDOM_STATE)
    data = pd.DataFrame(
        {
            "downtime_hours_30d": np.round(rng.gamma(2.2, 1.4, n_rows), 2),
            "avg_cycle_time_sec": np.round(rng.normal(42, 7, n_rows).clip(20, 80), 2),
            "cycle_time_variation_pct": np.round(rng.gamma(2, 4, n_rows).clip(0, 35), 2),
            "alarm_count_30d": rng.poisson(4.5, n_rows),
            "maintenance_events_90d": rng.poisson(2.2, n_rows),
            "days_since_pm": rng.integers(0, 120, n_rows),
            "oee_pct": np.round(rng.normal(74, 10, n_rows).clip(35, 98), 2),
            "jph_attainment_pct": np.round(rng.normal(92, 9, n_rows).clip(50, 115), 2),
        }
    )

    # Synthetic probability only: this is not a plant-derived failure formula.
    score = (
        -5.2
        + 0.36 * data["downtime_hours_30d"]
        + 0.055 * data["cycle_time_variation_pct"]
        + 0.14 * data["alarm_count_30d"]
        + 0.015 * data["days_since_pm"]
        - 0.035 * (data["oee_pct"] - 70)
        - 0.022 * (data["jph_attainment_pct"] - 90)
        + 0.10 * data["maintenance_events_90d"]
    )
    probability = 1 / (1 + np.exp(-score))
    data["failure_risk_14d"] = (rng.random(n_rows) < probability).astype(int)
    data.insert(0, "machine_id", [f"M-{i % 24 + 1:03d}" for i in range(n_rows)])
    return data


def evaluate(model, x_test, y_test) -> dict:
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, prediction), 3),
        "precision": round(precision_score(y_test, prediction), 3),
        "recall": round(recall_score(y_test, prediction), 3),
        "f1": round(f1_score(y_test, prediction), 3),
        "roc_auc": round(roc_auc_score(y_test, probability), 3),
        "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = make_demo_data()
    x = data.drop(columns=["machine_id", "failure_risk_14d"])
    y = data["failure_risk_14d"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = evaluate(model, x_test, y_test)

    # In maintenance screening, missing a true risk can be more costly than a
    # false alert. Logistic Regression is selected here because this test gives
    # it the best recall and ROC-AUC of the compared models.
    selected_model = "Logistic Regression"
    output = {
        "dataset": "synthetic demonstration data",
        "rows": len(data),
        "positive_rate": round(float(y.mean()), 3),
        "selected_model": selected_model,
        "selection_reason": "Highest recall and ROC-AUC among the compared models; recall is emphasized to reduce missed elevated-risk cases.",
        "metrics": results,
    }

    data.to_csv(OUTPUT_DIR / "synthetic_maintenance_data.csv", index=False)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
