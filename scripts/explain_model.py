"""
Explainability analysis for the Customer Churn Prediction System.
Uses feature importance from the RandomForest model.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def explain_churn(model_path="models/churn_model.pkl"):
    import mlflow.sklearn

    # Try loading from local, otherwise help user load from Registry
    if not Path(model_path).exists():
        print(
            "💡 Local model artifact not found. Please run Training script first or use MLflow Registry."
        )
        return

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Get components from our updated pipeline structure
    clf = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["pre"]

    # Resolve feature names
    feature_names = pre.get_feature_names_out()

    # Handle different model types
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        print("Model type not supported for simple importance extraction.")
        return

    results = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    results = results.sort_values(by="Importance", ascending=False)

    print("\n" + "=" * 40)
    print("TELCO FEATURE IMPORTANCE (GLOBAL)")
    print("=" * 40)
    print(results.head(10))
    print("=" * 40)
    print(f"\nInterpretation: Top churn driver is {results.iloc[0]['Feature']}")


if __name__ == "__main__":
    explain_churn()
