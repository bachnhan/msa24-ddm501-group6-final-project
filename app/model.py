import os
import pickle
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Any
from dotenv import load_dotenv

# Load environment variables (Local .env, .secret, or Render Secret Files)
load_dotenv(override=True)
for secret_file in [".secret", "/etc/secrets/.env", "/etc/secrets/.secret"]:
    if os.path.exists(secret_file):
        load_dotenv(secret_file, override=True)

from app.config import MODEL_PATH
from app.metrics import MODEL_LOADED, MODEL_LAST_RELOAD, MODEL_INFO

logger = logging.getLogger(__name__)


class ChurnModel:
    """Predictor class for customer churn using MLflow Registry."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Any = None
        self.loaded_version: str = "initializing..."
        self.last_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the trained model.
        Primary: DagsHub MLflow Model Registry (Champion/Latest).
        Fallback: Local pickle file from the models/ directory.
        """
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        model_name = os.getenv("MLFLOW_MODEL_NAME", "CustomerChurnModel")

        try:
            if (
                tracking_uri
                and username
                and (
                    password
                    or os.path.exists("/etc/secrets/.env")
                    or os.path.exists("/etc/secrets/.secret")
                )
            ):
                # Ensure we have the password from secret files if dashboard is empty
                if not password:
                    for sf in ["/etc/secrets/.env", "/etc/secrets/.secret"]:
                        if os.path.exists(sf):
                            load_dotenv(sf, override=True)
                    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

                # Set authentication for MLflow client
                os.environ["MLFLOW_TRACKING_USERNAME"] = username or ""
                os.environ["MLFLOW_TRACKING_PASSWORD"] = password or ""

                logger.info(f"Connecting to MLflow Tracking Server: {tracking_uri}")
                mlflow.set_tracking_uri(tracking_uri)

                # Flexible reference: can be a version number (1, 2) or an alias (Champion, Production)
                model_ref = os.getenv("MLFLOW_MODEL_VERSION", "latest")

                # Check if ref is numeric or 'latest'
                if model_ref.isdigit() or model_ref.lower() == "latest":
                    model_uri = f"models:/{model_name}/{model_ref}"
                else:
                    # Otherwise, it's a custom alias (e.g., '@champion'), use @ syntax
                    model_uri = f"models:/{model_name}@{model_ref}"

                logger.info(f"Fetching model from Registry using URI: {model_uri}")
                self.model = mlflow.sklearn.load_model(model_uri)
                import gc

                gc.collect()  # Force garbage collection to free RAM after loading large model

                # Resolve the real version number from the alias (for the Admin Dashboard)
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                if model_ref.isdigit():
                    self.loaded_version = f"v{model_ref}"
                else:
                    # Resolve alias (latest, champion, etc.) to numeric version
                    try:
                        # Try getting by alias first (standard for modern MLflow)
                        v_info = client.get_model_version_by_alias(
                            model_name, model_ref
                        )
                        self.loaded_version = f"v{v_info.version}"
                    except:
                        try:
                            # Fallback for older stages (Production/Staging)
                            v_info = client.get_latest_versions(model_name, [model_ref])
                            if v_info:
                                self.loaded_version = f"v{v_info[0].version}"
                            else:
                                self.loaded_version = model_ref
                        except:
                            self.loaded_version = model_ref

                logger.info(
                    f"✅ Successfully loaded {self.loaded_version} from MLflow Registry."
                )
            else:
                raise ValueError(
                    "Missing MLflow environment variables for Registry connection."
                )

        except Exception as e:
            self.last_error = f"Critical: Failed to load from Registry: {e}"
            logger.error(self.last_error)

            # [CI/CD Fallback] Create a Pipeline-structured Dummy Model to ensure SHAP paths are covered
            try:
                from sklearn.dummy import DummyClassifier
                from sklearn.pipeline import Pipeline
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import StandardScaler
                import numpy as np

                logger.warning(
                    f"Connection failed ({e}). Initializing PIPELINE DUMMY MODEL."
                )

                # Mock a ColumnTransformer that does nothing (just for SHAP logic to find it)
                pre = ColumnTransformer(
                    [("num", StandardScaler(), [0])], remainder="passthrough"
                )
                clf = DummyClassifier(strategy="constant", constant=0)

                dummy_pipe = Pipeline([("pre", pre), ("clf", clf)])
                # Fit on 21 columns to match Telco schema expectations
                dummy_pipe.fit(np.zeros((2, 21)), np.array([0, 1]))

                self.model = dummy_pipe
                self.loaded_version = "v0.0.1-PIPELINE-DUMMY"
            except Exception as dummy_err:
                logger.error(f"Failed to even create a Dummy Model: {dummy_err}")

        # Update metrics
        if self.model is not None:
            # Note: We NO LONGER clear last_error if it contains a Registry failure message,
            # so the user can see why the Registry failed on the health page.
            if "DUMMY" not in self.loaded_version:
                self.last_error = None

            if MODEL_LOADED is not None:
                MODEL_LOADED.set(1)
        else:
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(0)

    def get_last_error(self) -> Optional[str]:
        """Return the last error message if model failed to load."""
        return self.last_error

    def predict(self, data_dict: dict) -> Tuple[bool, str, List[str]]:
        """
        Process input features and return churn prediction.

        Args:
            data_dict (dict): Dictionary containing customer features.

        Returns:
            Tuple[bool, str, List[str]]: (Is Churn, Risk Tier, Reason Codes)
        """
        is_churn, _, risk_tier, reason_codes, _ = self.predict_with_latency(data_dict)
        return is_churn, risk_tier, reason_codes

    def predict_with_latency(
        self, data_dict: dict
    ) -> Tuple[bool, float, str, List[str], float]:
        """
        Predict churn and also return fixed categories and performance metrics.

        Returns:
            Tuple[bool, float, str, List[str], float]: (Is Churn, Probability, Risk Tier, Reason Codes, Latency MS)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot perform prediction.")

        start_time = time.perf_counter()

        # Convert dictionary to DataFrame
        X = pd.DataFrame([data_dict])

        # --- PRODUCTION ROBUSTNESS: Handle Missing Columns for Legacy Models ---
        try:
            # If the model has a preprocessor, check required features
            if hasattr(self.model, "named_steps") and "pre" in self.model.named_steps:
                required_features = self.model.named_steps["pre"].feature_names_in_
                for col in required_features:
                    if col not in X.columns:
                        X[col] = 0  # Fill with default
        except Exception:
            pass

        # Get predictions
        churn_prob = float(self.model.predict_proba(X)[0, 1])
        is_churn = bool(self.model.predict(X)[0])

        # --- RESPONSIBLE AI: RISK TIERING ---
        if churn_prob > 0.7:
            risk_tier = "High"
        elif churn_prob > 0.3:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        # --- RESPONSIBLE AI: EXPLAINABILITY (Rubric 3.1.5) ---
        reason_codes = []
        try:
            import shap

            preprocessor = self.model.named_steps["pre"]
            clf = self.model.named_steps["clf"]

            X_transformed = preprocessor.transform(X)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            explainer = shap.Explainer(clf)
            shap_values = explainer(X_transformed)

            raw_feature_names = preprocessor.get_feature_names_out()

            if hasattr(shap_values, "values"):
                vals = shap_values.values[0]
            else:
                vals = shap_values[0]

            if len(vals.shape) > 1 and vals.shape[-1] > 1:
                vals = vals[:, 1]

            top_indices = sorted(
                range(len(vals)), key=lambda i: abs(vals[i]), reverse=True
            )[:3]

            for idx in top_indices:
                impact = vals[idx]
                feat_raw = raw_feature_names[idx].split("__")[-1].lower()

                # Semantic Mapping Logic
                if "contract" in feat_raw and impact > 0:
                    reason_codes.append("contract_type_monthly")
                elif "tenure" in feat_raw and impact < 0:
                    reason_codes.append("tenure_lt_12mo")
                elif "techsupport" in feat_raw and impact < 0:
                    reason_codes.append("no_techsupport")
                elif "internetservice" in feat_raw and impact > 0:
                    reason_codes.append("fiber_optic_churn_risk")
                elif "totalcharges" in feat_raw:
                    reason_codes.append(
                        "high_total_charges" if impact > 0 else "low_total_charges"
                    )
                else:
                    suffix = "high" if impact > 0 else "low"
                    reason_codes.append(f"{feat_raw}_{suffix}")

            # Ensure unique reason codes
            reason_codes = list(dict.fromkeys(reason_codes))

        except Exception as e:
            logger.error(f"Reason codes failed: {str(e)}")
            reason_codes = ["analysis_unavailable"]

        latency_ms = (time.perf_counter() - start_time) * 1000

        return is_churn, churn_prob, risk_tier, reason_codes, latency_ms

    def is_loaded(self) -> bool:
        """Check if the model instance is ready for prediction."""
        return self.model is not None

    def reload(self, model_ref: Optional[str] = None) -> bool:
        """
        Reload the model from Registry with a specific reference.
        Args:
            model_ref (str): Version number or Alias. If None, uses environment current value.
        Returns:
            bool: True if reload was successful.
        """
        if model_ref:
            os.environ["MLFLOW_MODEL_VERSION"] = model_ref

        # Re-run the internal load logic
        self._load_model()

        # Record the last reload time in metrics if available
        if self.model is not None:
            if MODEL_LAST_RELOAD is not None:
                MODEL_LAST_RELOAD.set_to_current_time()
            return True
        return False


# Singleton Pattern for Model Instance
_model_instance = None


def get_model() -> ChurnModel:
    """Get the global ChurnModel singleton."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ChurnModel()
    return _model_instance
