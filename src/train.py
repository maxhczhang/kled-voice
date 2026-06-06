from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.dataset import build_feature_matrix, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline audio fraud detector.")
    parser.add_argument("--manifest", default="data/manifest.csv", help="CSV with path,label,source columns.")
    parser.add_argument("--model-out", default="models/baseline.joblib", help="Where to save the trained model.")
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
        help="Baseline classifier to train.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    examples = load_manifest(args.manifest)
    X, labels, names = build_feature_matrix(examples)

    unique_labels = sorted(set(labels))
    print(f"Loaded {len(labels)} examples with labels: {', '.join(unique_labels)}")
    print(f"Extracted {len(names)} acoustic features")

    if len(unique_labels) < 2:
        raise SystemExit(
            "Need at least two labels to train a classifier. "
            "Add real and synthetic speech examples to data/manifest.csv."
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    if not _can_stratify(y):
        raise SystemExit(
            "Need at least two examples per label for a held-out evaluation split. "
            "Add more clips before training."
        )

    stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = _make_model(args.model_type, args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = _positive_class_scores(model, X_test)

    print_metrics(y_test, y_pred, y_score, label_encoder)

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_names": names,
            "model_type": args.model_type,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    label_encoder: LabelEncoder,
) -> None:
    labels = label_encoder.classes_
    print("\nMetrics")
    print(f"accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"f1:        {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    if y_score is not None and len(labels) == 2:
        print(f"roc_auc:   {roc_auc_score(y_true, y_score):.4f}")

    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print("Confusion matrix")
    print(confusion_matrix(y_true, y_pred))


def _make_model(model_type: str, random_state: int) -> Pipeline:
    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=2_000, class_weight="balanced")
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def _positive_class_scores(model: Pipeline, X_test: np.ndarray) -> np.ndarray | None:
    classifier = model.named_steps["classifier"]
    if len(classifier.classes_) != 2:
        return None

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return None


def _can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return bool(np.all(counts >= 2))


if __name__ == "__main__":
    main()
