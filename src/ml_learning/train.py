from pathlib import Path
from typing import TypedDict

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "iris_classifier.joblib"


class ModelBundle(TypedDict):
    model: DecisionTreeClassifier
    feature_names: list[str]
    target_names: list[str]
    selected_depth: int
    cv_accuracy: float
    cv_macro_f1: float


def train_and_save_model(output_path: Path = MODEL_PATH) -> ModelBundle:
    iris = load_iris(as_frame=True)
    X_train, _, y_train, _ = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates: list[tuple[float, int, float]] = []
    for depth in range(1, 9):
        scores = cross_validate(
            DecisionTreeClassifier(max_depth=depth, random_state=42),
            X_train,
            y_train,
            cv=cross_validation,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        )
        candidates.append(
            (
                float(scores["test_accuracy"].mean()),
                depth,
                float(scores["test_macro_f1"].mean()),
            )
        )

    cv_accuracy, selected_depth, cv_macro_f1 = max(
        candidates, key=lambda result: (result[0], -result[1])
    )
    model = DecisionTreeClassifier(max_depth=selected_depth, random_state=42)
    model.fit(X_train, y_train)

    bundle: ModelBundle = {
        "model": model,
        "feature_names": list(iris.feature_names),
        "target_names": [str(name) for name in iris.target_names],
        "selected_depth": selected_depth,
        "cv_accuracy": cv_accuracy,
        "cv_macro_f1": cv_macro_f1,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return bundle


if __name__ == "__main__":
    trained_bundle = train_and_save_model()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Selected tree depth: {trained_bundle['selected_depth']}")
    print(f"Mean cross-validation accuracy: {trained_bundle['cv_accuracy']:.1%}")
    print(f"Mean cross-validation macro F1: {trained_bundle['cv_macro_f1']:.1%}")
