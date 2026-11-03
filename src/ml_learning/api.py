from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
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


class IrisFeatures(BaseModel):
    sepal_length_cm: float = Field(gt=0, description="Sepal length in centimeters")
    sepal_width_cm: float = Field(gt=0, description="Sepal width in centimeters")
    petal_length_cm: float = Field(gt=0, description="Petal length in centimeters")
    petal_width_cm: float = Field(gt=0, description="Petal width in centimeters")


class IrisPrediction(BaseModel):
    species: str
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Train it first with `python src/ml_learning/train.py`."
        )

    app.state.model_bundle = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Iris Classifier API",
    description="Predict an Iris species from its sepal and petal measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Iris Classifier API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=IrisPrediction)
def predict_species(features: IrisFeatures, request: Request) -> IrisPrediction:
    bundle: ModelBundle = request.app.state.model_bundle
    feature_values = {
        "sepal length (cm)": features.sepal_length_cm,
        "sepal width (cm)": features.sepal_width_cm,
        "petal length (cm)": features.petal_length_cm,
        "petal width (cm)": features.petal_width_cm,
    }
    model_input = pd.DataFrame(
        [[feature_values[name] for name in bundle["feature_names"]]],
        columns=bundle["feature_names"],
    )
    model = bundle["model"]
    predicted_class = int(model.predict(model_input)[0])
    class_position = list(model.classes_).index(predicted_class)
    confidence = float(model.predict_proba(model_input)[0][class_position])

    return IrisPrediction(
        species=bundle["target_names"][predicted_class],
        confidence=confidence,
    )
