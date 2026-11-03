import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from ml_learning import api
from ml_learning.train import train_and_save_model


class IrisApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temporary_directory.name) / "iris_classifier.joblib"
        train_and_save_model(self.model_path)
        self.model_path_patch = patch.object(api, "MODEL_PATH", self.model_path)
        self.model_path_patch.start()
        self.client_context = TestClient(api.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.model_path_patch.stop()
        self.temporary_directory.cleanup()

    def test_health_and_prediction_endpoints(self) -> None:
        root_response = self.client.get("/")
        health_response = self.client.get("/health")
        prediction_response = self.client.post(
            "/predict",
            json={
                "sepal_length_cm": 5.1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            },
        )

        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(root_response.json()["docs"], "/docs")
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(prediction_response.status_code, 200)
        self.assertEqual(prediction_response.json()["species"], "setosa")
        self.assertGreaterEqual(prediction_response.json()["confidence"], 0)
        self.assertLessEqual(prediction_response.json()["confidence"], 1)

    def test_prediction_rejects_non_positive_measurements(self) -> None:
        response = self.client.post(
            "/predict",
            json={
                "sepal_length_cm": 0,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_startup_explains_how_to_create_a_missing_model(self) -> None:
        missing_model_path = self.model_path.with_name("missing_model.joblib")

        with patch.object(api, "MODEL_PATH", missing_model_path):
            with self.assertRaisesRegex(RuntimeError, "Train it first"):
                with TestClient(api.app):
                    pass


if __name__ == "__main__":
    unittest.main()
