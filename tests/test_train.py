import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.datasets import load_iris

from ml_learning.train import train_and_save_model


class TrainModelTests(unittest.TestCase):
    def test_training_saves_a_model_that_can_predict_iris_species(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "iris_classifier.joblib"
            trained_bundle = train_and_save_model(model_path)

            self.assertTrue(model_path.is_file())
            loaded_bundle = joblib.load(model_path)
            iris = load_iris(as_frame=True)
            predictions = loaded_bundle["model"].predict(iris.data.iloc[:5])

            self.assertEqual(loaded_bundle["feature_names"], list(iris.feature_names))
            self.assertEqual(loaded_bundle["target_names"], [str(name) for name in iris.target_names])
            self.assertEqual(len(predictions), 5)
            self.assertTrue(set(predictions).issubset(set(iris.target.unique())))
            self.assertEqual(
                loaded_bundle["selected_depth"],
                trained_bundle["selected_depth"],
            )


if __name__ == "__main__":
    unittest.main()
