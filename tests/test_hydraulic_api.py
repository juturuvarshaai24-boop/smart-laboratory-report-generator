"""Tests for the hydraulic FastAPI inference service."""

import unittest

from fastapi.testclient import TestClient

from src.hydraulic.api import app, FEATURE_NAMES
from src.hydraulic.data import load_hydraulic_dataset


class HydraulicAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        dataset = load_hydraulic_dataset()
        row = dataset.model_features().iloc[0]
        cls.features = {name: float(row[name]) for name in FEATURE_NAMES}

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_model_info(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["feature_count"], 112)

    def test_valid_prediction(self):
        response = self.client.post(
            "/predict", json={"features": self.features}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["prediction"], [0, 1])
        self.assertGreaterEqual(response.json()["leakage_probability"], 0)
        self.assertLessEqual(response.json()["leakage_probability"], 1)

    def test_missing_features_rejected(self):
        response = self.client.post("/predict", json={"features": {}})
        self.assertEqual(response.status_code, 422)

    def test_unexpected_feature_rejected(self):
        features = dict(self.features)
        features["UNKNOWN_SENSOR"] = 1.0
        response = self.client.post("/predict", json={"features": features})
        self.assertEqual(response.status_code, 422)

    def test_non_finite_feature_rejected(self):
        features = dict(self.features)
        features[FEATURE_NAMES[0]] = "NaN"
        response = self.client.post("/predict", json={"features": features})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
