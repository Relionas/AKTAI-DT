"""Inference wrapper for neural and small-data ensemble classifiers."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.25)])
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FeatureViolationClassifier:
    def __init__(self, models_dir: str | Path, device: Optional[str] = None):
        self.models_dir = Path(models_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with open(self.models_dir / "model_metadata.json", "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        with open(self.models_dir / "scaler.pkl", "rb") as handle:
            self.scaler = pickle.load(handle)
        with open(self.models_dir / "multilabel_binarizer.pkl", "rb") as handle:
            self.binarizer = pickle.load(handle)
        self.binary_threshold = float(metadata.get("binary_threshold", 0.5))
        self.multilabel_thresholds = np.asarray(metadata.get("multilabel_thresholds", [0.5] * len(self.binarizer.classes_)), dtype=np.float32)
        if len(self.multilabel_thresholds) != len(self.binarizer.classes_):
            self.multilabel_thresholds = np.full(len(self.binarizer.classes_), 0.5, dtype=np.float32)
        self.ensemble = None
        ensemble_path = self.models_dir / metadata.get("ensemble_model", "ensemble_model.pkl")
        if ensemble_path.exists():
            with open(ensemble_path, "rb") as handle:
                self.ensemble = pickle.load(handle)
        input_dim = int(metadata["input_dim"])
        self.binary_model = MLP(input_dim, 1).to(self.device)
        self.multilabel_model = MLP(input_dim, len(self.binarizer.classes_)).to(self.device)
        self.binary_model.load_state_dict(torch.load(self.models_dir / metadata["binary_model"], map_location=self.device))
        self.multilabel_model.load_state_dict(torch.load(self.models_dir / metadata["multilabel_model"], map_location=self.device))
        self.binary_model.eval(); self.multilabel_model.eval()

    @staticmethod
    def _probability(model, x):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]
        decision = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-np.clip(decision, -40, 40)))

    def predict(self, feature_vector) -> Dict:
        vector = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        expected = int(getattr(self.scaler, "n_features_in_", vector.shape[1]))
        if vector.shape[1] != expected:
            raise ValueError(f"Expected {expected} features, got {vector.shape[1]}")
        scaled = self.scaler.transform(vector).astype(np.float32)
        if self.ensemble is not None:
            binary_models = self.ensemble["binary_models"]
            binary_probability = float(np.mean([self._probability(model, scaled)[0] for model in binary_models]))
            type_probabilities = np.zeros(len(self.binarizer.classes_), dtype=np.float32)
            for index, model in self.ensemble.get("type_models", {}).items():
                type_probabilities[int(index)] = self._probability(model, scaled)[0]
            binary_threshold = float(self.ensemble.get("binary_threshold", self.binary_threshold))
            thresholds = np.asarray(self.ensemble.get("multilabel_thresholds", self.multilabel_thresholds), dtype=np.float32)
        else:
            tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                binary_probability = float(torch.sigmoid(self.binary_model(tensor)).item())
                type_probabilities = torch.sigmoid(self.multilabel_model(tensor)).cpu().numpy()[0]
            binary_threshold, thresholds = self.binary_threshold, self.multilabel_thresholds
        has_violation = binary_probability >= binary_threshold
        violations: List[Dict] = []
        if has_violation:
            for label, probability, threshold in zip(self.binarizer.classes_, type_probabilities, thresholds):
                if probability >= threshold:
                    violations.append({"type": str(label), "confidence": round(float(probability), 4)})
            if not violations:
                best = int(np.argmax(type_probabilities))
                violations.append({"type": str(self.binarizer.classes_[best]), "confidence": round(float(type_probabilities[best]), 4)})
        return {"verdict": "violation" if has_violation else "no_violation", "violation_probability": round(binary_probability, 4), "violations": violations}


def load_if_available(models_dir: str | Path):
    try:
        return FeatureViolationClassifier(models_dir)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, pickle.UnpicklingError) as exc:
        print(f"⚠️ ML classifier unavailable: {exc}")
        return None


__all__ = ["FeatureViolationClassifier", "load_if_available"]
