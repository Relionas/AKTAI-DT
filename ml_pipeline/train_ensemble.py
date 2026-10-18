"""Train a small-data ensemble for the production inference path.

This script never reads jury test videos. It uses only the labeled training
features and annotations, performs out-of-fold threshold calibration, then fits
an ensemble on all available labeled samples.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from train_classifier import load_dataset, normalize_video_name


def safe_probability(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    decision = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -40, 40)))


def make_binary_models(seed):
    return [
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=seed),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed),
        ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
    ]


def make_type_model(seed, positive_count, negative_count):
    if positive_count >= 2 and negative_count >= 2:
        return LogisticRegression(C=0.2, class_weight="balanced", max_iter=3000, random_state=seed)
    if positive_count >= 1 and negative_count >= 2:
        return ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=seed, n_jobs=-1)
    return None


def blend_binary(models, x):
    return np.mean([safe_probability(model, x) for model in models], axis=0)


def threshold_for(y, p):
    if int(y.sum()) < 2 or int((y == 0).sum()) < 2:
        return 0.5
    candidates = np.linspace(0.25, 0.75, 21)
    scores = [f1_score(y, p >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--features", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    models_dir = Path(args.models_dir or config["data"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.features)
    feature_columns = [c for c in frame.columns if c not in {"video_name", "duration", "num_frames"}]
    x, labels, names = load_dataset(args.features, args.annotations)
    configured = [str(v) for v in config.get("violation_types", [])]
    classes = list(dict.fromkeys(configured + sorted({v for row in labels for v in row})))
    mlb = MultiLabelBinarizer(classes=classes)
    y_multi = mlb.fit_transform(labels).astype(int)
    y_binary = (y_multi.sum(axis=1) > 0).astype(int)

    folds = min(5, int(y_binary.sum()), int((y_binary == 0).sum()))
    oof_binary = np.zeros(len(x), dtype=np.float32)
    oof_multi = np.zeros((len(x), len(classes)), dtype=np.float32)
    supported = [i for i in range(len(classes)) if y_multi[:, i].sum() >= 2]

    print("\n🔬 Training small-data ensemble with out-of-fold calibration...")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y_binary), 1):
        scaler = StandardScaler().fit(x[train_idx])
        x_train, x_test = scaler.transform(x[train_idx]), scaler.transform(x[test_idx])
        binary_models = make_binary_models(100 + fold)
        for model in binary_models:
            model.fit(x_train, y_binary[train_idx])
        oof_binary[test_idx] = blend_binary(binary_models, x_test)
        for class_index in supported:
            model = make_type_model(200 + fold, int(y_multi[train_idx, class_index].sum()), int(len(train_idx) - y_multi[train_idx, class_index].sum()))
            if model is not None:
                model.fit(x_train, y_multi[train_idx, class_index])
                oof_multi[test_idx, class_index] = safe_probability(model, x_test)
        print(f"   Fold {fold}/{folds} complete")

    binary_threshold = threshold_for(y_binary, oof_binary)
    multi_thresholds = np.asarray([threshold_for(y_multi[:, i], oof_multi[:, i]) if i in supported else 0.5 for i in range(len(classes))], dtype=np.float32)
    binary_f1 = f1_score(y_binary, oof_binary >= binary_threshold, zero_division=0)
    multi_f1 = f1_score(y_multi[:, supported], oof_multi[:, supported] >= multi_thresholds[supported], average="macro", zero_division=0) if supported else 0.0
    print(f"   OOF ensemble Binary F1: {binary_f1:.3f}")
    print(f"   OOF ensemble Multi-label macro F1 (supported classes): {multi_f1:.3f}")
    print(f"   Binary threshold: {binary_threshold:.2f}")

    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    binary_models = make_binary_models(999)
    for model in binary_models:
        model.fit(x_scaled, y_binary)
    type_models = {}
    for class_index in supported:
        model = make_type_model(1000 + class_index, int(y_multi[:, class_index].sum()), int(len(x) - y_multi[:, class_index].sum()))
        if model is not None:
            model.fit(x_scaled, y_multi[:, class_index])
            type_models[int(class_index)] = model

    payload = {
        "schema_version": 1,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "classes": list(mlb.classes_),
        "binary_models": binary_models,
        "type_models": type_models,
        "binary_threshold": binary_threshold,
        "multilabel_thresholds": multi_thresholds,
        "oof_metrics": {"binary_f1": float(binary_f1), "multilabel_macro_f1_supported": float(multi_f1)},
    }
    with open(models_dir / "ensemble_model.pkl", "wb") as handle:
        pickle.dump(payload, handle)
    with open(models_dir / "multilabel_binarizer.pkl", "wb") as handle:
        pickle.dump(mlb, handle)
    with open(models_dir / "scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)
    metadata = {
        "schema_version": 4,
        "task": "binary_violation_plus_multilabel_types",
        "input_dim": len(feature_columns),
        "classes": list(mlb.classes_),
        "ensemble_model": "ensemble_model.pkl",
        "binary_model": "binary_model.pth",
        "multilabel_model": "multilabel_model.pth",
        "binary_threshold": float(binary_threshold),
        "multilabel_thresholds": [float(v) for v in multi_thresholds],
        "cv_metrics": payload["oof_metrics"],
        "inference_mode": "production_ensemble_all_data_fit",
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Saved ensemble_model.pkl and updated model_metadata.json")


if __name__ == "__main__":
    main()
