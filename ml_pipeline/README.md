# ML pipeline

The pipeline trains binary and multi-label models from `features/extracted_features.csv`. It normalizes video names, aggregates duplicate annotation rows, preserves all applicable labels, and separates benchmark reporting from production fitting.

## Standard training

```powershell
python train_classifier.py `
  --config config.yaml `
  --features "features/extracted_features.csv" `
  --annotations "C:\path\to\detailed_annotations.csv" `
  --mode both
```

`both` reports an honest out-of-fold estimate, a separate benchmark holdout, and then fits the production model on all labeled examples. With only 42 labeled videos and rare classes, these metrics should not be presented as a guarantee for hidden jury footage.

## Small-data ensemble

For the strongest available fit on the current labeled feature set, run the ensemble after standard training:

```powershell
python train_ensemble.py `
  --config config.yaml `
  --features "features/extracted_features.csv" `
  --annotations "C:\path\to\detailed_annotations.csv"
```

The ensemble uses out-of-fold calibration, multiple regularized linear/tree estimators, and a final fit on all labeled examples. It never reads hidden jury videos. The API automatically uses `ensemble_model.pkl` when present and falls back to the neural checkpoints otherwise.

## Artifacts

The training commands write neural checkpoints, the scaler, multi-label mapping, metadata, and (after the ensemble command) `ensemble_model.pkl`. Keep `model_metadata.json` with the artifacts used by the API.

## Runtime

`backend/feature_extractor.py` produces the same numeric feature schema from an uploaded video. `backend/ml_classifier.py` applies the ensemble when available and otherwise uses the neural checkpoints. The API merges ML predictions with temporal evidence frames.

## Limitations

The dataset contains 42 labeled videos and several classes with one example. Extra epochs cannot create missing visual variation. Additional labeled footage is the highest-value improvement for reliable performance on unseen jury videos.

## Reproducibility

The ensemble uses fixed random seeds and fixed fold construction. The OOF estimate is the appropriate generalization signal; benchmark results on known videos are reported separately.
