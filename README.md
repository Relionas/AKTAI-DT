# ForteBank Violation Detector

**Team:** AKTAI  
**Event:** ForteBank Datathon 2026  
**Deployment:** local-only Windows package

## Overview

ForteBank Violation Detector is a local computer-vision service for reviewing cashier-operation videos. It accepts `.mov` files and returns a binary verdict, possible violation types, timestamps, descriptions, and JPEG evidence frames encoded as Base64.

The production inference path combines temporal YOLOv8 analysis with the trained binary and multi-label feature models. All processing is local; video and frames are not sent to external AI services. The result is an explainable review aid for a human operator, not a final legal or disciplinary conclusion.

## Local submission values

| Field | Value |
|---|---|
| Web interface URL | `file:///C:/Users/vanac/OneDrive/Desktop/FINAL_PROJECT/frontend/index.html` |
| API base URL | `http://localhost:8000/api/v1` |
| API key for the jury | `fortebank-aktai-2026` |
| Git Link | `Not available — local submission package` |

`localhost` and the `file:///` URL are reachable only on the computer running this package. A public Web/API URL or Git repository must be supplied separately if the jury requires remote access.

## Requirements

Recommended local environment: Windows 10/11, Python 3.10+, an NVIDIA GPU with CUDA support, and sufficient disk space for the model weights and temporary video files. CPU execution is possible but substantially slower.

## Run locally

From the project root in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Remove-Item Env:FORTEBANK_BENCHMARK_GROUND_TRUTH -ErrorAction SilentlyContinue
python backend\api.py
```

The API listens on `http://localhost:8000`. In a second PowerShell window, verify the service:

```powershell
Invoke-RestMethod http://localhost:8000/health | Format-List
```

The expected production state is `status=healthy`, `ml_classifier_loaded=True`, and `benchmark_mode=False`.

## Web interface

Open `frontend/index.html` after starting the API. If the browser blocks requests from a local file, serve the project folder with Python:

```powershell
python -m http.server 8080
```

Then open `http://localhost:8080/frontend/index.html`.

## API usage

Upload a video:

```powershell
$upload = curl.exe -s -X POST `
  -H "X-API-Key: fortebank-aktai-2026" `
  -F "file=@C:\path\to\video.mov" `
  http://localhost:8000/api/v1/analyze | ConvertFrom-Json

$jobId = $upload.job_id
```

Poll until complete:

```powershell
do {
    Start-Sleep -Seconds 10
    $result = Invoke-RestMethod `
      -Uri "http://localhost:8000/api/v1/jobs/$jobId" `
      -Headers @{"X-API-Key"="fortebank-aktai-2026"}
} while ($result.status -eq "processing")

$result | Select-Object job_id, status, verdict, ml_violation_probability, inference_mode | Format-List
```

The public endpoints are `POST /api/v1/analyze` and `GET /api/v1/jobs/{job_id}`. The response contract contains `verdict` (`violation` or `no_violation`) and, when applicable, `violations` with `type`, `timecode_sec`, `description`, and `evidence_frames`.

## Training pipeline

The extracted feature matrix is already available at `ml_pipeline/features/extracted_features.csv`. To retrain both the benchmark and production models:

```powershell
cd ml_pipeline
python train_classifier.py `
  --config config.yaml `
  --features "features/extracted_features.csv" `
  --annotations "C:\Users\vanac\OneDrive\Desktop\datathon problem\detailed_annotations.csv" `
  --mode both
```

`both` reports a holdout benchmark and an out-of-fold estimate, then fits the production model on all labeled examples using out-of-fold-calibrated thresholds. The out-of-fold estimate is the more honest indicator of generalization; neither metric guarantees performance on hidden jury videos.

## Model artifacts

`ml_pipeline/models/` contains `binary_model.pth`, `multilabel_model.pth`, `best_model.pth` (legacy alias), `scaler.pkl`, `multilabel_binarizer.pkl`, `label_encoder.pkl`, and `model_metadata.json`.

## Verification checklist

Run syntax checks before submission:

```powershell
python -m py_compile backend\api.py backend\ml_classifier.py ml_pipeline\feature_extractor.py ml_pipeline\train_classifier.py
Invoke-RestMethod http://localhost:8000/health | Format-List
```

For a smoke test, use a known labeled video such as `Х-30.mov` and verify that the response contains a completed verdict, both expected types where applicable, timestamps, and at least one evidence frame per returned type.

## Scope and limitations

The labeled dataset contains 42 videos and several rare violation classes. The system therefore reports possible violations for human review and does not claim guaranteed 100% accuracy on unseen footage. Ground-truth lookup is disabled by default in the jury path. It can only be enabled explicitly for a local benchmark with `FORTEBANK_BENCHMARK_GROUND_TRUTH=1`; that mode must not be used for jury scoring.

## Documentation

- `FINAL_SUBMISSION.md` — values and handoff checklist for the jury.
- `docs/FOR_JURY.md` — local demonstration instructions.
- `docs/API_GUIDE.md` — API contract and examples.
- `docs/TECHNICAL_DOCS.md` — architecture and limitations.
- `docs/DEPLOYMENT.md` — local deployment notes.
- `ml_pipeline/README.md` — training and model artifacts.
