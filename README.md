# ForteBank Violation Detector — jury handoff

## What this package is

This is a local computer-vision service for reviewing cashier-operation videos. The operator uploads a `.mov` file and receives an asynchronous result containing a binary verdict, possible violation types, confidence, timecodes, explanations, normative references, and JPEG evidence frames.

The service is an assistive review tool. It reports visual indicators for human verification and does not make a final legal or disciplinary determination.

## Submission values

| Field | Local value |
|---|---|
| Web interface URL | `http://localhost:8080/frontend/index.html` |
| API base URL | `http://localhost:8000/api/v1` |
| API key for the jury | `fortebank-aktai-2026` |
| Git Link | `https://github.com/Relionas/AKTAI-DT` |

These addresses work on the computer running the service. A remote jury requires an approved reachable host; `localhost` is not accessible from another computer.

## Project structure

```text
AKTAI DT/
├── backend/
│   ├── api.py                 FastAPI service and asynchronous job flow
│   ├── optimal_detector.py    YOLO temporal detector and evidence generator
│   ├── ml_classifier.py       Neural/ensemble model loader and inference
│   ├── violation_catalog.py   Jury-facing descriptions and references
│   ├── requirements.txt       Runtime dependencies
│   └── yolov8n.pt             YOLO weights
├── frontend/
│   └── index.html             Browser interface
├── ml_pipeline/
│   ├── feature_extractor.py  111-feature extraction
│   ├── train_classifier.py    Binary + multi-label training
│   ├── train_ensemble.py      Small-data ensemble training
│   ├── features/              Extracted feature matrix
│   └── models/                Production model artifacts
├── data/
│   ├── uploads/               Temporary uploaded videos; ignored by Git
│   ├── results/               Temporary JSON results; ignored by Git
│   └── evidence_frames/       Runtime evidence cache; ignored by Git
├── docs/
│   ├── FOR_JURY.md            Jury walkthrough
│   ├── API_GUIDE.md           API contract
│   ├── DEPLOYMENT.md          Local deployment notes
│   └── TECHNICAL_DOCS.md      Architecture summary
├── FINAL_SUBMISSION.md        Submission fields and handoff checklist
└── README.md                  This document
```

## Start the service on Windows

Use three PowerShell windows.

### Window 1: API

```powershell
cd "C:\Users\vanac\OneDrive\Desktop\AKTAI DT"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
Remove-Item Env:FORTEBANK_BENCHMARK_GROUND_TRUTH -ErrorAction SilentlyContinue
py .\backend\api.py
```

Keep this window open. The expected startup state is:

```text
Detector ready | benchmark_mode=False | ml_classifier_loaded=True
Uvicorn running on http://0.0.0.0:8000
```

### Window 2: readiness check

```powershell
Invoke-RestMethod http://localhost:8000/health | Format-List
```

Expected:

```text
status               : healthy
ml_classifier_loaded : True
benchmark_mode       : False
```

### Window 3: static frontend server

```powershell
cd "C:\Users\vanac\OneDrive\Desktop\AKTAI DT"
py -m http.server 8080
```

Open:

```text
http://localhost:8080/frontend/index.html
```

Serving the frontend through `http.server` is preferred to opening the HTML directly with `file:///`, because it avoids browser restrictions around local requests.

## Jury walkthrough through the Web interface

Open the interface, choose a `.mov` file, and click the analysis button. The page displays a processing state while the API works asynchronously. When complete, it shows the verdict and one card per returned possible violation.

Each card can contain the violation title, the type, timestamp, confidence, a detailed explanation, what to check in the video, a normative reference, a limitation note, and one or more evidence frames.

For a local smoke test, use an annotated video such as:

```text
C:\Users\vanac\OneDrive\Desktop\datathon problem\open\videos\Х-30.mov
```

`Х-30` is useful because it contains multiple labeled types. Then use a negative example such as `Х-21.mov` and verify that the verdict is `no_violation`.

## API walkthrough

Upload a video:

```powershell
$upload = curl.exe -s -X POST `
  -H "X-API-Key: fortebank-aktai-2026" `
  -F "file=@C:\Users\vanac\OneDrive\Desktop\datathon problem\open\videos\Х-30.mov" `
  http://localhost:8000/api/v1/analyze | ConvertFrom-Json

$jobId = $upload.job_id
$upload | Format-List
```

Poll the job:

```powershell
do {
    Start-Sleep -Seconds 10
    $result = Invoke-RestMethod `
      -Uri "http://localhost:8000/api/v1/jobs/$jobId" `
      -Headers @{"X-API-Key"="fortebank-aktai-2026"}
    Write-Host "Status: $($result.status)"
} while ($result.status -eq "processing")
```

Inspect the result without printing Base64 images:

```powershell
$result | Select-Object job_id, status, verdict, ml_violation_probability, inference_mode | Format-List

$result.violations | Select-Object `
  type,
  title,
  timecode_sec,
  confidence,
  rule_reference,
  @{Name="frame_time"; Expression={$_.evidence_frames[0].timestamp_sec}},
  @{Name="frame_count"; Expression={$_.evidence_frames.Count}} | Format-Table -AutoSize
```

For each returned card, `timecode_sec` and the first evidence-frame timestamp should match. When several violations exist, evidence is kept per type rather than shared across cards.

## What the result means

`violation` means that the service found one or more possible visual indicators. `no_violation` means that no configured visual detector or model signal passed the current decision path. Neither value replaces human review.

Temporal detector evidence is preferred when available. A type-only ML prediction may receive an orienting candidate frame and is explicitly marked as requiring verification. A screenshot is not proof of an invisible fact such as a transaction record, authorization, or actual lock state.

## Model and training artifacts

The runtime loads the trained artifacts under `ml_pipeline/models/`. The current package includes the binary model, multi-label model, ensemble, scaler, class mapping, and metadata. The ensemble was calibrated using out-of-fold predictions and trained only on the labeled feature set.

To retrain, do not use hidden jury videos. Use only the project training features and annotations:

```powershell
cd "C:\Users\vanac\OneDrive\Desktop\AKTAI DT\ml_pipeline"
python train_classifier.py `
  --config config.yaml `
  --features "features/extracted_features.csv" `
  --annotations "C:\path\to\detailed_annotations.csv" `
  --mode both

python train_ensemble.py `
  --config config.yaml `
  --features "features/extracted_features.csv" `
  --annotations "C:\path\to\detailed_annotations.csv"
```

## Final verification checklist

Before the demonstration, confirm that the API is running, `/health` reports `ml_classifier_loaded=True`, benchmark mode is `False`, the frontend can upload a `.mov`, a job reaches `done`, and the response contains verdict, type, timestamp, description, and evidence frames. Keep the benchmark environment variable unset.

The local Git repository is clean after the latest commit. Runtime uploads, result JSON files, evidence caches, Python caches, and local environments are intentionally excluded from the repository.

## Known limitations

The labeled dataset is small and has rare violation classes. Performance on hidden footage cannot be guaranteed from the known-video benchmark. Some visual types require human verification because the camera may not show the relevant detail. The normative references in the UI were taken from the supplied project materials and should be checked against the current official edition before being presented as a legal conclusion.
