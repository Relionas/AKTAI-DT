# Jury handoff guide

## Purpose

This package provides a local-only review service for visual indicators of cashier-operation violations. It accepts `.mov` videos, processes them locally, and returns a verdict, possible violation types, timestamps, explanations, and evidence frames.

## Local submission values

- Web interface URL: `file:///C:/Users/vanac/OneDrive/Desktop/FINAL_PROJECT/frontend/index.html`
- API base URL: `http://localhost:8000/api/v1`
- API key: `fortebank-aktai-2026`
- Git Link: `Not available — local submission package`

## Start

```powershell
cd C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Remove-Item Env:FORTEBANK_BENCHMARK_GROUND_TRUTH -ErrorAction SilentlyContinue
python backend\api.py
```

Check readiness:

```powershell
Invoke-RestMethod http://localhost:8000/health | Format-List
```

The production server should report `healthy`, `ml_classifier_loaded=True`, and `benchmark_mode=False`.

## API flow

Upload a video to `POST /api/v1/analyze` using the `file` multipart field and the `X-API-Key` header. The service returns a `job_id`. Poll `GET /api/v1/jobs/{job_id}` until the status becomes `done`.

Example:

```powershell
$upload = curl.exe -s -X POST `
  -H "X-API-Key: fortebank-aktai-2026" `
  -F "file=@C:\path\to\video.mov" `
  http://localhost:8000/api/v1/analyze | ConvertFrom-Json

$jobId = $upload.job_id

do {
    Start-Sleep -Seconds 10
    $result = Invoke-RestMethod `
      -Uri "http://localhost:8000/api/v1/jobs/$jobId" `
      -Headers @{"X-API-Key"="fortebank-aktai-2026"}
} while ($result.status -eq "processing")

$result | Select-Object job_id, status, verdict, ml_violation_probability, inference_mode | Format-List
```

## Architecture

The runtime combines temporal YOLOv8 object analysis and trained feature models. Temporal detections provide time-localized evidence when available. The binary model estimates whether a violation is present; the multi-label model estimates possible violation types. ML-only types receive an orienting representative frame and are explicitly described as requiring operator verification.

## Evaluation note

The labeled set contains 42 videos and rare classes. Benchmark results on known videos and out-of-fold estimates are reported separately. The production path does not use ground-truth lookup by filename. The service is an assistive review system and does not make a final legal or disciplinary determination.

## Troubleshooting

If `ml_classifier_loaded` is `False`, stop the server and verify that `ml_pipeline/models/` contains the checkpoints, scaler, binarizer, and metadata. If an API job returns `404`, check that the exact `job_id` returned by the upload request is used. Avoid overwriting the job ID variable with a second failed upload command.
