# API guide

## Authentication

All protected endpoints require the `X-API-Key` header:

```text
X-API-Key: fortebank-aktai-2026
```

## Health

```powershell
Invoke-RestMethod http://localhost:8000/health | Format-List
```

A ready production process reports `healthy`, `ml_classifier_loaded=True`, and `benchmark_mode=False`.

## Upload a video

`POST /api/v1/analyze` accepts a `.mov` file in the multipart field `file` and returns an asynchronous job identifier.

```powershell
$upload = curl.exe -s -X POST `
  -H "X-API-Key: fortebank-aktai-2026" `
  -F "file=@C:\path\to\video.mov" `
  http://localhost:8000/api/v1/analyze | ConvertFrom-Json

$jobId = $upload.job_id
```

Response:

```json
{"job_id":"3f2a9c1e","status":"processing"}
```

## Get a result

`GET /api/v1/jobs/{job_id}` returns `processing`, `done`, or `failed`.

```powershell
$result = Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/jobs/$jobId" `
  -Headers @{"X-API-Key"="fortebank-aktai-2026"}
```

Completed response shape:

```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "посторонний",
      "confidence": 0.91,
      "timecode_sec": 15.9,
      "description": "Описание наблюдаемого возможного нарушения.",
      "evidence_frames": [
        {"timestamp_sec": 15.9, "image_base64": "/9j/4AAQ..."}
      ]
    }
  ],
  "ml_violation_probability": 0.98,
  "inference_mode": "production_ml_temporal"
}
```

The service is asynchronous because video analysis may take several minutes. The result is a review aid and does not constitute a final legal or disciplinary conclusion.
