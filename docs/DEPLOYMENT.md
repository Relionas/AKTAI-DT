# Deployment guide

## Scope

This package is prepared for local Windows demonstration. It runs the FastAPI service locally and keeps video processing on the host machine. A remote jury URL requires a separately approved deployment and is not created by this local package.

## Windows manual deployment

From the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Remove-Item Env:FORTEBANK_BENCHMARK_GROUND_TRUTH -ErrorAction SilentlyContinue
python backend\api.py
```

Verify readiness:

```powershell
Invoke-RestMethod http://localhost:8000/health | Format-List
```

The jury path must show `benchmark_mode=False`. Do not set `FORTEBANK_BENCHMARK_GROUND_TRUTH` when processing hidden jury videos.

## Docker note

The repository includes a Dockerfile for environments with compatible CPU/GPU runtime support. Before using it for a remote deployment, verify that model weights are included, that the container has access to the required device runtime, and that the service completes a full `.mov` smoke test within the organizer's time limit.

## Operational checklist

Confirm that model artifacts exist under `ml_pipeline/models/`, that `ml_classifier_loaded=True`, that upload returns a `job_id`, that polling reaches `done`, and that every returned violation has a type, timestamp, description, and evidence frame. Do not commit API keys or uploaded videos to a public repository.

## Troubleshooting

If the API returns `404` for a job, use the exact `job_id` from the immediately preceding upload response. If the model is unavailable, reinstall `backend/requirements.txt` and verify all model artifacts. If a browser blocks a `file:///` frontend request, serve the project root with `python -m http.server 8080` and open `http://localhost:8080/frontend/index.html`.
