# Final jury package

## Submission fields

- **Web interface URL:** `http://localhost:8080/frontend/index.html`
- **API base URL:** `http://localhost:8000/api/v1`
- **API key:** `fortebank-aktai-2026`
- **Git Link:** `Not available — local submission package`

The URLs are local to the demonstration computer. A remote jury cannot access `localhost` without an approved network deployment.

## Start

```powershell
cd "C:\Users\vanac\OneDrive\Desktop\AKTAI DT"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
Remove-Item Env:FORTEBANK_BENCHMARK_GROUND_TRUTH -ErrorAction SilentlyContinue
py .\backend\api.py
```

In a second window:

```powershell
Invoke-RestMethod http://localhost:8000/health | Format-List
```

Expected: `healthy`, `ml_classifier_loaded=True`, and `benchmark_mode=False`.

In a third window:

```powershell
cd "C:\Users\vanac\OneDrive\Desktop\AKTAI DT"
py -m http.server 8080
```

Open `http://localhost:8080/frontend/index.html`.

## Recommended smoke tests

Use `Х-30.mov` to demonstrate multiple violation cards and `Х-21.mov` to demonstrate a negative result. The interface renders detailed descriptions, confidence, timecodes, normative references, limitations, and evidence frames.

The API is asynchronous. Upload returns a `job_id`; poll the job endpoint until `status=done`. When several violations are returned, each card's evidence frame is regenerated from its own timestamp.

## Important constraints

Do not set `FORTEBANK_BENCHMARK_GROUND_TRUTH` for jury evaluation. Do not add hidden jury videos to training. Do not expose the API key in a public repository. The service is a visual review aid and not a final legal conclusion.

See `README.md` for full structure, commands, API examples, model notes, and troubleshooting.
