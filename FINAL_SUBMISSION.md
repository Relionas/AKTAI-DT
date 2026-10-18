# Final submission handoff

## Submission fields

**Web interface URL**

`file:///C:/Users/vanac/OneDrive/Desktop/FINAL_PROJECT/frontend/index.html`

**API base URL**

`http://localhost:8000/api/v1`

**API key for the jury**

`fortebank-aktai-2026`

**Git Link**

`Not available — local submission package`

## Important local-access note

The URLs above are local. They work only on the computer where the project is running. If the organizers require remote access, deploy the same package to an approved reachable host and replace both URL fields before submission. Do not expose the API key in a public repository.

## Start procedure

From the project root, open PowerShell and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Remove-Item Env:FORTEBANK_BENCHMARK_GROUND_TRUTH -ErrorAction SilentlyContinue
python backend\api.py
```

In another PowerShell window, confirm:

```powershell
Invoke-RestMethod http://localhost:8000/health | Format-List
```

Expected production state:

```text
status               : healthy
ml_classifier_loaded : True
benchmark_mode       : False
```

## Final checks before handover

Confirm that the API is running in production mode, the model artifacts load, the Web interface can upload a `.mov` file, the job reaches `done`, and the response contains `verdict`, `violations`, `timecode_sec`, `description`, and `evidence_frames`. Keep `FORTEBANK_BENCHMARK_GROUND_TRUTH` unset for jury evaluation.
