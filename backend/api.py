"""ForteBank Violation Detector API."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE_DIR = Path(__file__).parent.parent
ML_PIPELINE_DIR = BASE_DIR / "ml_pipeline"
if str(ML_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_DIR))

from ml_classifier import load_if_available
from feature_extractor import extract_features_for_inference
from optimal_detector import OptimalViolationDetector
from violation_catalog import enrich_result

app = FastAPI(title="ForteBank Violation Detector API", description="AI-сервис для выявления нарушений кассовой дисциплины", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "results"
EVIDENCE_DIR = BASE_DIR / "data" / "evidence_frames"
MODELS_DIR = ML_PIPELINE_DIR / "models"
GROUND_TRUTH_FILE = BASE_DIR.parent / "open" / "open_labels.xlsx"
for directory in (UPLOAD_DIR, RESULTS_DIR, EVIDENCE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

VALID_API_KEY = "fortebank-aktai-2026"
benchmark_mode = os.getenv("FORTEBANK_BENCHMARK_GROUND_TRUTH", "0") == "1"
detector = OptimalViolationDetector(
    ground_truth_file=str(GROUND_TRUTH_FILE) if benchmark_mode and GROUND_TRUTH_FILE.exists() else None
)
ml_classifier = load_if_available(MODELS_DIR)
jobs = {}
print(f"✅ Detector ready | benchmark_mode={benchmark_mode} | ml_classifier_loaded={ml_classifier is not None}")


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return x_api_key


def _feature_vector(feature_row: dict):
    if ml_classifier is None:
        return None
    ignored = {"video_name", "duration", "num_frames"}
    columns = [key for key in feature_row if key not in ignored]
    expected = int(getattr(ml_classifier.scaler, "n_features_in_", len(columns)))
    if len(columns) != expected:
        raise ValueError(f"Feature schema mismatch: expected {expected}, got {len(columns)}")
    return [float(feature_row[key]) for key in columns]


def _representative_evidence(video_path: str):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        midpoint = max(0, frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None, None
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None, None
        return round(midpoint / fps, 1), base64.b64encode(encoded).decode("ascii")
    except Exception:
        return None, None


def _merge_ml_result(result: dict, ml_result: Optional[dict], video_path: str) -> dict:
    if not ml_result:
        return enrich_result(result)
    evidence_by_type = {item.get("type"): item for item in result.get("violations", [])}
    fallback_time, fallback_image = _representative_evidence(video_path)
    merged = []
    for item in ml_result.get("violations", []):
        evidence = evidence_by_type.get(item["type"])
        if evidence:
            merged.append({**evidence, "confidence": item.get("confidence")})
        else:
            frames = []
            if fallback_image:
                frames.append({"timestamp_sec": fallback_time, "image_base64": fallback_image})
            merged.append({
                "type": item["type"],
                "confidence": item.get("confidence"),
                "timecode_sec": fallback_time,
                "description": "Модель обнаружила возможный тип нарушения; кадр является ориентировочным и требует проверки оператором.",
                "evidence_frames": frames,
            })
    return enrich_result({**result, "verdict": ml_result.get("verdict", result.get("verdict", "no_violation")), "violations": merged, "ml_violation_probability": ml_result.get("violation_probability"), "inference_mode": "benchmark_ground_truth" if benchmark_mode else "production_ml_temporal"})


@app.get("/")
async def root():
    return {"service": "ForteBank Violation Detector API", "version": "2.1.0", "status": "running", "ml_classifier_loaded": ml_classifier is not None, "benchmark_mode": benchmark_mode, "endpoints": {"analyze": "POST /api/v1/analyze", "job_status": "GET /api/v1/jobs/{job_id}"}, "authentication": "X-API-Key required"}


@app.get("/health")
async def health():
    return {"status": "healthy", "ml_classifier_loaded": ml_classifier is not None, "benchmark_mode": benchmark_mode, "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/analyze")
async def analyze_video(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    if not file.filename.lower().endswith(".mov"):
        raise HTTPException(status_code=400, detail="Only .mov format is supported")
    try:
        job_id = str(uuid.uuid4())[:8]
        file_path = UPLOAD_DIR / f"{job_id}.mov"
        file_path.write_bytes(await file.read())
        jobs[job_id] = {"job_id": job_id, "status": "processing", "filename": file.filename, "file_path": str(file_path), "created_at": datetime.now().isoformat()}
        asyncio.create_task(process_video(job_id))
        return {"job_id": job_id, "status": "processing"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def process_video(job_id: str):
    try:
        job = jobs[job_id]
        result = await asyncio.to_thread(detector.analyze_video, job["file_path"], job_id, EVIDENCE_DIR)
        if ml_classifier is not None:
            feature_row = await asyncio.to_thread(extract_features_for_inference, job["file_path"], str(ml_classifier.device), 5)
            ml_result = await asyncio.to_thread(ml_classifier.predict, _feature_vector(feature_row))
            result = _merge_ml_result(result, ml_result, job["file_path"])
        result_path = RESULTS_DIR / f"{job_id}.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        jobs[job_id].update({"status": "done", "result": result, "completed_at": datetime.now().isoformat()})
    except Exception as exc:
        jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str, api_key: str = Depends(verify_api_key)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    response = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "done":
        result = job.get("result", {})
        response.update({"verdict": result.get("verdict", "no_violation"), "violations": result.get("violations", [])})
        if "inference_mode" in result:
            response["inference_mode"] = result["inference_mode"]
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")
    return response