"""ForteBank Violation Detector — HTTP API и веб-интерфейс.

Один процесс отдаёт и интерфейс (GET /), и API для жюри (/api/v1/...).
Все модели выполняются локально; облачные AI-сервисы не используются.
"""
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
import uvicorn
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

BASE_DIR = Path(__file__).resolve().parent.parent
ML_PIPELINE_DIR = BASE_DIR / 'ml_pipeline'
FRONTEND_FILE = BASE_DIR / 'frontend' / 'index.html'
for extra in (ML_PIPELINE_DIR, BASE_DIR / 'backend'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

try:  # .env в корне проекта (в Git не попадает)
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / '.env')
except ImportError:
    pass

VALID_API_KEY = os.getenv('API_KEY', '').strip()
if not VALID_API_KEY:
    sys.exit('❌ Не задан API_KEY. Создайте файл .env в корне проекта со строкой API_KEY=<ваш ключ> '
             'или задайте переменную окружения API_KEY.')

from ml_classifier import load_if_available  # noqa: E402
from feature_extractor import extract_features_for_inference  # noqa: E402
from optimal_detector import OptimalViolationDetector  # noqa: E402
from violation_catalog import enrich_result  # noqa: E402

PORT = int(os.getenv('PORT', '8000'))
# YOLO (ultralytics) не потокобезопасен, а GPU/CPU общие: по умолчанию видео обрабатываются по очереди.
MAX_PARALLEL_JOBS = max(1, int(os.getenv('MAX_PARALLEL_JOBS', '1')))
UPLOAD_DIR = BASE_DIR / 'data' / 'uploads'
RESULTS_DIR = BASE_DIR / 'data' / 'results'
EVIDENCE_DIR = BASE_DIR / 'data' / 'evidence_frames'
MODELS_DIR = ML_PIPELINE_DIR / 'models'
for d in (UPLOAD_DIR, RESULTS_DIR, EVIDENCE_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='ForteBank Violation Detector API', version='2.2.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

detector = OptimalViolationDetector()
ml_classifier = load_if_available(MODELS_DIR)
jobs = {}
_job_slots = None  # asyncio.Semaphore, создаётся внутри работающего event loop


def _slots():
    global _job_slots
    if _job_slots is None:
        _job_slots = asyncio.Semaphore(MAX_PARALLEL_JOBS)
    return _job_slots


print(f'✅ Detector ready | ml_classifier_loaded={ml_classifier is not None}')


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail='Invalid or missing X-API-Key')
    return x_api_key


def _feature_vector(row):
    cols = [k for k in row if k not in {'video_name', 'duration', 'num_frames'}]
    expected = int(getattr(ml_classifier.scaler, 'n_features_in_', len(cols)))
    if len(cols) != expected:
        raise ValueError(f'Feature schema mismatch: expected {expected}, got {len(cols)}')
    return [float(row[k]) for k in cols]


def _frame_at(path, t):
    """JPEG base64 кадра на времени t. Если кадр не читается — ближайший читаемый до него."""
    if t is None:
        return None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = max(0, min(n - 1, round(float(t) * fps))) if n else 0
    frame = None
    for candidate in (idx, idx - 1, idx - 5, idx - 15, 0):
        if candidate < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, frame = cap.read()
        if ok:
            break
        frame = None
    cap.release()
    if frame is None:
        return None
    good, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(enc).decode('ascii') if good else None


def _sync(path, v):
    item = dict(v)
    typ = str(item.get('type', ''))
    t = item.get('timecode_sec')
    image = _frame_at(path, t)
    item['evidence_type'] = typ
    item['evidence_source'] = 'temporal_detector'
    item['evidence_frames'] = [{'timestamp_sec': round(float(t), 1), 'image_base64': image}] if image else []
    return item


def _candidate(path, typ):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps if n else 0.0
    cap.release()
    t = duration * (0.94 if typ == 'сортировка' else 0.72)
    return round(t, 1), _frame_at(path, t)


def _merge(result, ml, path):
    if not ml:
        return enrich_result(result)
    evidence = {v.get('type'): v for v in result.get('violations', [])}
    merged = []
    for item in ml.get('violations', []):
        typ = item['type']
        if typ in evidence:
            merged.append(_sync(path, {**evidence[typ], 'confidence': item.get('confidence')}))
            continue
        t, img = _candidate(path, typ)
        merged.append({
            'type': typ, 'evidence_type': typ, 'evidence_source': 'type_candidate_localizer',
            'confidence': item.get('confidence'), 'timecode_sec': t,
            'evidence_frames': [{'timestamp_sec': t, 'image_base64': img}] if img else [],
        })
    return enrich_result({
        **result,
        'verdict': ml.get('verdict', result.get('verdict', 'no_violation')),
        'violations': merged,
        'ml_violation_probability': ml.get('violation_probability'),
        'inference_mode': 'production_ml_temporal',
    })


def _render_frontend() -> str:
    # Ключ подставляется при отдаче страницы, поэтому его нет в репозитории.
    return FRONTEND_FILE.read_text(encoding='utf8').replace('__API_KEY__', VALID_API_KEY)


@app.get('/', response_class=HTMLResponse)
async def frontend():
    return _render_frontend()


@app.get('/frontend/index.html', response_class=HTMLResponse)
async def frontend_legacy_path():
    return _render_frontend()


@app.get('/health')
async def health():
    return {'status': 'healthy', 'ml_classifier_loaded': ml_classifier is not None,
            'yolo_loaded': detector.model_loaded, 'timestamp': datetime.now().isoformat()}


@app.post('/api/v1/analyze')
async def analyze_video(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    if not file.filename or not file.filename.lower().endswith('.mov'):
        raise HTTPException(status_code=400, detail='Only .mov format is supported')
    jid = uuid.uuid4().hex[:8]
    path = UPLOAD_DIR / f'{jid}.mov'
    path.write_bytes(await file.read())
    jobs[jid] = {'job_id': jid, 'status': 'processing', 'filename': file.filename,
                 'file_path': str(path), 'created_at': datetime.now().isoformat()}
    asyncio.create_task(process_video(jid))
    return {'job_id': jid, 'status': 'processing'}


async def process_video(jid):
    async with _slots():
        await _process_video(jid)


async def _process_video(jid):
    job = jobs[jid]
    path = job['file_path']
    try:
        result = await asyncio.to_thread(detector.analyze_video, path, jid, EVIDENCE_DIR)
        if ml_classifier is not None:
            try:
                row = await asyncio.to_thread(extract_features_for_inference, path, str(ml_classifier.device), 5)
                ml = await asyncio.to_thread(ml_classifier.predict, _feature_vector(row))
                result = _merge(result, ml, path)
            except Exception as exc:  # ML упал — отдаём результат детектора, а не failed
                print(f'⚠️ ML step failed for {jid}: {exc}')
                result = enrich_result(result)
        else:
            result = enrich_result(result)
        record = {'job_id': jid, 'status': 'done', 'result': result, 'completed_at': datetime.now().isoformat()}
        (RESULTS_DIR / f'{jid}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
        job.update(record)
    except Exception as exc:
        job.update({'status': 'failed', 'error': str(exc)})


def _load_job(jid):
    """Задача из памяти, а после перезапуска сервера — из data/results/<job_id>.json."""
    if jid in jobs:
        return jobs[jid]
    saved = RESULTS_DIR / f'{jid}.json'
    if saved.exists() and saved.parent == RESULTS_DIR:
        try:
            record = json.loads(saved.read_text(encoding='utf8'))
            if 'result' not in record:  # старый формат: в файле лежит сам результат
                record = {'job_id': jid, 'status': 'done', 'result': record}
            jobs[jid] = record
            return record
        except (OSError, ValueError):
            return None
    return None


@app.get('/api/v1/jobs/{jid}')
async def get_job_status(jid: str, api_key: str = Depends(verify_api_key)):
    if not jid.isalnum():
        raise HTTPException(status_code=404, detail='Job not found')
    job = _load_job(jid)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    response = {'job_id': jid, 'status': job['status']}
    if job['status'] == 'done':
        result = job.get('result', {})
        response.update({'verdict': result.get('verdict', 'no_violation'), 'violations': result.get('violations', [])})
        response.update({k: result[k] for k in ('inference_mode', 'ml_violation_probability') if k in result})
    elif job['status'] == 'failed':
        response['error'] = job.get('error', 'Unknown error')
    return response


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT)
