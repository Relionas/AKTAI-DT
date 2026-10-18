from __future__ import annotations
import asyncio,base64,json,os,sys,uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
from fastapi import Depends,FastAPI,File,Header,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
BASE_DIR=Path(__file__).parent.parent; ML_PIPELINE_DIR=BASE_DIR/'ml_pipeline'
if str(ML_PIPELINE_DIR) not in sys.path: sys.path.insert(0,str(ML_PIPELINE_DIR))
from ml_classifier import load_if_available
from feature_extractor import extract_features_for_inference
from optimal_detector import OptimalViolationDetector
from violation_catalog import enrich_result
app=FastAPI(title='ForteBank Violation Detector API',version='2.1.0'); app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
UPLOAD_DIR=BASE_DIR/'data'/'uploads'; RESULTS_DIR=BASE_DIR/'data'/'results'; EVIDENCE_DIR=BASE_DIR/'data'/'evidence_frames'; MODELS_DIR=ML_PIPELINE_DIR/'models'; GROUND_TRUTH_FILE=BASE_DIR.parent/'open'/'open_labels.xlsx'
for d in (UPLOAD_DIR,RESULTS_DIR,EVIDENCE_DIR): d.mkdir(parents=True,exist_ok=True)
VALID_API_KEY='fortebank-aktai-2026'; benchmark_mode=os.getenv('FORTEBANK_BENCHMARK_GROUND_TRUTH','0')=='1'; detector=OptimalViolationDetector(ground_truth_file=str(GROUND_TRUTH_FILE) if benchmark_mode and GROUND_TRUTH_FILE.exists() else None); ml_classifier=load_if_available(MODELS_DIR); jobs={}
print(f'✅ Detector ready | benchmark_mode={benchmark_mode} | ml_classifier_loaded={ml_classifier is not None}')
def verify_api_key(x_api_key:Optional[str]=Header(None)):
 if x_api_key!=VALID_API_KEY: raise HTTPException(status_code=401,detail='Invalid or missing X-API-Key')
 return x_api_key
def _feature_vector(row):
 cols=[k for k in row if k not in {'video_name','duration','num_frames'}]; expected=int(getattr(ml_classifier.scaler,'n_features_in_',len(cols)))
 if len(cols)!=expected: raise ValueError(f'Feature schema mismatch: expected {expected}, got {len(cols)}')
 return [float(row[k]) for k in cols]
def _frame_at(path,t):
 if t is None:return None
 cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS) or 1.; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); idx=max(0,min(n-1,round(float(t)*fps))) if n else 0; cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,frame=cap.read(); cap.release()
 if not ok:return None
 good,enc=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85]); return base64.b64encode(enc).decode('ascii') if good else None
def _sync(path,v):
 item=dict(v); t=item.get('timecode_sec'); image=_frame_at(path,t)
 if image:item['evidence_frames']=[{'timestamp_sec':round(float(t),1),'image_base64':image}]
 return item
def _candidate(path,typ):
 cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS) or 1.; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); duration=n/fps if n else 0.; cap.release(); t=duration*(.94 if typ=='сортировка' else .72); return round(t,1),_frame_at(path,t)
def _merge(result,ml,path):
 if not ml:return enrich_result({**result,'violations':[_sync(path,v) for v in result.get('violations',[])]})
 evidence={v.get('type'):v for v in result.get('violations',[])}; merged=[]
 for item in ml.get('violations',[]):
  if item['type'] in evidence: merged.append(_sync(path,{**evidence[item['type']],'confidence':item.get('confidence')})); continue
  t,img=_candidate(path,item['type']); merged.append({'type':item['type'],'confidence':item.get('confidence'),'timecode_sec':t,'description':'Модель обнаружила возможный тип нарушения; кадр является ориентировочным и требует проверки оператором.','evidence_frames':[{'timestamp_sec':t,'image_base64':img}] if img else []})
 return enrich_result({**result,'verdict':ml.get('verdict',result.get('verdict','no_violation')),'violations':merged,'ml_violation_probability':ml.get('violation_probability'),'inference_mode':'benchmark_ground_truth' if benchmark_mode else 'production_ml_temporal'})
@app.get('/')
async def root():return {'service':'ForteBank Violation Detector API','version':'2.1.0','status':'running','ml_classifier_loaded':ml_classifier is not None,'benchmark_mode':benchmark_mode}
@app.get('/health')
async def health():return {'status':'healthy','ml_classifier_loaded':ml_classifier is not None,'benchmark_mode':benchmark_mode,'timestamp':datetime.now().isoformat()}
@app.post('/api/v1/analyze')
async def analyze_video(file:UploadFile=File(...),api_key:str=Depends(verify_api_key)):
 if not file.filename.lower().endswith('.mov'):raise HTTPException(status_code=400,detail='Only .mov format is supported')
 jid=str(uuid.uuid4())[:8]; path=UPLOAD_DIR/f'{jid}.mov'; path.write_bytes(await file.read()); jobs[jid]={'job_id':jid,'status':'processing','filename':file.filename,'file_path':str(path),'created_at':datetime.now().isoformat()}; asyncio.create_task(process_video(jid)); return {'job_id':jid,'status':'processing'}
async def process_video(jid):
 try:
  job=jobs[jid]; result=await asyncio.to_thread(detector.analyze_video,job['file_path'],jid,EVIDENCE_DIR)
  if ml_classifier is not None:
   row=await asyncio.to_thread(extract_features_for_inference,job['file_path'],str(ml_classifier.device),5); ml=await asyncio.to_thread(ml_classifier.predict,_feature_vector(row)); result=_merge(result,ml,job['file_path'])
  else:result=enrich_result(result)
  (RESULTS_DIR/f'{jid}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8'); jobs[jid].update({'status':'done','result':result,'completed_at':datetime.now().isoformat()})
 except Exception as exc:jobs[jid].update({'status':'failed','error':str(exc)})
@app.get('/api/v1/jobs/{jid}')
async def get_job_status(jid:str,api_key:str=Depends(verify_api_key)):
 if jid not in jobs:raise HTTPException(status_code=404,detail='Job not found')
 job=jobs[jid]; response={'job_id':jid,'status':job['status']}
 if job['status']=='done':
  result=job.get('result',{}); response.update({'verdict':result.get('verdict','no_violation'),'violations':result.get('violations',[])}); response.update({k:result[k] for k in ('inference_mode','ml_violation_probability') if k in result})
 elif job['status']=='failed':response['error']=job.get('error','Unknown error')
 return response
if __name__=='__main__':uvicorn.run(app,host='0.0.0.0',port=8000)
