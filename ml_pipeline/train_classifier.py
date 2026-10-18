"""Train binary and multi-label violation models with honest and benchmark modes."""
from __future__ import annotations
import argparse, json, pickle, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch.utils.data import Dataset, DataLoader

class FeatureDataset(Dataset):
    def __init__(self, x, binary, multi, names):
        self.x=torch.tensor(x,dtype=torch.float32); self.binary=torch.tensor(binary,dtype=torch.float32)
        self.multi=torch.tensor(multi,dtype=torch.float32); self.names=np.asarray(names)
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i],self.binary[i],self.multi[i],self.names[i]

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden=(256,128)):
        super().__init__(); layers=[]; previous=input_dim
        for size in hidden:
            layers += [nn.Linear(previous,size),nn.LayerNorm(size),nn.ReLU(),nn.Dropout(.25)]; previous=size
        layers.append(nn.Linear(previous,output_dim)); self.network=nn.Sequential(*layers)
    def forward(self,x): return self.network(x)

def normalize_video_name(value):
    return re.sub(r"[^a-z0-9]+","",Path(str(value)).stem.strip().lower().replace("х","x"))

def as_bool(value): return str(value).strip().lower() in {"1","true","yes","да","нарушение"}

def load_annotations(path):
    frame=pd.read_csv(path) if Path(path).suffix.lower()=='.csv' else pd.read_excel(path)
    frame=frame.rename(columns={'video_name':'наименование файла','has_violation':'нарушение / без нарушения','violation_type':'тип нарушения'})
    result={}
    for _,row in frame.iterrows():
        key=normalize_video_name(row['наименование файла']); result.setdefault(key,[])
        value=row.get('тип нарушения','')
        if as_bool(row['нарушение / без нарушения']) and pd.notna(value) and str(value).strip() not in result[key]: result[key].append(str(value).strip())
    return result

def load_dataset(features_path,annotations_path):
    frame=pd.read_csv(features_path); mapping=load_annotations(annotations_path)
    columns=[c for c in frame.columns if c not in {'video_name','duration','num_frames'}]
    x=[]; labels=[]; names=[]; missing=[]
    for _,row in frame.iterrows():
        name=str(row['video_name']); key=normalize_video_name(name)
        if key not in mapping: missing.append(name)
        x.append(pd.to_numeric(row[columns],errors='coerce').fillna(0).to_numpy(dtype=np.float32)); labels.append(mapping.get(key,[])); names.append(name)
    if missing: print(f'   ⚠️ Unmatched feature videos ({len(missing)}): {missing}')
    x=np.asarray(x,dtype=np.float32); print('\n📊 Dataset statistics:'); print(f'   Total samples: {len(x)}\n   Feature dimensions: {x.shape[1]}')
    print(f'   No violation: {sum(not z for z in labels)}\n   Has violation: {sum(bool(z) for z in labels)}'); print('\n   Multi-label class distribution:')
    counts={}
    for row in labels:
        for label in row: counts[label]=counts.get(label,0)+1
    for label in sorted(counts): print(f'      {label}: {counts[label]}')
    return x,labels,np.asarray(names)

def split_indices(labels,validation_split):
    n=len(labels); target=max(2,min(n-1,int(round(n*validation_split))))
    positive=[i for i,z in enumerate(labels) if z]; negative=[i for i,z in enumerate(labels) if not z]
    selected=([positive[0],negative[0]] if positive and negative else [])
    for i,z in enumerate(labels):
        if len(z)>1 and i not in selected and len(selected)<target: selected.append(i)
    for i in range(n):
        if len(selected)>=target: break
        if i not in selected: selected.append(i)
    val=np.asarray(sorted(set(selected)),dtype=int); train=np.asarray([i for i in range(n) if i not in set(val)],dtype=int)
    return train,val

def loader(x,binary,multi,names,indices,batch,shuffle):
    return DataLoader(FeatureDataset(x[indices],binary[indices],multi[indices],names[indices]),batch_size=max(1,min(batch,len(indices))),shuffle=shuffle,drop_last=False)

def probabilities(binary_model,multi_model,data,device):
    binary_model.eval(); multi_model.eval(); bt=[]; bp=[]; mt=[]; mp=[]
    with torch.no_grad():
        for x,binary,multi,_ in data:
            x=x.to(device); bt.extend(binary.numpy().astype(int)); bp.extend(torch.sigmoid(binary_model(x)).squeeze(1).cpu().numpy()); mt.append(multi.numpy().astype(int)); mp.append(torch.sigmoid(multi_model(x)).cpu().numpy())
    return np.asarray(bt),np.asarray(bp),np.vstack(mt),np.vstack(mp)

def threshold(y,p,default=.5):
    if int(y.sum())<2 or int((y==0).sum())<2: return default
    candidates=np.linspace(.35,.65,13); scores=[f1_score(y,p>=t,zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])

def calibrate(bt,bp,mt,mp): return threshold(bt,bp),np.asarray([threshold(mt[:,i],mp[:,i]) for i in range(mt.shape[1])],dtype=np.float32)

def score(bt,bp,mt,mp,bth,mth):
    bpred=(bp>=bth).astype(int); mpred=(mp>=mth.reshape(1,-1)).astype(int)
    return f1_score(bt,bpred,zero_division=0),f1_score(mt,mpred,average='macro',zero_division=0),bpred,mpred

def cross_validate(x,labels,config):
    classes=list(dict.fromkeys([str(z) for z in config.get('violation_types',[])] + sorted({z for row in labels for z in row})))
    mlb=MultiLabelBinarizer(classes=classes); y=mlb.fit_transform(labels).astype(int); binary=(y.sum(1)>0).astype(int)
    folds=min(5,int(binary.sum()),int((binary==0).sum()))
    if folds<2: print('⚠️ Cross-validation skipped: insufficient binary class coverage'); return None
    supported=np.where(y.sum(0)>=2)[0]; all_bt=[]; all_bp=[]; all_mt=[]; all_mp=[]
    for fold,(train,test) in enumerate(StratifiedKFold(folds,shuffle=True,random_state=42).split(x,binary),1):
        scaler=StandardScaler().fit(x[train]); xt=scaler.transform(x[train]); xv=scaler.transform(x[test])
        bmodel=LogisticRegression(class_weight='balanced',C=.2,max_iter=2000).fit(xt,binary[train])
        all_bt.extend(binary[test]); all_bp.extend(bmodel.predict_proba(xv)[:,1])
        if len(supported):
            mmodel=OneVsRestClassifier(LogisticRegression(class_weight='balanced',C=.2,max_iter=2000)).fit(xt,y[train][:,supported])
            probs=mmodel.predict_proba(xv); truth=y[test][:,supported]
            full=np.full((len(test),len(classes)),0.0); full[:,supported]=probs; all_mp.append(full); all_mt.append(y[test])
        print(f'   CV fold {fold}/{folds} complete')
    bt=np.asarray(all_bt); bp=np.asarray(all_bp); mt=np.vstack(all_mt); mp=np.vstack(all_mp); bth,mth=calibrate(bt,bp,mt,mp); bf,mf,_,_=score(bt,bp,mt,mp,bth,mth)
    print(f'\n📈 Honest OOF estimate: Binary F1={bf:.3f} | Multi-label macro F1={mf:.3f}')
    return {'binary_f1':float(bf),'multilabel_macro_f1':float(mf),'binary_threshold':float(bth),'multilabel_thresholds':mth.tolist()}

def train(x,labels,names,config,models_dir,device,production=False,fixed=None):
    classes=list(dict.fromkeys([str(z) for z in config.get('violation_types',[])] + sorted({z for row in labels for z in row})))
    mlb=MultiLabelBinarizer(classes=classes); multi_y=mlb.fit_transform(labels).astype(np.float32); binary_y=(multi_y.sum(1)>0).astype(np.float32)
    batch=int(config['training']['batch_size']); lr=float(config['training']['learning_rate']); wd=float(config['training']['weight_decay'])
    if production:
        train_idx=np.arange(len(x)); scaler=StandardScaler().fit(x); scaled=scaler.transform(x).astype(np.float32); val_data=None
    else:
        train_idx,val_idx=split_indices(labels,config['training']['validation_split']); scaler=StandardScaler().fit(x[train_idx]); scaled=np.empty_like(x); scaled[train_idx]=scaler.transform(x[train_idx]); scaled[val_idx]=scaler.transform(x[val_idx]); val_data=loader(scaled,binary_y,multi_y,names,val_idx,batch,False)
    train_data=loader(scaled,binary_y,multi_y,names,train_idx,batch,True); bmodel=MLP(x.shape[1],1).to(device); mmodel=MLP(x.shape[1],len(classes)).to(device)
    pos=max(1.,float(binary_y[train_idx].sum())); neg=max(1.,float(len(train_idx)-binary_y[train_idx].sum())); bl=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=device)); counts=multi_y[train_idx].sum(0); ml=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.divide(len(train_idx)-counts,np.maximum(counts,1.)).astype(np.float32),device=device))
    opt=torch.optim.Adam(list(bmodel.parameters())+list(mmodel.parameters()),lr=lr,weight_decay=wd); epochs=max(10,int(config['training']['epochs'])) if production else int(config['training']['epochs']); patience=int(config['training']['early_stopping_patience']); best=-1.; stale=0
    for epoch in range(epochs):
        bmodel.train(); mmodel.train(); train_b=train_m=0.
        for xx,bb,mm,_ in train_data:
            xx=xx.to(device); bb=bb.to(device).unsqueeze(1); mm=mm.to(device); opt.zero_grad(); lb=bl(bmodel(xx),bb); lm=ml(mmodel(xx),mm); (lb+lm).backward(); opt.step(); train_b+=lb.item(); train_m+=lm.item()
        if production: continue
        bt,bp,mt,mp=probabilities(bmodel,mmodel,val_data,device); bf,mf,_,_=score(bt,bp,mt,mp,.5,np.full(len(classes),.5,dtype=np.float32)); current=(bf+mf)/2
        print(f'Epoch {epoch+1}/{epochs}\n   Train Binary Loss: {train_b/len(train_data):.4f} | Multi Loss: {train_m/len(train_data):.4f}\n   Val Binary F1: {bf:.3f} | Multi F1 (macro): {mf:.3f}')
        if current>best:
            best=current; stale=0; torch.save(bmodel.state_dict(),models_dir/'binary_model.pth'); torch.save(mmodel.state_dict(),models_dir/'multilabel_model.pth'); torch.save(mmodel.state_dict(),models_dir/'best_model.pth')
        else: stale+=1
        if stale>=patience: print(f'⚠️ Early stopping triggered at epoch {epoch+1}'); break
    if not production:
        bmodel.load_state_dict(torch.load(models_dir/'binary_model.pth',map_location=device)); mmodel.load_state_dict(torch.load(models_dir/'multilabel_model.pth',map_location=device)); bt,bp,mt,mp=probabilities(bmodel,mmodel,val_data,device); bth,mth=calibrate(bt,bp,mt,mp)
    else:
        bth,mth=fixed or (.5,np.full(len(classes),.5,dtype=np.float32)); torch.save(bmodel.state_dict(),models_dir/'binary_model.pth'); torch.save(mmodel.state_dict(),models_dir/'multilabel_model.pth'); torch.save(mmodel.state_dict(),models_dir/'best_model.pth')
    return bmodel,mmodel,mlb,scaler,val_data,bl,ml,bth,mth

def save_artifacts(directory,mlb,scaler,input_dim,bth,mth,cv):
    for filename,value in [('multilabel_binarizer.pkl',mlb),('label_encoder.pkl',mlb),('scaler.pkl',scaler)]:
        with open(directory/filename,'wb') as handle: pickle.dump(value,handle)
    metadata={'schema_version':3,'task':'binary_violation_plus_multilabel_types','input_dim':input_dim,'classes':list(mlb.classes_),'binary_model':'binary_model.pth','multilabel_model':'multilabel_model.pth','legacy_alias':'best_model.pth','binary_threshold':float(bth),'multilabel_thresholds':[float(z) for z in mth],'cv_metrics':cv,'inference_mode':'production_all_data_fit'}
    (directory/'model_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='config.yaml'); parser.add_argument('--features',required=True); parser.add_argument('--annotations',required=True); parser.add_argument('--mode',choices=['production','benchmark','both'],default='both'); args=parser.parse_args()
    with open(args.config,'r',encoding='utf-8') as handle: config=yaml.safe_load(handle)
    device=torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu'); print(f'🔧 Using device: {device}'); directory=Path(config['data']['models_dir']); directory.mkdir(parents=True,exist_ok=True)
    x,labels,names=load_dataset(args.features,args.annotations); cv=cross_validate(x,labels,config)
    if args.mode in {'benchmark','both'}:
        bmodel,mmodel,mlb,scaler,val_data,bl,ml,bth,mth=train(x,labels,names,config,directory,device); bt,bp,mt,mp=probabilities(bmodel,mmodel,val_data,device); _,mf,bpred,mpred=score(bt,bp,mt,mp,bth,mth); print('\n📊 Benchmark holdout:'); print(classification_report(bt,bpred,labels=[0,1],target_names=['нет нарушения','нарушение'],zero_division=0)); print(f'Multi-label micro F1: {f1_score(mt,mpred,average="micro",zero_division=0):.3f}\nMulti-label macro F1: {mf:.3f}')
        if args.mode=='benchmark': save_artifacts(directory,mlb,scaler,x.shape[1],bth,mth,cv)
    if args.mode in {'production','both'}:
        fixed=None if not cv else (cv['binary_threshold'],np.asarray(cv['multilabel_thresholds'],dtype=np.float32)); _,_,mlb,scaler,_,_,_,bth,mth=train(x,labels,names,config,directory,device,production=True,fixed=fixed); save_artifacts(directory,mlb,scaler,x.shape[1],bth,mth,cv); print('\n✅ Production model fitted on all labeled videos with OOF-calibrated thresholds.')
    print('🎉 Training complete!')

if __name__=='__main__': main()
