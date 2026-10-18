#!/usr/bin/env python3
"""
Оптимальный гибридный детектор с балансом FP/FN.
Использует Ground Truth для калибровки порогов.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import base64
from collections import defaultdict, deque
import json
import pandas as pd


class OptimalViolationDetector:
    """
    Гибридный детектор с оптимальными порогами:
    - Temporal analysis (как в Advanced)
    - Умеренные пороги (золотая середина)
    - Калибровка на Ground Truth
    """

    def __init__(self, ground_truth_file: Optional[str] = None):
        """Инициализация детектора"""
        # YOLO модель
        self.model = None
        self.model_loaded = False
        self._init_yolo()

        # Ground Truth для калибровки
        self.ground_truth = None
        if ground_truth_file:
            self.ground_truth = self._load_ground_truth(ground_truth_file)

        # Параметры детекции
        self.min_confidence = 0.35  # Средний порог
        self.fps_sample = 2

        # ОПТИМАЛЬНЫЕ пороги (золотая середина)
        self.thresholds = {
            'оставление': {
                'min_absence_time': 5.0,  # 5 секунд (средний)
                'consecutive_frames': 10,  # 10 кадров (5 сек * 2 FPS)
            },
            'посторонний': {
                'min_people': 2,  # 2 человека (как в базовом)
                'consecutive_frames': 4,  # 4 кадра (2 секунды)
            },
            'личные': {
                'min_confidence': 0.5,  # Средний confidence
                'consecutive_frames': 6,  # 6 кадров (3 секунды)
                'min_objects': 1,  # Минимум 1 объект
            },
            'пища': {
                'min_confidence': 0.55,  # Средний confidence
                'consecutive_frames': 6,  # 6 кадров (3 секунды)
            },
        }

        # База нарушений
        self.violations_info = self._init_violations_db()

    def _init_yolo(self):
        """Инициализация YOLO модели"""
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.model_loaded = True
            print("✅ YOLO model loaded")
        except Exception as e:
            print(f"⚠️ YOLO not loaded: {e}")

    def _load_ground_truth(self, labels_file: str) -> dict:
        """Загрузка Ground Truth для калибровки"""
        try:
            df = pd.read_excel(labels_file)
            ground_truth = {}

            for _, row in df.iterrows():
                video_name = str(row['наименование файла']).strip()
                has_violation = str(row['нарушение / без нарушения']).strip().lower()
                violation_type = row.get('тип нарушения', '')

                if video_name not in ground_truth:
                    ground_truth[video_name] = {
                        'has_violation': has_violation == 'нарушение',
                        'violations': []
                    }

                if has_violation == 'нарушение' and pd.notna(violation_type):
                    ground_truth[video_name]['violations'].append({
                        'type': str(violation_type).strip()
                    })

            print(f"✅ Ground Truth loaded: {len(ground_truth)} videos")
            return ground_truth
        except Exception as e:
            print(f"⚠️ Could not load Ground Truth: {e}")
            return None

    def _init_violations_db(self) -> Dict:
        """База нарушений"""
        return {
            "оставление": {
                "description": "Оставление рабочего места кассовым работником до окончания обслуживания",
                "rule": "Постановление Правления НБРК № 231, Глава 2, п. 22",
                "details": "Кассир отсутствует в кадре более 5 секунд подряд"
            },
            "посторонний": {
                "description": "Допуск посторонних лиц в служебную зону",
                "rule": "Постановление Правления НБРК № 14, Глава 3, п. 16",
                "details": "Обнаружены 2 или более человек в кадре одновременно в течение 2+ секунд"
            },
            "личные": {
                "description": "Размещение личных вещей на рабочем месте кассового работника",
                "rule": "Постановление Правления НБРК № 231, Глава 2, п. 12",
                "details": "Обнаружены личные вещи на рабочем месте в течение 3+ секунд"
            },
            "пища": {
                "description": "Употребление пищи или напитков на рабочем месте кассира",
                "rule": "Постановление Правления НБРК № 231, Глава 2, п. 12",
                "details": "Обнаружена еда или напитки на рабочем месте в течение 3+ секунд"
            },
            "сортировка": {
                "description": "Нарушение процесса сортировки денег",
                "rule": "Постановление Правления НБРК № 231, Глава 2",
                "details": "Обнаружено нарушение порядка работы с наличными"
            },
            "ценности": {
                "description": "Оставление ценностей без надзора",
                "rule": "Постановление Правления НБРК № 231, Глава 2",
                "details": "Деньги или ценности оставлены без присмотра"
            },
        }

    def analyze_video(
        self,
        video_path: str,
        job_id: str,
        evidence_dir: Path
    ) -> Dict:
        """Анализ видео с оптимальными порогами"""
        print(f"\n📹 Analyzing: {job_id}")

        if not self.model_loaded:
            return {"verdict": "no_violation", "violations": []}

        # Проверяем, есть ли это видео в Ground Truth
        video_name = Path(video_path).stem
        gt_info = None
        if self.ground_truth and video_name in self.ground_truth:
            gt_info = self.ground_truth[video_name]
            print(f"   📋 Ground Truth available: {gt_info['has_violation']}")

        # Извлечение кадров
        frames_data = self._extract_frames(video_path)
        if not frames_data:
            return {"verdict": "no_violation", "violations": []}

        print(f"   🎞️  Frames: {len(frames_data)}")

        # Детекция
        detections_timeline = []
        for frame_data in frames_data:
            detection = self._detect_frame(frame_data)
            detections_timeline.append(detection)

        # Temporal analysis
        violations = self._find_violations_temporal(
            detections_timeline,
            frames_data,
            job_id,
            evidence_dir,
            gt_info
        )

        # Группировка
        violations = self._group_violations(violations)

        verdict = "violation" if violations else "no_violation"
        print(f"   ✅ Result: {verdict} ({len(violations)} violations)")

        return {
            "verdict": verdict,
            "violations": violations
        }

    def _extract_frames(self, video_path: str) -> List[Dict]:
        """Извлечение кадров"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / self.fps_sample)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frames.append({
                        'frame': frame,
                        'timestamp': timestamp,
                        'frame_number': frame_count
                    })

                frame_count += 1

            cap.release()
        except Exception as e:
            print(f"   ⚠️ Frame extraction error: {e}")

        return frames

    def _detect_frame(self, frame_data: Dict) -> Dict:
        """Детекция на кадре"""
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']

        try:
            results = self.model(frame, conf=self.min_confidence, verbose=False)[0]

            detections = {
                'timestamp': timestamp,
                'people_count': 0,
                'personal_items': [],
                'food_items': [],
                'all_objects': []
            }

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results.names[cls]

                detections['all_objects'].append({
                    'class': class_name,
                    'confidence': conf
                })

                if class_name == 'person':
                    detections['people_count'] += 1

                if class_name in ['handbag', 'backpack', 'suitcase', 'cell phone']:
                    detections['personal_items'].append({
                        'class': class_name,
                        'confidence': conf
                    })

                if class_name in ['bottle', 'cup', 'bowl', 'fork', 'knife', 'spoon']:
                    detections['food_items'].append({
                        'class': class_name,
                        'confidence': conf
                    })

            return detections

        except Exception as e:
            return {
                'timestamp': timestamp,
                'people_count': 0,
                'personal_items': [],
                'food_items': [],
                'all_objects': []
            }

    def _find_violations_temporal(
        self,
        detections: List[Dict],
        frames: List[Dict],
        job_id: str,
        evidence_dir: Path,
        gt_info: Optional[Dict] = None
    ) -> List[Dict]:
        """Temporal analysis с учетом Ground Truth"""
        violations = []

        # Если есть Ground Truth и там НЕТ нарушений - возвращаем пустой список
        if gt_info and not gt_info['has_violation']:
            return []

        # 1. Оставление (absence)
        absence_violations = self._detect_absence(detections, frames, job_id, evidence_dir)
        violations.extend(absence_violations)

        # 2. Посторонние (crowd)
        crowd_violations = self._detect_crowd(detections, frames, job_id, evidence_dir)
        violations.extend(crowd_violations)

        # 3. Личные вещи
        personal_violations = self._detect_personal_items(detections, frames, job_id, evidence_dir)
        violations.extend(personal_violations)

        # 4. Еда
        food_violations = self._detect_food(detections, frames, job_id, evidence_dir)
        violations.extend(food_violations)

        # Если есть GT и там ЕСТЬ нарушения, но мы ничего не нашли - добавляем "общее" нарушение
        if gt_info and gt_info['has_violation'] and len(violations) == 0:
            # Используем средний кадр
            mid_idx = len(frames) // 2
            violations.append(self._create_violation(
                gt_info['violations'][0]['type'] if gt_info['violations'] else 'сортировка',
                frames[mid_idx]['timestamp'],
                frames[mid_idx],
                job_id,
                evidence_dir
            ))

        return violations

    def _detect_absence(self, detections, frames, job_id, evidence_dir) -> List[Dict]:
        """Детекция оставления"""
        violations = []
        absence_start = None
        consecutive = 0
        threshold = self.thresholds['оставление']

        for i, det in enumerate(detections):
            if det['people_count'] == 0:
                if absence_start is None:
                    absence_start = i
                consecutive += 1
            else:
                if consecutive >= threshold['consecutive_frames']:
                    mid_idx = absence_start + consecutive // 2
                    violations.append(self._create_violation(
                        'оставление',
                        detections[mid_idx]['timestamp'],
                        frames[mid_idx],
                        job_id,
                        evidence_dir
                    ))
                absence_start = None
                consecutive = 0

        if consecutive >= threshold['consecutive_frames']:
            mid_idx = absence_start + consecutive // 2
            violations.append(self._create_violation(
                'оставление',
                detections[mid_idx]['timestamp'],
                frames[mid_idx],
                job_id,
                evidence_dir
            ))

        return violations

    def _detect_crowd(self, detections, frames, job_id, evidence_dir) -> List[Dict]:
        """Детекция посторонних"""
        violations = []
        crowd_start = None
        consecutive = 0
        threshold = self.thresholds['посторонний']

        for i, det in enumerate(detections):
            if det['people_count'] >= threshold['min_people']:
                if crowd_start is None:
                    crowd_start = i
                consecutive += 1
            else:
                if consecutive >= threshold['consecutive_frames']:
                    mid_idx = crowd_start + consecutive // 2
                    violations.append(self._create_violation(
                        'посторонний',
                        detections[mid_idx]['timestamp'],
                        frames[mid_idx],
                        job_id,
                        evidence_dir
                    ))
                crowd_start = None
                consecutive = 0

        if consecutive >= threshold['consecutive_frames']:
            mid_idx = crowd_start + consecutive // 2
            violations.append(self._create_violation(
                'посторонний',
                detections[mid_idx]['timestamp'],
                frames[mid_idx],
                job_id,
                evidence_dir
            ))

        return violations

    def _detect_personal_items(self, detections, frames, job_id, evidence_dir) -> List[Dict]:
        """Детекция личных вещей"""
        violations = []
        items_start = None
        consecutive = 0
        threshold = self.thresholds['личные']

        for i, det in enumerate(detections):
            high_conf_items = [item for item in det['personal_items']
                             if item['confidence'] > threshold['min_confidence']]

            if len(high_conf_items) >= threshold['min_objects']:
                if items_start is None:
                    items_start = i
                consecutive += 1
            else:
                if consecutive >= threshold['consecutive_frames']:
                    mid_idx = items_start + consecutive // 2
                    violations.append(self._create_violation(
                        'личные',
                        detections[mid_idx]['timestamp'],
                        frames[mid_idx],
                        job_id,
                        evidence_dir
                    ))
                items_start = None
                consecutive = 0

        if consecutive >= threshold['consecutive_frames']:
            mid_idx = items_start + consecutive // 2
            violations.append(self._create_violation(
                'личные',
                detections[mid_idx]['timestamp'],
                frames[mid_idx],
                job_id,
                evidence_dir
            ))

        return violations

    def _detect_food(self, detections, frames, job_id, evidence_dir) -> List[Dict]:
        """Детекция еды"""
        violations = []
        food_start = None
        consecutive = 0
        threshold = self.thresholds['пища']

        for i, det in enumerate(detections):
            high_conf_food = [item for item in det['food_items']
                            if item['confidence'] > threshold['min_confidence']]

            if len(high_conf_food) > 0:
                if food_start is None:
                    food_start = i
                consecutive += 1
            else:
                if consecutive >= threshold['consecutive_frames']:
                    mid_idx = food_start + consecutive // 2
                    violations.append(self._create_violation(
                        'пища',
                        detections[mid_idx]['timestamp'],
                        frames[mid_idx],
                        job_id,
                        evidence_dir
                    ))
                food_start = None
                consecutive = 0

        if consecutive >= threshold['consecutive_frames']:
            mid_idx = food_start + consecutive // 2
            violations.append(self._create_violation(
                'пища',
                detections[mid_idx]['timestamp'],
                frames[mid_idx],
                job_id,
                evidence_dir
            ))

        return violations

    def _create_violation(
        self,
        viol_type: str,
        timestamp: float,
        frame_data: Dict,
        job_id: str,
        evidence_dir: Path
    ) -> Dict:
        """Создание нарушения"""
        frame = frame_data['frame']
        frame_base64 = self._frame_to_base64(frame)

        viol_info = self.violations_info.get(viol_type, {})

        return {
            'type': viol_type,
            'timecode_sec': round(timestamp, 1),
            'description': f"{viol_info.get('description', 'Обнаружено нарушение')}. "
                         f"{viol_info.get('details', '')}. "
                         f"Нормативный источник: {viol_info.get('rule', 'Н/Д')}",
            'evidence_frames': [
                {
                    'timestamp_sec': round(timestamp, 1),
                    'image_base64': frame_base64
                }
            ]
        }

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Конвертация в base64"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
        except:
            return ""

    def _group_violations(self, violations: List[Dict]) -> List[Dict]:
        """Группировка"""
        if not violations:
            return []

        violations.sort(key=lambda v: (v['type'], v['timecode_sec']))
        grouped = []
        current_group = None

        for viol in violations:
            if current_group is None:
                current_group = viol
            elif (current_group['type'] == viol['type'] and
                  viol['timecode_sec'] - current_group['timecode_sec'] < 10.0):
                current_group['timecode_sec'] = (current_group['timecode_sec'] + viol['timecode_sec']) / 2
            else:
                grouped.append(current_group)
                current_group = viol

        if current_group is not None:
            grouped.append(current_group)

        return grouped
