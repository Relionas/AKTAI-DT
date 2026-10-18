"""
Advanced Feature Extractor for ForteBank Violation Detection
Team: AKTAI
Uses: YOLOv8 + EfficientNet + Optical Flow + Rule-based features
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from ultralytics import YOLO
import timm
from torchvision import transforms
from tqdm import tqdm
import pandas as pd


class AdvancedFeatureExtractor:
    """Extract multi-modal features from surveillance videos"""

    def __init__(self, device='cuda'):
        self.device = device
        print(f"🔧 Initializing Feature Extractor on {device}...")

        # YOLOv8 for object detection
        print("📦 Loading YOLOv8...")
        self.yolo = YOLO('yolov8n.pt')

        # EfficientNet for visual features
        print("🎨 Loading EfficientNet-B4...")
        self.visual_model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        self.visual_model = self.visual_model.to(device)
        self.visual_model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # COCO classes we care about
        self.relevant_classes = {
            0: 'person',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            39: 'bottle',
            41: 'cup',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone'
        }

        print("✅ Feature Extractor ready!\n")

    def extract_video_features(self, video_path: str, target_fps: int = 5) -> Dict:
        """Extract all features from a video"""

        video_path = Path(video_path)
        video_name = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Sample frames at target FPS
        frame_interval = max(1, int(fps / target_fps))

        # Storage for features
        object_detections = []
        visual_features = []
        frames_data = []
        prev_gray = None
        optical_flows = []

        frame_idx = 0
        sampled_frame_idx = 0

        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames
                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / fps

                    # 1. YOLO object detection
                    results = self.yolo(frame, verbose=False)[0]
                    detections = self._parse_yolo_results(results, timestamp)
                    object_detections.extend(detections)

                    # 2. Visual features (EfficientNet)
                    img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
                    visual_feat = self.visual_model(img_tensor).cpu().numpy()[0]
                    visual_features.append(visual_feat)

                    # 3. Optical flow
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
                        optical_flows.append(flow_magnitude)
                    prev_gray = gray

                    frames_data.append({
                        'frame_idx': sampled_frame_idx,
                        'timestamp': timestamp,
                        'person_count': sum(1 for d in detections if d['class'] == 'person')
                    })

                    sampled_frame_idx += 1

                frame_idx += 1

        cap.release()

        # Aggregate features
        features = self._aggregate_features(
            video_name, duration, object_detections,
            visual_features, optical_flows, frames_data
        )

        return features

    def _parse_yolo_results(self, results, timestamp: float) -> List[Dict]:
        """Parse YOLO detection results"""
        detections = []

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls = int(box.cls[0])
            if cls in self.relevant_classes:
                detections.append({
                    'timestamp': timestamp,
                    'class': self.relevant_classes[cls],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                })

        return detections

    def _aggregate_features(self, video_name: str, duration: float,
                           object_detections: List, visual_features: List,
                           optical_flows: List, frames_data: List) -> Dict:
        """Aggregate all features into a single feature vector"""

        # Object-based features
        person_counts = [f['person_count'] for f in frames_data]

        # Person absence detection
        zero_person_frames = sum(1 for c in person_counts if c == 0)
        max_absence_duration = 0
        current_absence = 0
        for c in person_counts:
            if c == 0:
                current_absence += 1
            else:
                max_absence_duration = max(max_absence_duration, current_absence)
                current_absence = 0
        max_absence_duration = max(max_absence_duration, current_absence) * (duration / len(person_counts))

        # Multiple people detection
        multi_person_frames = sum(1 for c in person_counts if c >= 2)
        max_people_duration = 0
        current_multi = 0
        for c in person_counts:
            if c >= 2:
                current_multi += 1
            else:
                max_people_duration = max(max_people_duration, current_multi)
                current_multi = 0
        max_people_duration = max(max_people_duration, current_multi) * (duration / len(person_counts))

        # Object presence
        bag_frames = sum(1 for d in object_detections if d['class'] in ['backpack', 'handbag', 'suitcase'])
        phone_frames = sum(1 for d in object_detections if d['class'] == 'cell phone')
        bottle_frames = sum(1 for d in object_detections if d['class'] in ['bottle', 'cup'])

        # Motion features
        avg_motion = np.mean(optical_flows) if optical_flows else 0
        max_motion = np.max(optical_flows) if optical_flows else 0

        # Visual features (average pooling)
        visual_feat_avg = np.mean(visual_features, axis=0) if visual_features else np.zeros(1792)

        features = {
            'video_name': video_name,
            'duration': duration,
            'num_frames': len(frames_data),

            # Person features
            'avg_person_count': np.mean(person_counts) if person_counts else 0,
            'max_person_count': np.max(person_counts) if person_counts else 0,
            'zero_person_ratio': zero_person_frames / len(person_counts) if person_counts else 0,
            'max_absence_duration': max_absence_duration,

            # Multiple people
            'multi_person_ratio': multi_person_frames / len(person_counts) if person_counts else 0,
            'max_people_duration': max_people_duration,

            # Object presence
            'bag_presence_ratio': bag_frames / len(frames_data) if frames_data else 0,
            'phone_presence_ratio': phone_frames / len(frames_data) if frames_data else 0,
            'bottle_presence_ratio': bottle_frames / len(frames_data) if frames_data else 0,

            # Motion
            'avg_motion': avg_motion,
            'max_motion': max_motion,

            # Visual features (first 100 dims for compactness)
            **{f'visual_{i}': visual_feat_avg[i] for i in range(min(100, len(visual_feat_avg)))}
        }

        return features

    def extract_dataset_features(self, video_dir: str, output_path: str, target_fps: int = 5):
        """Extract features from all videos in a directory"""

        video_dir = Path(video_dir)
        videos = sorted(list(video_dir.glob("*.mov")))

        print(f"📹 Found {len(videos)} videos")
        print(f"🎯 Target FPS: {target_fps}")
        print(f"💾 Output: {output_path}\n")

        all_features = []

        for video_path in tqdm(videos, desc="Extracting features"):
            try:
                features = self.extract_video_features(str(video_path), target_fps=target_fps)
                all_features.append(features)
            except Exception as e:
                print(f"❌ Error processing {video_path.name}: {e}")

        # Save to CSV
        df = pd.DataFrame(all_features)
        df.to_csv(output_path, index=False)

        print(f"\n✅ Features extracted and saved!")
        print(f"   Total videos: {len(all_features)}")
        print(f"   Feature dimensions: {len(df.columns) - 3}")  # exclude video_name, duration, num_frames

        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    extractor = AdvancedFeatureExtractor(device=args.device)
    extractor.extract_dataset_features(args.video_dir, args.output, args.fps)


def extract_features_for_inference(video_path: str, device: str = 'cuda', target_fps: int = 5) -> Dict:
    """Extract the same 111-feature schema used during classifier training."""
    return AdvancedFeatureExtractor(device=device).extract_video_features(video_path, target_fps=target_fps)


__all__ = ['AdvancedFeatureExtractor', 'extract_features_for_inference']
