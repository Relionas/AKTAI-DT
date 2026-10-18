"""
Training Pipeline - Extract Features and Train Classifier
Team: AKTAI
Full pipeline for violation detection model training
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: list, description: str):
    """Run a command and show progress"""
    print("\n" + "=" * 80)
    print(f"🚀 {description}")
    print("=" * 80)

    result = subprocess.run(command, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Error in: {description}")
        sys.exit(1)

    print(f"✅ {description} - DONE!")
    return result


def main():
    """Run full training pipeline"""

    print("\n" + "=" * 80)
    print("🎯 FORTEBANK VIOLATION DETECTION - FULL TRAINING PIPELINE")
    print("Team: AKTAI")
    print("=" * 80)

    # Paths
    video_dir = r"C:\Users\vanac\OneDrive\Desktop\datathon problem\open\videos"
    # Keep the annotation path local to the project; the upload folder is temporary.
    annotations_path = r"C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT\data\open_labels_detailed.xlsx"
    features_output = r"C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT\ml_pipeline\features\extracted_features.csv"
    config_path = r"C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT\ml_pipeline\config.yaml"

    # Create directories
    Path(features_output).parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract Features
    print("\n📊 STEP 1: FEATURE EXTRACTION")
    print("This will take ~20-30 minutes for 42 videos with RTX 4070")
    input("Press Enter to start feature extraction...")

    run_command([
        sys.executable,
        r"C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT\ml_pipeline\feature_extractor.py",
        "--video_dir", video_dir,
        "--output", features_output,
        "--fps", "5",
        "--device", "cuda"
    ], "Feature Extraction")

    # Step 2: Train Classifier
    print("\n🧠 STEP 2: TRAIN CLASSIFIER")
    print("Training deep learning classifier on extracted features")
    input("Press Enter to start training...")

    run_command([
        sys.executable,
        r"C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT\ml_pipeline\train_classifier.py",
        "--config", config_path,
        "--features", features_output,
        "--annotations", annotations_path
    ], "Classifier Training")

    print("\n" + "=" * 80)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print("\n📦 Trained artifacts saved in:")
    print(f"   - {r'C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT\ml_pipeline\models\best_model.pth'}")
    print(f"   - {r'C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT\ml_pipeline\models\label_encoder.pkl'}")
    print(f"   - {r'C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT\ml_pipeline\models\scaler.pkl'}")
    print("\n🚀 Next step: Integrate trained model into inference pipeline!")


if __name__ == "__main__":
    main()
