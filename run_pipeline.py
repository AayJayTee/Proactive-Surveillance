# run_pipeline.py
# ============================================================
# FINAL END-TO-END PIPELINE
# Action + Anomaly + Explainability + Captioning
# ============================================================

import os
import cv2
import json
import sys
import torch
import numpy as np
import warnings
import csv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- SILENCE ALL WARNINGS ----------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "0"

warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

# ---------------- IMPORTS ----------------
from action_recognition.infer_action import run_action_recognition
from action_recognition.cnn3d_model import generate_model
from anomaly_detection.feature_extractor import VideoFeatureExtractor
from anomaly_detection.infer_anomaly import AnomalyInferencer
from explainability.gradcam_action import ActionGradCAM, save_gradcam_video
from explainability.activation_maps import ActivationMapExplainer, save_activation_video
from captioning.caption_frames import SceneCaptioner, save_captions

# ---------------- CONFIG ----------------
ACTION_MODEL_PATH = "action_recognition/best_3dcnn.pth"
ANOMALY_MODEL_PATH = "anomaly_detection/best_anomaly_model.pth"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_KEYFRAMES = 3  # number of keyframes to extract for captioning (reduce for speed)


# ============================================================
# EVENT BUILDER
# ============================================================
def build_timeline(anomaly_scores, fps, action_label, action_conf):
    events = []
    threshold = 0.2                        #change to configure threshold to detect anomalies
    active = None

    for i, score in enumerate(anomaly_scores):
        time_sec = i / fps

        if score > threshold:
            if active is None:
                active = {
                    "start": time_sec,
                    "max_score": float(score)
                }
            else:
                active["max_score"] = max(active["max_score"], float(score))
        else:
            if active:
                active["end"] = time_sec
                active["type"] = "Anomaly"
                events.append(active)
                active = None

    # Add action event
    events.append({
        "start": 0,
        "end": len(anomaly_scores)/fps,
        "type": action_label,
        "confidence": float(action_conf)
    })

    return events


# ============================================================
# REPORT GENERATION
# ============================================================
def export_csv(events, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)


def export_pdf(report_data, path):
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Visionary AI Surveillance Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(report_data["caption"], styles["Normal"]))
    elements.append(Spacer(1, 12))

    for e in report_data["timeline"]:
        text = f"{e['type']} event from {e['start']:.2f}s to {e['end']:.2f}s"
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 6))

    doc.build(elements)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main(VIDEO_PATH):

    # 1️⃣ ACTION
    print("🔹 Loading Action Recognition Model...")
    action_result = run_action_recognition(
        VIDEO_PATH,
        ACTION_MODEL_PATH
    )
    print(f"   ✅ Action Predicted: {action_result['action']}")
    print(f"   📊 Confidence: {action_result['confidence']:.4f}")
    
    # 2️⃣ ANOMALY
    backbone = generate_model(num_classes=101)
    backbone.load_state_dict(torch.load(ACTION_MODEL_PATH, map_location=DEVICE))
    backbone.eval()

    feature_extractor = VideoFeatureExtractor(backbone, device=DEVICE)
    anomaly_model = AnomalyInferencer(
        checkpoint_path=ANOMALY_MODEL_PATH,
        device=DEVICE
    )

    features = feature_extractor.extract(VIDEO_PATH)

    total_frames = int(
        cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FRAME_COUNT)
    )

    anomaly_scores = anomaly_model.infer(features, total_frames)

    fps = cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FPS) or 20

    # 3️⃣ EXPLAINABILITY
    gradcam = ActionGradCAM(ACTION_MODEL_PATH)
    cam_maps, _ = gradcam.generate(VIDEO_PATH)
    save_gradcam_video(VIDEO_PATH, cam_maps, os.path.join(OUTPUT_DIR, "gradcam_video.mp4"))

    activation_explainer = ActivationMapExplainer(ACTION_MODEL_PATH)
    activation_maps = activation_explainer.generate(VIDEO_PATH)
    save_activation_video(VIDEO_PATH, activation_maps, os.path.join(OUTPUT_DIR, "activation_video.mp4"))

    # 4️⃣ CAPTIONING
    captioner = SceneCaptioner(device=DEVICE)
    keyframes = []
    cap = cv2.VideoCapture(VIDEO_PATH)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keyframes.append(frame)
    cap.release()

    # downsample frames to a small set of keyframes for captioning
    keyframes = keyframes[::max(len(keyframes)//CAPTION_KEYFRAMES, 1)]
    frame_captions = captioner.caption_video_frames(keyframes)

    final_caption = captioner.build_final_caption(
        frame_captions,
        action_label=action_result["action"],
        action_confidence=action_result["confidence"]
    )

    # 5️⃣ TIMELINE
    timeline = build_timeline(
        anomaly_scores,
        fps,
        action_result["action"],
        action_result["confidence"]
    )

    # 6️⃣ ALERTS
    alerts = []
    for e in timeline:
        if e.get("max_score", 0) > 0.8:
            alerts.append("🚨 High anomaly detected")

    # 7️⃣ REPORT EXPORT
    report = {
        "action_recognition": action_result,
        "anomaly": {
            "max_score": float(anomaly_scores.max()),
            "mean_score": float(anomaly_scores.mean())
        },
        "caption": final_caption,
        "timeline": timeline,
        "alerts": alerts,
        "outputs": {
            "gradcam_video": "/output/gradcam_video.mp4",
            "activation_video": "/output/activation_video.mp4"
        }
    }

    with open(os.path.join(OUTPUT_DIR, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    export_csv(timeline, os.path.join(OUTPUT_DIR, "events.csv"))
    export_pdf(report, os.path.join(OUTPUT_DIR, "report.pdf"))

    return report


if __name__ == "__main__":
    VIDEO_PATH = sys.argv[1]
    main(VIDEO_PATH)