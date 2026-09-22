"""
AuraDesk Configuration
All thresholds, model paths, and UI settings.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ─── Platform Detection ───
USE_NPU = os.environ.get("AURADESK_USE_NPU", "0") == "1"
PLATFORM = "snapdragon" if USE_NPU else "cpu"

# ─── Camera Settings ───
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ─── AI Model Settings ───
FACE_DETECTION_CONFIDENCE = 0.6
FACE_MESH_CONFIDENCE = 0.5
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 2

# ─── Emotion Thresholds (landmark-based) ───
SMILE_THRESHOLD = 0.35
SURPRISE_THRESHOLD = 0.55
ANGER_THRESHOLD = 0.25
SAD_THRESHOLD = 0.20

# ─── Fatigue Detection ───
EAR_THRESHOLD = 0.22          # Eye Aspect Ratio below this = blink
EAR_CONSECUTIVE_FRAMES = 3    # Frames below threshold = confirmed blink
YAWN_THRESHOLD = 0.6           # Mouth Aspect Ratio above this = yawn
FATIGUE_BLINK_RATE = 25        # Blinks/min above this = fatigued
FATIGUE_YAWN_COUNT = 3         # Yawns in 5 min = fatigued

# ─── Gaze Settings ───
GAZE_SMOOTHING_FACTOR = 0.3   # Exponential smoothing
GAZE_HEATMAP_DECAY = 0.995    # Heatmap decay rate per frame

# ─── Gesture Definitions ───
GESTURES = {
    "fist": "SCREENSHOT",
    "open_palm": "PLAY / PAUSE",
    "point_up": "VOLUME UP",
    "point_down": "VOLUME DOWN",
    "peace": "SWITCH DESKTOP",
    "thumbs_up": "CONFIRM",
    "none": "",
}

GESTURE_COOLDOWN_MS = 1500  # Minimum ms between gesture triggers

# ─── Cognitive State Weights ───
FOCUS_DECAY_RATE = 0.02       # Focus score decay when looking away
FOCUS_GAIN_RATE = 0.05        # Focus score gain when looking at screen
STRESS_BLINK_WEIGHT = 0.3     # How much blink rate affects stress
STRESS_EMOTION_WEIGHT = 0.7   # How much emotion affects stress

# ─── Wellness Thresholds ───
BREAK_REMINDER_MINUTES = 45   # Remind after continuous work
FATIGUE_WARNING_LEVEL = 0.7   # 0-1 scale
STRESS_WARNING_LEVEL = 0.6    # 0-1 scale
HIGH_FOCUS_DND_LEVEL = 0.85   # Auto-DND when focus above this

# ─── UI Colors (Qualcomm Snapdragon Theme) ───
@dataclass
class Colors:
    # Backgrounds (Carbon Black / Charcoal)
    bg_primary: str = "#050505"
    bg_secondary: str = "#0B0B0C"
    bg_panel: str = "rgba(16, 16, 18, 0.85)"
    bg_card: str = "rgba(24, 24, 28, 0.7)"
    bg_hover: str = "rgba(35, 35, 40, 0.8)"

    # Accents (Snapdragon Reds)
    accent_cyan: str = "#FF1414"     # Replaced with Snapdragon Red
    accent_magenta: str = "#D30000"  # Crimson
    accent_lime: str = "#FFFFFF"     # White (for contrast/focus)
    accent_amber: str = "#FF4400"
    accent_coral: str = "#FF2222"
    accent_purple: str = "#A00000"

    # Text
    text_primary: str = "#FFFFFF"
    text_secondary: str = "rgba(255, 255, 255, 0.7)"
    text_muted: str = "rgba(255, 255, 255, 0.4)"

    # Borders
    border_subtle: str = "rgba(255, 255, 255, 0.1)"
    border_accent: str = "rgba(255, 20, 20, 0.3)"

    # Shadows
    shadow_dark: str = "rgba(0, 0, 0, 0.6)"
    shadow_glow_cyan: str = "rgba(255, 20, 20, 0.15)"    # Red glow
    shadow_glow_magenta: str = "rgba(211, 0, 0, 0.15)"   # Crimson glow

COLORS = Colors()

# ─── UI Dimensions ───
WINDOW_MIN_WIDTH = 1280
WINDOW_MIN_HEIGHT = 800
SIDEBAR_WIDTH = 72
PANEL_BORDER_RADIUS = 12
CARD_BORDER_RADIUS = 8

# ─── Emotion Mapping (Professional) ───
EMOTION_EMOJIS = {
    "happy": "RELAXED",
    "sad": "LOW",
    "angry": "FRUSTRATED",
    "surprised": "ALERT",
    "fearful": "ANXIOUS",
    "disgusted": "NEGATIVE",
    "neutral": "CALM",
    "focused": "FOCUSED",
}

# ─── ONNX Model Paths (for Snapdragon deployment) ───
ONNX_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "onnx_models")
FACE_DETECTION_ONNX = os.path.join(ONNX_MODELS_DIR, "face_detection.onnx")
EMOTION_ONNX = os.path.join(ONNX_MODELS_DIR, "emotion_fer.onnx")
HAND_LANDMARK_ONNX = os.path.join(ONNX_MODELS_DIR, "hand_landmark.onnx")
GAZE_ESTIMATION_ONNX = os.path.join(ONNX_MODELS_DIR, "gaze_estimation.onnx")
