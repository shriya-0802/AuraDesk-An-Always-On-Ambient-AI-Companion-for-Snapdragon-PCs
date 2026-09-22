"""
Inference Engine
Orchestrates all AI models and processes frames through the complete pipeline.
"""
import time
import numpy as np

from models.face_detector import FaceDetector
from models.emotion_classifier import EmotionClassifier
from models.hand_tracker import HandTracker
from models.gaze_estimator import GazeEstimator
from models.fatigue_detector import FatigueDetector
from core.context_fusion import ContextFusion


class InferenceEngine:
    """Central AI pipeline that runs all models on each frame."""

    def __init__(self):
        print("[InferenceEngine] Initializing AI models...")

        self.face_detector = FaceDetector()
        self.emotion_classifier = EmotionClassifier()
        self.hand_tracker = HandTracker()
        self.gaze_estimator = GazeEstimator()
        self.fatigue_detector = FatigueDetector()
        self.context_fusion = ContextFusion()

        print("[InferenceEngine] All models ready ✓")

    def process_frame(self, frame_rgb: np.ndarray):
        """
        Run the complete inference pipeline on a single frame.

        Args:
            frame_rgb: RGB image (H, W, 3)

        Returns:
            FrameResult with all AI outputs
        """
        from core.camera_manager import FrameResult

        result = FrameResult(
            frame_rgb=frame_rgb,
            timestamp=time.time(),
        )

        # ── Step 1: Face Detection ──
        face_result = self.face_detector.detect(frame_rgb)
        result.face_detected = face_result["detected"]
        result.face_landmarks = face_result["landmarks"]
        result.face_bbox = face_result["bbox"]
        result.num_faces = face_result.get("num_faces", 0)

        if result.face_detected and result.face_landmarks is not None:
            # ── Step 2: Emotion Classification ──
            emotion_result = self.emotion_classifier.classify(result.face_landmarks)
            result.emotion = emotion_result["emotion"]
            result.emotion_confidence = emotion_result["confidence"]
            result.emotion_emoji = emotion_result["emoji"]
            result.emotion_scores = emotion_result["scores"]

            # ── Step 3: Fatigue & Drowsiness Detection ──
            fatigue_result = self.fatigue_detector.detect(result.face_landmarks)
            result.fatigue_level = fatigue_result["fatigue_level"]
            result.blink_rate = fatigue_result["blink_rate"]
            result.total_blinks = fatigue_result["total_blinks"]
            result.ear = fatigue_result["ear"]
            result.is_blinking = fatigue_result["is_blinking"]
            result.yawn_count = fatigue_result["yawn_count"]
            result.is_yawning = fatigue_result["is_yawning"]
            result.is_drowsy = fatigue_result.get("is_drowsy", False)
            result.mar = fatigue_result["mar"]
            result.head_pitch = fatigue_result["head_pitch"]
            result.head_yaw = fatigue_result["head_yaw"]
            result.session_minutes = self.fatigue_detector.get_session_duration_minutes()

            # ── Step 4: Gaze & Attention Estimation ──
            gaze_result = self.gaze_estimator.estimate(result.face_landmarks)
            result.gaze_point = gaze_result["gaze_point"]
            result.gaze_vector = gaze_result["gaze_vector"]
            result.looking_at_screen = gaze_result["looking_at_screen"]
        else:
            result.session_minutes = self.fatigue_detector.get_session_duration_minutes()
            result.looking_at_screen = False

        # ── Step 5: Hand Gesture Recognition ──
        hand_result = self.hand_tracker.detect(frame_rgb)
        result.hands_detected = hand_result["detected"]
        result.hands_data = hand_result["hands"]
        result.active_gesture = hand_result["active_gesture"]
        result.gesture_triggered = hand_result["gesture_triggered"]

        # ── Step 6: Context Fusion ──
        cognitive = self.context_fusion.update(result)
        result.focus_score = cognitive["focus_score"]
        result.stress_level = cognitive["stress_level"]
        result.alerts = cognitive["alerts"]
        result.shoulder_surfing = cognitive.get("shoulder_surfing", False)
        result.awa_trigger = cognitive.get("awa_trigger", False)
        result.trends = cognitive["trends"]

        return result

    def calibrate_gaze(self, landmarks):
        """Calibrate gaze estimation to current eye position."""
        if landmarks is not None:
            self.gaze_estimator.calibrate(landmarks)

    def release(self):
        """Release model resources."""
        self.face_detector.release()
        self.hand_tracker.release()
        print("[InferenceEngine] All models released ✓")
