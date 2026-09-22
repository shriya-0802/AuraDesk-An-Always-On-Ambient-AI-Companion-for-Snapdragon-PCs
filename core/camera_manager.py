"""
Camera Manager
Threaded webcam capture using OpenCV + QThread.
Runs the full AI inference pipeline on each frame.
"""
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from PyQt6.QtCore import QThread, pyqtSignal, QMutex

from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


@dataclass
class FrameResult:
    """Complete result from processing a single frame."""
    # Raw frame
    frame_rgb: Optional[np.ndarray] = None
    timestamp: float = 0.0
    fps: float = 0.0

    # Face detection
    face_detected: bool = False
    face_landmarks: Optional[np.ndarray] = None
    face_bbox: Optional[tuple] = None
    num_faces: int = 0

    # Emotion
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    emotion_emoji: str = "😐"
    emotion_scores: Dict[str, float] = field(default_factory=dict)

    # Fatigue
    fatigue_level: float = 0.0
    blink_rate: float = 0.0
    total_blinks: int = 0
    ear: float = 0.45
    is_blinking: bool = False
    yawn_count: int = 0
    is_yawning: bool = False
    is_drowsy: bool = False
    mar: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0

    # Hands
    hands_detected: bool = False
    hands_data: list = field(default_factory=list)
    active_gesture: str = "none"
    gesture_triggered: bool = False

    # Gaze
    gaze_point: tuple = (0.5, 0.5)
    gaze_vector: tuple = (0.0, 0.0)
    looking_at_screen: bool = True

    # Cognitive state (computed by context fusion)
    focus_score: float = 0.50
    stress_level: float = 0.14
    session_minutes: float = 0.0
    alerts: list = field(default_factory=list)
    shoulder_surfing: bool = False
    awa_trigger: bool = False
    trends: Dict[str, list] = field(default_factory=dict)


class CameraThread(QThread):
    """Background thread for webcam capture and AI inference."""

    frame_ready = pyqtSignal(object)  # Emits FrameResult
    error_occurred = pyqtSignal(str)
    camera_opened = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._mutex = QMutex()
        self._cap = None
        self._inference_engine = None
        self._frame_count = 0
        self._fps_start = time.time()
        self._current_fps = 0.0

    def set_inference_engine(self, engine):
        """Set the inference engine to use for processing frames."""
        self._inference_engine = engine

    def run(self):
        """Main capture loop."""
        self._running = True

        # Open camera
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            self.error_occurred.emit("Could not open camera. Please check permissions.")
            self.camera_opened.emit(False)
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        self.camera_opened.emit(True)

        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self.msleep(10)
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Compute FPS
            self._frame_count += 1
            elapsed = time.time() - self._fps_start
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start = time.time()

            # Run inference pipeline
            if self._inference_engine:
                result = self._inference_engine.process_frame(frame_rgb)
                result.fps = self._current_fps
            else:
                result = FrameResult(
                    frame_rgb=frame_rgb,
                    timestamp=time.time(),
                    fps=self._current_fps,
                )

            self.frame_ready.emit(result)

            # Small sleep
            self.msleep(5)

        # Cleanup
        if self._cap:
            self._cap.release()

    def stop(self):
        """Stop capture thread."""
        self._running = False
        self.wait(3000)
