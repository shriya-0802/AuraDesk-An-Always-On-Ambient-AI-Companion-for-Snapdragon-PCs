"""
Fatigue Detector Module
Detects user fatigue through:
  1. Eye Aspect Ratio (EAR) — adaptive blink detection & drowsiness
  2. Mouth Aspect Ratio (MAR) — instant yawn detection
  3. Head pose analysis — head drooping & nodding off
"""
import time
import numpy as np
from collections import deque
from config.settings import (
    EAR_THRESHOLD,
    EAR_CONSECUTIVE_FRAMES,
    YAWN_THRESHOLD,
    FATIGUE_BLINK_RATE,
    FATIGUE_YAWN_COUNT,
)


class FatigueDetector:
    """Detects fatigue through blink rate, prolonged eye closure, yawns, and head droop."""

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454

    def __init__(self):
        # Blink tracking
        self._blink_counter = 0
        self._ear_below_threshold_frames = 0
        self._blink_timestamps = deque(maxlen=200)

        # Adaptive open-eye baseline tracking
        self._open_ear_history = deque(maxlen=60)
        self._open_ear_baseline = 0.28

        # Yawn tracking
        self._yawn_counter = 0
        self._is_yawning = False
        self._yawn_frames = 0
        self._yawn_timestamps = deque(maxlen=50)

        # Drowsiness tracking
        self._is_drowsy = False

        # Session
        self._session_start = time.time()

        # Fatigue level (0-1, smoothed)
        self._fatigue_level = 0.10

    def detect(self, landmarks: np.ndarray) -> dict:
        """
        Analyze landmarks for fatigue indicators.

        Args:
            landmarks: np.ndarray (478, 3) face mesh landmarks in [0, 1]

        Returns:
            dict with fatigue metrics
        """
        if landmarks is None or len(landmarks) < 468:
            return self._default_result()

        now = time.time()

        # ── 1. Eye Aspect Ratio (EAR) ──
        left_ear = self._compute_ear(landmarks, self.LEFT_EYE)
        right_ear = self._compute_ear(landmarks, self.RIGHT_EYE)
        avg_ear = float((left_ear + right_ear) / 2.0)

        # Update running open-eye baseline (TFLite model gives ~0.20-0.35 for open eyes)
        if avg_ear > 0.15:
            self._open_ear_history.append(avg_ear)
            if len(self._open_ear_history) >= 10:
                self._open_ear_baseline = float(np.median(self._open_ear_history))

        # Dynamic blink threshold (40% below open-eye baseline)
        blink_thresh = max(0.12, min(0.20, self._open_ear_baseline * 0.60))
        is_blinking = False

        if avg_ear < blink_thresh:
            self._ear_below_threshold_frames += 1
        else:
            # Blink detected on release of closure
            if 1 <= self._ear_below_threshold_frames <= 10:
                self._blink_counter += 1
                self._blink_timestamps.append(now)
                is_blinking = True
            self._ear_below_threshold_frames = 0

        # Drowsiness / prolonged closure (> 0.6s at ~30fps = 18 frames)
        self._is_drowsy = self._ear_below_threshold_frames > 18

        # Blink rate calculation (blinks in last 60s)
        recent_blinks = [t for t in self._blink_timestamps if now - t < 60]
        elapsed = max(now - self._session_start, 1.0)
        if elapsed < 40:
            observed_rate = len(recent_blinks) * (60.0 / elapsed)
            weight = elapsed / 40.0
            blink_rate = weight * observed_rate + (1.0 - weight) * 14.0
        else:
            blink_rate = float(len(recent_blinks))

        # ── 2. Mouth Aspect Ratio (MAR) — Yawn Detection ──
        mar = self._compute_mar(landmarks)
        was_yawning = self._is_yawning

        # Trigger yawn when mouth opens wide (> 0.25 MAR for 4+ frames)
        # TFLite model gives MAR ~0.02-0.06 for closed mouth, ~0.15+ for open mouth
        if mar > 0.25:
            self._yawn_frames += 1
            if self._yawn_frames >= 4:
                self._is_yawning = True
                if not was_yawning:
                    self._yawn_counter += 1
                    self._yawn_timestamps.append(now)
        else:
            self._yawn_frames = 0
            self._is_yawning = False

        recent_yawns = len([t for t in self._yawn_timestamps if now - t < 180])

        # ── 3. Head Pitch (Drooping / Nodding off) ──
        head_pitch, head_yaw = self._estimate_head_pose(landmarks)

        # ── 4. Compute Overall Fatigue Level ──
        fatigue_signals = [0.10]

        # Drowsiness / eye closure
        if self._is_drowsy:
            closure_duration = self._ear_below_threshold_frames / 30.0
            fatigue_signals.append(min(0.55, 0.35 + closure_duration * 0.20))

        # Yawning
        if self._is_yawning:
            fatigue_signals.append(0.45)
        elif recent_yawns > 0:
            fatigue_signals.append(min(0.35, recent_yawns * 0.20))

        # High blink rate
        if blink_rate > 22:
            blink_fatigue = min(0.30, (blink_rate - 22) / 18.0 * 0.30)
            fatigue_signals.append(blink_fatigue)

        # Head drooping forward
        if head_pitch > 0.30:
            fatigue_signals.append(min(0.25, (head_pitch - 0.30) * 0.5))

        # Long session drift
        session_hours = elapsed / 3600.0
        fatigue_signals.append(min(0.15, session_hours * 0.08))

        raw_fatigue = min(1.0, sum(fatigue_signals))

        # Fast response when drowsiness/yawn triggers, smooth recovery
        alpha = 0.30 if raw_fatigue > self._fatigue_level else 0.08
        self._fatigue_level = (1.0 - alpha) * self._fatigue_level + alpha * raw_fatigue

        return {
            "fatigue_level": float(np.clip(self._fatigue_level, 0.0, 1.0)),
            "blink_rate": float(blink_rate),
            "total_blinks": self._blink_counter,
            "ear": avg_ear,
            "is_blinking": is_blinking,
            "yawn_count": self._yawn_counter,
            "is_yawning": self._is_yawning,
            "is_drowsy": self._is_drowsy,
            "mar": float(mar),
            "head_pitch": float(head_pitch),
            "head_yaw": float(head_yaw),
        }

    def _compute_ear(self, lm: np.ndarray, indices: list) -> float:
        """Compute Eye Aspect Ratio."""
        p = lm[indices, :2]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return float((v1 + v2) / (2.0 * max(h, 1e-6)))

    def _compute_mar(self, lm: np.ndarray) -> float:
        """Compute Mouth Aspect Ratio."""
        mouth_h = np.linalg.norm(lm[self.MOUTH_TOP, :2] - lm[self.MOUTH_BOTTOM, :2])
        mouth_w = np.linalg.norm(lm[self.MOUTH_LEFT, :2] - lm[self.MOUTH_RIGHT, :2])
        return float(mouth_h / max(mouth_w, 1e-6))

    def _estimate_head_pose(self, lm: np.ndarray) -> tuple:
        """Estimate vertical pitch and horizontal yaw."""
        forehead = lm[self.FOREHEAD, :2]
        nose = lm[self.NOSE_TIP, :2]
        chin = lm[self.CHIN, :2]
        left_cheek = lm[self.LEFT_CHEEK, :2]
        right_cheek = lm[self.RIGHT_CHEEK, :2]

        upper_h = max(1e-6, np.linalg.norm(nose - forehead))
        lower_h = max(1e-6, np.linalg.norm(chin - nose))
        ratio = upper_h / lower_h
        pitch = max(0.0, float(ratio - 1.0))

        dist_left = np.linalg.norm(nose - left_cheek)
        dist_right = np.linalg.norm(nose - right_cheek)
        total_span = max(dist_left + dist_right, 1e-6)
        yaw = float((dist_left - dist_right) / total_span)

        return pitch, yaw

    def _default_result(self):
        return {
            "fatigue_level": float(self._fatigue_level),
            "blink_rate": 0.0,
            "total_blinks": self._blink_counter,
            "ear": 0.45,
            "is_blinking": False,
            "yawn_count": self._yawn_counter,
            "is_yawning": False,
            "is_drowsy": False,
            "mar": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
        }

    def get_session_duration_minutes(self) -> float:
        """Get session duration in minutes."""
        return (time.time() - self._session_start) / 60.0
