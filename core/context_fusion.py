"""
Context Fusion Engine
Combines all AI model outputs into a unified cognitive state: Focus, Stress, and Fatigue.
Applies temporal smoothing, fast distraction drop, and adaptive wellness alerts.
"""
import time
import numpy as np
from collections import deque
from config.settings import (
    FATIGUE_WARNING_LEVEL,
    STRESS_WARNING_LEVEL,
    HIGH_FOCUS_DND_LEVEL,
    BREAK_REMINDER_MINUTES,
)


class ContextFusion:
    """Fuses outputs from all AI models into a coherent cognitive state."""

    def __init__(self):
        # Smoothed cognitive scores
        self._focus_score = 0.50
        self._stress_level = 0.14

        # History for trend sparklines
        self._focus_history = deque(maxlen=180)
        self._stress_history = deque(maxlen=180)
        self._fatigue_history = deque(maxlen=180)
        self._emotion_history = deque(maxlen=60)

        # Event tracking
        self._last_break_reminder = time.time()
        self._continuous_focus_start = None
        self._distraction_start = None

        # Alerts
        self._active_alerts = []

    def update(self, frame_result) -> dict:
        """
        Update cognitive state from latest frame result.

        Args:
            frame_result: FrameResult from inference engine

        Returns:
            dict with 'focus_score', 'stress_level', 'alerts', 'trends'
        """
        now = time.time()

        # ── 1. Focus Score Calculation ──
        if frame_result.face_detected and frame_result.looking_at_screen:
            # User is looking directly at screen
            gain = 0.008
            if getattr(frame_result, 'is_yawning', False) or getattr(frame_result, 'is_drowsy', False):
                gain = -0.015

            self._focus_score = min(0.96, self._focus_score + gain)
            if self._continuous_focus_start is None:
                self._continuous_focus_start = now
            self._distraction_start = None
        else:
            # User is looking away or turned away
            decay = 0.015 if frame_result.face_detected else 0.025
            self._focus_score = max(0.05, self._focus_score - decay)
            self._continuous_focus_start = None
            if self._distraction_start is None:
                self._distraction_start = now

        # ── 2. Stress Level Calculation ──
        stress_signals = [0.12]

        if frame_result.emotion in ("angry", "fearful"):
            stress_signals.append(frame_result.emotion_confidence * 0.45)
        elif frame_result.emotion == "sad":
            stress_signals.append(frame_result.emotion_confidence * 0.20)
        elif frame_result.emotion == "happy":
            stress_signals.append(-0.06)

        if frame_result.blink_rate > 22:
            blink_stress = min(0.30, (frame_result.blink_rate - 22) / 18.0 * 0.30)
            stress_signals.append(blink_stress)

        if frame_result.face_landmarks is not None and len(frame_result.face_landmarks) >= 300:
            lm = frame_result.face_landmarks
            brow_dist = np.linalg.norm(lm[55, :2] - lm[285, :2])
            eye_dist = max(1e-6, np.linalg.norm(lm[33, :2] - lm[263, :2]))
            brow_ratio = brow_dist / eye_dist
            if brow_ratio < 0.28:
                tension = (0.28 - brow_ratio) * 2.0
                stress_signals.append(min(0.35, tension))

        raw_stress = float(np.clip(sum(stress_signals), 0.05, 0.95))
        self._stress_level = 0.88 * self._stress_level + 0.12 * raw_stress

        # ── 3. Store History ──
        self._focus_history.append(float(self._focus_score))
        self._stress_history.append(float(self._stress_level))
        self._fatigue_history.append(float(frame_result.fatigue_level))
        self._emotion_history.append(frame_result.emotion)

        # ── 4. Generate Alerts ──
        self._active_alerts = self._check_alerts(frame_result, now)

        # ── 5. Hackathon Features (Shoulder Surfing & AWA) ──
        shoulder_surfing = getattr(frame_result, 'num_faces', 0) > 1

        # AWA check (if stress or fatigue > 0.80 for recent frames)
        awa_trigger = False
        if len(self._stress_history) >= 30 and len(self._fatigue_history) >= 30:
            avg_recent_stress = sum(list(self._stress_history)[-30:]) / 30.0
            avg_recent_fatigue = sum(list(self._fatigue_history)[-30:]) / 30.0
            if avg_recent_stress > 0.80 or avg_recent_fatigue > 0.80:
                awa_trigger = True

        return {
            "focus_score": float(self._focus_score),
            "stress_level": float(self._stress_level),
            "alerts": self._active_alerts,
            "shoulder_surfing": shoulder_surfing,
            "awa_trigger": awa_trigger,
            "trends": {
                "focus": list(self._focus_history),
                "stress": list(self._stress_history),
                "fatigue": list(self._fatigue_history),
            },
        }

    def _check_alerts(self, result, now: float) -> list:
        """Check for wellness alerts based on current state."""
        alerts = []

        # Fatigue / Drowsiness warning
        if result.fatigue_level > FATIGUE_WARNING_LEVEL or getattr(result, 'is_drowsy', False):
            alerts.append({
                "type": "fatigue",
                "severity": "warning",
                "icon": "😴",
                "title": "Fatigue Detected",
                "message": "Drowsiness or fatigue detected. Take a short 5-minute break.",
                "action": "Take a 5-minute break",
            })

        # Stress warning
        if self._stress_level > STRESS_WARNING_LEVEL:
            alerts.append({
                "type": "stress",
                "severity": "warning",
                "icon": "😰",
                "title": "Elevated Stress",
                "message": "Facial tension and stress elevated. Try a 2-minute breathing exercise.",
                "action": "Start breathing exercise",
            })

        # Break reminder
        if self._continuous_focus_start:
            focus_duration = (now - self._continuous_focus_start) / 60.0
            if focus_duration > BREAK_REMINDER_MINUTES:
                if now - self._last_break_reminder > 300:
                    alerts.append({
                        "type": "break",
                        "severity": "info",
                        "icon": "🧘",
                        "title": "Break Time",
                        "message": f"You've been focused for {int(focus_duration)} minutes. Time to stretch!",
                        "action": "Take a stretch break",
                    })
                    self._last_break_reminder = now

        # Deep focus mode indicator
        if self._focus_score > HIGH_FOCUS_DND_LEVEL:
            alerts.append({
                "type": "deep_focus",
                "severity": "info",
                "icon": "🎯",
                "title": "Deep Focus Mode",
                "message": "You're in deep focus. Notifications paused.",
                "action": None,
            })

        # Looking Away / Distraction reminder
        if self._distraction_start and (now - self._distraction_start) > 4:
            alerts.append({
                "type": "distraction",
                "severity": "info",
                "icon": "👀",
                "title": "Looking Away",
                "message": "You've turned away from the screen. Refocus when ready.",
                "action": "Refocus",
            })

        return alerts

    def get_session_summary(self) -> dict:
        """Get summary of the current session."""
        return {
            "avg_focus": float(np.mean(self._focus_history)) if self._focus_history else 0.5,
            "avg_stress": float(np.mean(self._stress_history)) if self._stress_history else 0.14,
            "avg_fatigue": float(np.mean(self._fatigue_history)) if self._fatigue_history else 0.1,
            "dominant_emotion": self._get_dominant_emotion(),
            "data_points": len(self._focus_history),
        }

    def _get_dominant_emotion(self) -> str:
        """Get the most frequent emotion in recent history."""
        if not self._emotion_history:
            return "neutral"
        from collections import Counter
        counter = Counter(self._emotion_history)
        return counter.most_common(1)[0][0]
