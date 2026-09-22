"""
Emotion Classifier Module
Estimates emotions from 478 face mesh landmarks using geometric feature analysis.
Computes smile ratio, brow tension/furrowing, eye openness, and mouth geometry.
"""
import numpy as np
from config.settings import EMOTION_EMOJIS


class EmotionClassifier:
    """Classifies facial emotions from face mesh landmarks."""

    EMOTIONS = ["happy", "sad", "angry", "surprised", "neutral"]

    def __init__(self):
        self._prev_emotion = "neutral"
        self._smoothed_scores = {e: 0.20 for e in self.EMOTIONS}
        self._smoothed_scores["neutral"] = 0.60

    def classify(self, landmarks: np.ndarray) -> dict:
        """
        Classify facial emotion from face mesh landmarks.

        Args:
            landmarks: np.ndarray of shape (478, 3), normalized [0, 1]

        Returns:
            dict with:
                - 'emotion': str
                - 'confidence': float [0, 1]
                - 'emoji': str
                - 'scores': dict of emotion -> score
        """
        if landmarks is None or len(landmarks) < 468:
            return self._default_result()

        raw_scores = self._compute_emotion_scores(landmarks)

        # Smooth scores with exponential moving average
        for e in self.EMOTIONS:
            self._smoothed_scores[e] = (
                0.65 * self._smoothed_scores.get(e, 0.0) + 0.35 * raw_scores.get(e, 0.0)
            )

        # Normalize smoothed scores
        total = sum(self._smoothed_scores.values())
        if total > 0:
            norm_scores = {k: v / total for k, v in self._smoothed_scores.items()}
        else:
            norm_scores = self._smoothed_scores

        dominant_emotion = max(norm_scores, key=norm_scores.get)
        confidence = float(norm_scores[dominant_emotion])

        if confidence < 0.28:
            dominant_emotion = "neutral"
            confidence = max(0.50, float(norm_scores.get("neutral", 0.50)))

        emoji = EMOTION_EMOJIS.get(dominant_emotion, "😐")

        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "emoji": emoji,
            "scores": norm_scores,
        }

    def _compute_emotion_scores(self, lm: np.ndarray) -> dict:
        """Compute emotion scores from landmark geometry."""
        scores = {}

        face_width = max(1e-6, np.linalg.norm(lm[234, :2] - lm[454, :2]))
        eye_dist = max(1e-6, np.linalg.norm(lm[33, :2] - lm[263, :2]))

        # ── 1. Happy (Smile) ──
        mouth_width = np.linalg.norm(lm[61, :2] - lm[291, :2])
        mouth_ratio = mouth_width / face_width

        mouth_center_y = (lm[13, 1] + lm[14, 1]) / 2.0
        corner_y = (lm[61, 1] + lm[291, 1]) / 2.0
        corner_lift = (mouth_center_y - corner_y) / face_width

        smile_score = np.clip((mouth_ratio - 0.38) * 3.5 + corner_lift * 12.0, 0.0, 1.0)
        scores["happy"] = float(smile_score)

        # ── 2. Surprised ──
        left_ear = self._eye_openness(lm, [362, 385, 387, 263, 373, 380])
        right_ear = self._eye_openness(lm, [33, 160, 158, 133, 153, 144])
        eye_openness = (left_ear + right_ear) / 2.0

        mouth_height = np.linalg.norm(lm[13, :2] - lm[14, :2])
        mouth_open_ratio = mouth_height / face_width

        surprise_score = np.clip(
            (eye_openness - 0.48) * 4.0 + (mouth_open_ratio - 0.12) * 5.0, 0.0, 1.0
        )
        scores["surprised"] = float(surprise_score)

        # ── 3. Angry / Tense ──
        brow_inner_dist = np.linalg.norm(lm[55, :2] - lm[285, :2])
        brow_furrow = brow_inner_dist / eye_dist

        left_brow_y = np.mean(lm[[276, 283, 282, 295, 300], 1])
        right_brow_y = np.mean(lm[[46, 53, 52, 65, 70], 1])
        left_eye_y = np.mean(lm[[362, 263], 1])
        right_eye_y = np.mean(lm[[33, 133], 1])
        brow_eye_dist = ((left_eye_y - left_brow_y) + (right_eye_y - right_brow_y)) / (2.0 * face_width)

        anger_score = np.clip(
            (0.30 - brow_furrow) * 3.5 + (0.10 - brow_eye_dist) * 8.0, 0.0, 1.0
        )
        scores["angry"] = float(anger_score)

        # ── 4. Sadness ──
        corner_drop = (corner_y - mouth_center_y) / face_width
        sad_score = np.clip(corner_drop * 10.0, 0.0, 1.0)
        scores["sad"] = float(sad_score)

        # ── 5. Neutral ──
        max_expressive = max(scores["happy"], scores["surprised"], scores["angry"], scores["sad"])
        scores["neutral"] = float(np.clip(1.0 - max_expressive * 1.6, 0.20, 0.90))

        return scores

    def _eye_openness(self, lm: np.ndarray, indices: list) -> float:
        """Compute Eye Aspect Ratio."""
        p = lm[indices, :2]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return float((v1 + v2) / (2.0 * max(h, 1e-6)))

    def _default_result(self):
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "emoji": "😐",
            "scores": {e: 0.20 for e in self.EMOTIONS},
        }
