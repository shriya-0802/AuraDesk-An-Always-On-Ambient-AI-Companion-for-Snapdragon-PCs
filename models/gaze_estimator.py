"""
Gaze Estimator Module
Estimates gaze direction using iris landmarks + 3D head pose from MediaPipe Face Mesh.
Determines whether user is looking at the screen vs looking away / turned away / looking down.
"""
import numpy as np
from config.settings import GAZE_SMOOTHING_FACTOR


class GazeEstimator:
    """Estimates where the user is looking based on iris position and head orientation."""

    # Iris landmarks
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    # Eye corner landmarks
    LEFT_EYE_INNER = 362
    LEFT_EYE_OUTER = 263
    LEFT_EYE_TOP = 386
    LEFT_EYE_BOTTOM = 374

    RIGHT_EYE_INNER = 133
    RIGHT_EYE_OUTER = 33
    RIGHT_EYE_TOP = 159
    RIGHT_EYE_BOTTOM = 145

    # Head reference landmarks
    NOSE_TIP = 1
    FOREHEAD = 10
    CHIN = 152
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454

    def __init__(self):
        self._smoothed_gaze = np.array([0.5, 0.5], dtype=np.float32)
        self._calibration_offset = np.array([0.0, 0.0], dtype=np.float32)

    def estimate(self, landmarks: np.ndarray) -> dict:
        """
        Estimate gaze direction and determine if looking at screen.

        Args:
            landmarks: np.ndarray (478, 3) normalized landmarks in [0, 1]

        Returns:
            dict with:
                - 'gaze_point': tuple (x, y) normalized [0, 1] screen position
                - 'gaze_vector': tuple (dx, dy)
                - 'looking_at_screen': bool
                - 'head_yaw': float
                - 'head_pitch': float
        """
        if landmarks is None or len(landmarks) < 478:
            return self._default_result()

        # ── 1. Head Pose (Yaw and Pitch) ──
        nose = landmarks[self.NOSE_TIP, :2]
        left_cheek = landmarks[self.LEFT_CHEEK, :2]
        right_cheek = landmarks[self.RIGHT_CHEEK, :2]
        forehead = landmarks[self.FOREHEAD, :2]
        chin = landmarks[self.CHIN, :2]

        # Head Yaw (horizontal turning)
        dist_left = np.linalg.norm(nose - left_cheek)
        dist_right = np.linalg.norm(nose - right_cheek)
        total_span = max(dist_left + dist_right, 1e-6)
        # yaw: 0 = centered, >0 = turned right (user's right), <0 = turned left
        head_yaw = float((dist_left - dist_right) / total_span)

        # Head Pitch (vertical tilting / drooping)
        upper_h = max(1e-6, np.linalg.norm(nose - forehead))
        lower_h = max(1e-6, np.linalg.norm(chin - nose))
        pitch_ratio = float(upper_h / lower_h)
        # pitch offset from normal ~0.95
        head_pitch = float(pitch_ratio - 0.95)

        # ── 2. Iris Position (Eye Gaze) ──
        left_ratio = self._compute_iris_ratio(
            landmarks, self.LEFT_IRIS,
            self.LEFT_EYE_INNER, self.LEFT_EYE_OUTER,
            self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM
        )
        right_ratio = self._compute_iris_ratio(
            landmarks, self.RIGHT_IRIS,
            self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER,
            self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM
        )
        avg_iris = (left_ratio + right_ratio) / 2.0

        # ── 3. Combine Head Pose + Iris for True Screen Gaze ──
        # Iris displacement from center 0.5
        iris_dx = float(avg_iris[0] - 0.5)
        iris_dy = float(avg_iris[1] - 0.5)

        # Gaze point on screen
        raw_gaze_x = 0.5 + head_yaw * 1.6 + iris_dx * 1.2 + self._calibration_offset[0]
        raw_gaze_y = 0.5 + head_pitch * 1.4 + iris_dy * 1.2 + self._calibration_offset[1]

        gaze_x = float(np.clip(raw_gaze_x, 0.05, 0.95))
        gaze_y = float(np.clip(raw_gaze_y, 0.05, 0.95))

        raw_gaze = np.array([gaze_x, gaze_y], dtype=np.float32)
        self._smoothed_gaze = (
            GAZE_SMOOTHING_FACTOR * raw_gaze
            + (1.0 - GAZE_SMOOTHING_FACTOR) * self._smoothed_gaze
        )

        gaze_vector = self._smoothed_gaze - np.array([0.5, 0.5])

        # ── 4. Determine if user is looking at the screen ──
        # Looking away if head is turned > ~25 degrees, head is tilted down/up excessively, or eyes are averted
        # TFLite model produces yaw values 0.10-0.35 even when looking straight ahead
        head_aligned = abs(head_yaw) <= 0.38
        pitch_aligned = (-0.50 <= head_pitch <= 0.50)
        iris_aligned = (0.20 <= avg_iris[0] <= 0.80)

        looking_at_screen = bool(head_aligned and pitch_aligned and iris_aligned)

        return {
            "gaze_point": (float(self._smoothed_gaze[0]), float(self._smoothed_gaze[1])),
            "gaze_vector": (float(gaze_vector[0]), float(gaze_vector[1])),
            "looking_at_screen": looking_at_screen,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "left_iris_ratio": (float(left_ratio[0]), float(left_ratio[1])),
            "right_iris_ratio": (float(right_ratio[0]), float(right_ratio[1])),
        }

    def _compute_iris_ratio(
        self, lm: np.ndarray, iris_indices: list,
        inner: int, outer: int, top: int, bottom: int
    ) -> np.ndarray:
        """Compute iris center position ratio within eye boundary."""
        iris_center = np.mean(lm[iris_indices, :2], axis=0)

        eye_inner = lm[inner, :2]
        eye_outer = lm[outer, :2]
        eye_top = lm[top, :2]
        eye_bottom = lm[bottom, :2]

        eye_width = np.linalg.norm(eye_outer - eye_inner)
        if eye_width < 1e-6:
            x_ratio = 0.5
        else:
            eye_h_vec = eye_outer - eye_inner
            iris_vec = iris_center - eye_inner
            x_ratio = np.dot(iris_vec, eye_h_vec) / (eye_width ** 2)

        eye_height = np.linalg.norm(eye_bottom - eye_top)
        if eye_height < 1e-6:
            y_ratio = 0.5
        else:
            eye_v_vec = eye_bottom - eye_top
            iris_vec = iris_center - eye_top
            y_ratio = np.dot(iris_vec, eye_v_vec) / (eye_height ** 2)

        return np.array([np.clip(x_ratio, 0.0, 1.0), np.clip(y_ratio, 0.0, 1.0)])

    def calibrate(self, landmarks: np.ndarray):
        """Calibrate gaze offset to center."""
        result = self.estimate(landmarks)
        current = np.array(result["gaze_point"])
        self._calibration_offset = np.array([0.5, 0.5]) - current

    def _default_result(self):
        return {
            "gaze_point": (0.5, 0.5),
            "gaze_vector": (0.0, 0.0),
            "looking_at_screen": True,
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "left_iris_ratio": (0.5, 0.5),
            "right_iris_ratio": (0.5, 0.5),
        }
