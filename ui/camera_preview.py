"""
Camera Preview Widget
Displays the live webcam feed with AI overlay annotations:
  - Face bounding box with corner accents
  - Facial landmarks & contour
  - Clean emotion label & confidence badge
  - Hand skeleton (only when real hand is in view)
  - Iris gaze dots & FPS/Blink/Yawn/Drowsy status
"""
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class CameraPreview(QWidget):
    """Widget displaying annotated camera feed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._current_result = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setObjectName("cameraView")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(480, 360)
        self.image_label.setScaledContents(False)
        layout.addWidget(self.image_label)

    def update_frame(self, result):
        """Update display with annotated frame."""
        self._current_result = result
        if result.frame_rgb is None:
            return

        frame = result.frame_rgb.copy()
        h, w = frame.shape[:2]

        # Draw AI overlays
        self._draw_face_overlay(frame, result, w, h)
        self._draw_hand_overlay(frame, result, w, h)
        self._draw_gaze_overlay(frame, result, w, h)
        self._draw_info_overlay(frame, result, w, h)

        # Convert to QPixmap
        pixmap = self._frame_to_pixmap(frame)

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _draw_face_overlay(self, frame, result, w, h):
        """Draw face bounding box, contour, and emotion label."""
        if not result.face_detected or result.face_bbox is None:
            return

        bbox = result.face_bbox
        x1 = max(0, int(bbox[0] * w))
        y1 = max(0, int(bbox[1] * h))
        x2 = min(w, int(bbox[2] * w))
        y2 = min(h, int(bbox[3] * h))

        # Cyan bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 212, 255), 2)

        # Sleek corner accents
        corner_len = max(10, min(22, (x2 - x1) // 4, (y2 - y1) // 4))
        corner_color = (0, 245, 255)
        t = 3
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), corner_color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), corner_color, t)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), corner_color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), corner_color, t)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), corner_color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), corner_color, t)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), corner_color, t)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), corner_color, t)

        # Emotion label (clean ASCII text)
        label = f"{result.emotion.upper()} {int(result.emotion_confidence * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        label_x = x1
        label_y = y1 - 8
        if label_y - th < 0:
            label_y = y2 + th + 10

        # Background pill
        cv2.rectangle(frame,
                      (label_x - 4, label_y - th - 5),
                      (label_x + tw + 6, label_y + 4),
                      (12, 12, 28), -1)
        cv2.rectangle(frame,
                      (label_x - 4, label_y - th - 5),
                      (label_x + tw + 6, label_y + 4),
                      (0, 212, 255), 1)
        cv2.putText(frame, label, (label_x, label_y),
                    font, font_scale, (255, 255, 255), thickness)

        # Subtle facial landmark dots
        if result.face_landmarks is not None:
            contour_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454,
                323, 361, 288, 397, 365, 379, 378, 400,
                377, 152, 148, 176, 149, 150, 136, 172,
                58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                33, 133, 159, 145, 362, 263, 386, 374,
                61, 291, 13, 14,
            ]
            for idx in contour_indices:
                if idx < len(result.face_landmarks):
                    px = int(result.face_landmarks[idx][0] * w)
                    py = int(result.face_landmarks[idx][1] * h)
                    cv2.circle(frame, (px, py), 1, (0, 212, 255), -1)

    def _draw_hand_overlay(self, frame, result, w, h):
        """Draw hand skeleton ONLY when real hands are detected."""
        if not result.hands_detected or not result.hands_data:
            return

        CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        for hand in result.hands_data:
            lm = hand.get("landmarks")
            if lm is None or len(lm) < 21:
                continue

            for start, end in CONNECTIONS:
                p1 = (int(lm[start][0] * w), int(lm[start][1] * h))
                p2 = (int(lm[end][0] * w), int(lm[end][1] * h))
                cv2.line(frame, p1, p2, (57, 255, 20), 2)

            for i, pt in enumerate(lm):
                px, py = int(pt[0] * w), int(pt[1] * h)
                if i in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (px, py), 5, (57, 255, 20), -1)
                else:
                    cv2.circle(frame, (px, py), 2, (200, 255, 200), -1)

            gesture_label = hand.get("gesture_label", "")
            if gesture_label:
                wrist_x = int(lm[0][0] * w)
                wrist_y = int(lm[0][1] * h) + 25
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(gesture_label, font, 0.6, 2)
                cv2.rectangle(frame,
                              (wrist_x - 4, wrist_y - th - 6),
                              (wrist_x + tw + 4, wrist_y + 4),
                              (10, 25, 10), -1)
                cv2.rectangle(frame,
                              (wrist_x - 4, wrist_y - th - 6),
                              (wrist_x + tw + 4, wrist_y + 4),
                              (57, 255, 20), 1)
                cv2.putText(frame, gesture_label, (wrist_x, wrist_y),
                            font, 0.6, (57, 255, 20), 2)

    def _draw_gaze_overlay(self, frame, result, w, h):
        """Draw iris dots and gaze direction."""
        if not result.face_detected or result.face_landmarks is None:
            return

        lm = result.face_landmarks
        if len(lm) >= 478:
            left_iris = np.mean(lm[[474, 475, 476, 477], :2], axis=0)
            right_iris = np.mean(lm[[469, 470, 471, 472], :2], axis=0)

            for iris in [left_iris, right_iris]:
                cx, cy = int(iris[0] * w), int(iris[1] * h)
                cv2.circle(frame, (cx, cy), 3, (168, 85, 247), -1)

    def _draw_info_overlay(self, frame, result, w, h):
        """Draw FPS and live state badges."""
        fps_text = f"FPS: {int(result.fps)}"
        cv2.putText(frame, fps_text, (w - 85, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

        y_offset = 50
        if result.is_blinking:
            cv2.putText(frame, "BLINK", (w - 85, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 190, 11), 1)
            y_offset += 25

        if getattr(result, 'is_drowsy', False):
            cv2.putText(frame, "DROWSY", (w - 95, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 68, 68), 1)
            y_offset += 25

        if getattr(result, 'is_yawning', False):
            cv2.putText(frame, "YAWN", (w - 85, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 68, 68), 1)
            y_offset += 25

        if not result.looking_at_screen and result.face_detected:
            cv2.putText(frame, "AWAY", (w - 85, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 120, 255), 1)

    def _frame_to_pixmap(self, frame_rgb: np.ndarray) -> QPixmap:
        """Convert RGB numpy array to QPixmap."""
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)
