"""
Face Detector Module
Uses OpenCV YuNet ONNX face detector + MediaPipe Face Landmark TFLite model via LiteRT.
Extracts 478 3D facial landmarks, bounding boxes, and iris centers.
"""
import os
import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
    LITERT_AVAILABLE = True
except ImportError:
    Interpreter = None
    LITERT_AVAILABLE = False

from config.settings import (
    FACE_DETECTION_CONFIDENCE,
    FACE_MESH_CONFIDENCE,
    USE_NPU,
)


class FaceDetector:
    """Detects faces and extracts 478 3D facial landmarks."""

    # Key landmark indices for various computations
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
    LEFT_EYEBROW = [276, 283, 282, 295, 300]
    RIGHT_EYEBROW = [46, 53, 52, 65, 70]
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    def __init__(self):
        self.yunet_detector = None
        self.lm_interp = None
        self._demo_mode = "live"
        self._frame_count = 0
        self._smoothed_bbox = None
        self._init_models()

    def _init_models(self):
        """Initialize YuNet face detector and LiteRT face landmark model."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yunet_path = os.path.join(base_dir, "onnx_models", "face_detection_yunet.onnx")
        landmark_path = os.path.join(base_dir, "tflite_models", "face_landmarks_detector.tflite")

        # 1. Initialize YuNet
        if os.path.exists(yunet_path):
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model=yunet_path,
                    config="",
                    input_size=(640, 480),
                    score_threshold=0.35,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                print("[FaceDetector] YuNet Face Detector loaded successfully.")
            except Exception as e:
                print(f"[FaceDetector] YuNet init error: {e}")

        # 2. Initialize LiteRT Face Landmarks
        if LITERT_AVAILABLE and os.path.exists(landmark_path):
            try:
                self.lm_interp = Interpreter(model_path=landmark_path)
                self.lm_interp.allocate_tensors()
                self.lm_in = self.lm_interp.get_input_details()[0]
                self.lm_out = self.lm_interp.get_output_details()
                print("[FaceDetector] LiteRT Face Landmark Detector (478 pts) loaded.")
            except Exception as e:
                print(f"[FaceDetector] LiteRT Face Landmark init error: {e}")

    def detect(self, frame_rgb: np.ndarray) -> dict:
        """
        Detect face and extract 478 3D landmarks.

        Args:
            frame_rgb: RGB image as numpy array (H, W, 3)

        Returns:
            dict with:
                - 'detected': bool
                - 'landmarks': np.ndarray of shape (478, 3) or None
                - 'bbox': tuple (x1, y1, x2, y2) normalized [0, 1] or None
        """
        # If in manual simulation demo mode (and not 'live')
        if self._demo_mode != "live":
            landmarks = self._generate_simulated_landmarks()
            return {
                "detected": True,
                "landmarks": landmarks,
                "bbox": (0.28, 0.18, 0.72, 0.75),
            }

        if frame_rgb is None:
            return {"detected": False, "landmarks": None, "bbox": None}

        h, w = frame_rgb.shape[:2]

        # Use YuNet for face detection
        if self.yunet_detector is not None:
            self.yunet_detector.setInputSize((w, h))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            _, faces = self.yunet_detector.detect(frame_bgr)

            if faces is not None and len(faces) > 0:
                # Pick largest face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                face = faces[0]
                fx, fy, fw, fh = face[:4].astype(int)
                conf = float(face[-1])

                # Normalized bounding box with margin
                margin_x = int(fw * 0.15)
                margin_y = int(fh * 0.15)
                bx1 = max(0.0, float(fx - margin_x) / w)
                by1 = max(0.0, float(fy - margin_y) / h)
                bx2 = min(1.0, float(fx + fw + margin_x) / w)
                by2 = min(1.0, float(fy + fh + margin_y) / h)
                raw_bbox = (bx1, by1, bx2, by2)

                # Smooth bounding box
                if self._smoothed_bbox is None:
                    self._smoothed_bbox = raw_bbox
                else:
                    self._smoothed_bbox = (
                        0.7 * self._smoothed_bbox[0] + 0.3 * raw_bbox[0],
                        0.7 * self._smoothed_bbox[1] + 0.3 * raw_bbox[1],
                        0.7 * self._smoothed_bbox[2] + 0.3 * raw_bbox[2],
                        0.7 * self._smoothed_bbox[3] + 0.3 * raw_bbox[3],
                    )

                # Square crop for landmark extraction
                cx = fx + fw / 2.0
                cy = fy + fh / 2.0
                crop_size = max(fw, fh) * 1.5
                x1 = max(0, int(cx - crop_size / 2.0))
                y1 = max(0, int(cy - crop_size / 2.0))
                x2 = min(w, int(cx + crop_size / 2.0))
                y2 = min(h, int(cy + crop_size / 2.0))

                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10 and self.lm_interp is not None:
                    crop_h, crop_w = crop.shape[:2]
                    crop_resized = cv2.resize(crop, (256, 256)).astype(np.float32) / 255.0
                    crop_input = np.expand_dims(crop_resized, axis=0)

                    self.lm_interp.set_tensor(self.lm_in["index"], crop_input)
                    self.lm_interp.invoke()

                    raw_lm = self.lm_interp.get_tensor(self.lm_out[0]["index"]).reshape(-1, 3)

                    # Map landmarks back to full frame normalized coords [0, 1]
                    lm = np.zeros_like(raw_lm)
                    lm[:, 0] = (x1 + (raw_lm[:, 0] / 256.0) * crop_w) / float(w)
                    lm[:, 1] = (y1 + (raw_lm[:, 1] / 256.0) * crop_h) / float(h)
                    lm[:, 2] = raw_lm[:, 2] / 256.0

                    return {
                        "detected": True,
                        "landmarks": lm,
                        "bbox": self._smoothed_bbox,
                        "num_faces": len(faces)
                    }

                return {
                    "detected": True,
                    "landmarks": None,
                    "bbox": self._smoothed_bbox,
                    "num_faces": len(faces)
                }

        self._smoothed_bbox = None
        return {"detected": False, "landmarks": None, "bbox": None, "num_faces": 0}

    def _generate_simulated_landmarks(self) -> np.ndarray:
        """Generate realistic facial landmarks for quick testing/demo mode."""
        self._frame_count = getattr(self, "_frame_count", 0) + 1
        t = self._frame_count * 0.05

        cx = 0.50 + np.sin(t * 0.5) * 0.002
        cy = 0.45 + np.cos(t * 0.7) * 0.002

        lm = np.zeros((478, 3), dtype=np.float32)
        demo_mode = self._demo_mode

        is_blinking = (self._frame_count % 90 in (0, 1, 2)) or (demo_mode == "blink")
        is_yawning = (demo_mode == "yawn") or (self._frame_count % 350 in range(200, 220))
        is_stressed = (demo_mode == "stressed")
        is_looking_away = (demo_mode == "looking_away")
        is_focused = (demo_mode in ("focused", "normal")) and not is_stressed and not is_looking_away

        # Face contour
        lm[10] = [cx, cy - 0.22, 0]
        lm[152] = [cx, cy + 0.22, 0]
        lm[234] = [cx - 0.18, cy, 0]
        lm[454] = [cx + 0.18, cy, 0]
        lm[1] = [cx, cy - 0.02, -0.05]

        brow_y = cy - (0.08 if is_stressed else 0.12)
        brow_inner_dist = 0.02 if is_stressed else 0.04
        lm[55] = [cx - brow_inner_dist, brow_y, 0]
        lm[285] = [cx + brow_inner_dist, brow_y, 0]
        lm[46] = [cx - 0.14, brow_y - 0.01, 0]
        lm[53] = [cx - 0.11, brow_y - 0.02, 0]
        lm[52] = [cx - 0.08, brow_y - 0.02, 0]
        lm[65] = [cx - 0.05, brow_y - 0.01, 0]
        lm[70] = [cx - brow_inner_dist, brow_y, 0]

        lm[276] = [cx + 0.14, brow_y - 0.01, 0]
        lm[283] = [cx + 0.11, brow_y - 0.02, 0]
        lm[282] = [cx + 0.08, brow_y - 0.02, 0]
        lm[295] = [cx + 0.05, brow_y - 0.01, 0]
        lm[300] = [cx + brow_inner_dist, brow_y, 0]

        eye_y = cy - 0.06
        eye_open_h = 0.002 if is_blinking else (0.008 if is_focused else 0.010)

        lm[362] = [cx + 0.03, eye_y, 0]
        lm[263] = [cx + 0.11, eye_y, 0]
        lm[386] = [cx + 0.07, eye_y - eye_open_h, 0]
        lm[374] = [cx + 0.07, eye_y + eye_open_h, 0]
        lm[385] = [cx + 0.05, eye_y - eye_open_h, 0]
        lm[387] = [cx + 0.09, eye_y - eye_open_h, 0]
        lm[373] = [cx + 0.09, eye_y + eye_open_h, 0]
        lm[380] = [cx + 0.05, eye_y + eye_open_h, 0]

        lm[133] = [cx - 0.03, eye_y, 0]
        lm[33] = [cx - 0.11, eye_y, 0]
        lm[159] = [cx - 0.07, eye_y - eye_open_h, 0]
        lm[145] = [cx - 0.07, eye_y + eye_open_h, 0]
        lm[160] = [cx - 0.09, eye_y - eye_open_h, 0]
        lm[158] = [cx - 0.05, eye_y - eye_open_h, 0]
        lm[144] = [cx - 0.05, eye_y + eye_open_h, 0]
        lm[153] = [cx - 0.09, eye_y + eye_open_h, 0]

        gaze_offset_x = 0.12 if is_looking_away else (np.sin(t * 0.2) * 0.001)
        gaze_offset_y = 0.08 if is_looking_away else (np.cos(t * 0.3) * 0.001)

        for idx in self.LEFT_IRIS:
            lm[idx] = [cx + 0.07 + gaze_offset_x, eye_y + gaze_offset_y, 0]
        for idx in self.RIGHT_IRIS:
            lm[idx] = [cx - 0.07 + gaze_offset_x, eye_y + gaze_offset_y, 0]

        mouth_top_y = cy + 0.06
        mouth_bot_y = cy + (0.17 if is_yawning else 0.068)
        mouth_w = 0.07 + (0.01 if is_focused else 0.0)

        lm[13] = [cx, mouth_top_y, 0]
        lm[14] = [cx, mouth_bot_y, 0]
        lm[61] = [cx - mouth_w, mouth_top_y + 0.005, 0]
        lm[291] = [cx + mouth_w, mouth_top_y + 0.005, 0]

        for i in range(478):
            if np.all(lm[i] == 0):
                angle = (i / 478.0) * 2 * np.pi
                r = 0.13 + 0.03 * np.cos(3 * angle)
                lm[i] = [cx + r * np.cos(angle), cy + r * np.sin(angle), 0.0]

        return lm

    def release(self):
        """Clean up resources."""
        self.yunet_detector = None
        self.lm_interp = None
