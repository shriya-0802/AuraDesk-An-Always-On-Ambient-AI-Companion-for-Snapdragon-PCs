"""
Hand Tracker Module
Detects hands and classifies gestures using LiteRT Palm & Hand Landmark models.
Extracts 21 3D hand landmarks and classifies gestures:
  - ✊ fist (Screenshot)
  - 🖐️ open palm (Media Play/Pause)
  - 👆 point up (Volume Up)
  - 👇 point down (Volume Down)
  - ✌️ peace (Switch Desktop)
  - 👍 thumbs up (Confirm)
"""
import os
import time
import cv2
import numpy as np
from collections import deque

try:
    from ai_edge_litert.interpreter import Interpreter
    LITERT_AVAILABLE = True
except ImportError:
    Interpreter = None
    LITERT_AVAILABLE = False

from config.settings import GESTURE_COOLDOWN_MS, GESTURES


class HandTracker:
    """Tracks hands and classifies gestures from landmark positions."""

    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    def __init__(self):
        self.palm_interp = None
        self.hand_lm_interp = None
        self._last_gesture_time = 0.0
        self._last_gesture = "none"
        self._gesture_history = deque(maxlen=5)
        self._miss_count = 0  # Track consecutive misses for forgiving stability
        self._init_models()

    def _init_models(self):
        """Initialize palm and hand landmark models."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        palm_path = os.path.join(base_dir, "tflite_models", "hand_detector.tflite")
        hand_lm_path = os.path.join(base_dir, "tflite_models", "hand_landmarks_detector.tflite")

        if LITERT_AVAILABLE and os.path.exists(palm_path) and os.path.exists(hand_lm_path):
            try:
                self.palm_interp = Interpreter(model_path=palm_path)
                self.palm_interp.allocate_tensors()
                self.palm_in = self.palm_interp.get_input_details()[0]
                self.palm_out = self.palm_interp.get_output_details()

                self.hand_lm_interp = Interpreter(model_path=hand_lm_path)
                self.hand_lm_interp.allocate_tensors()
                self.hand_lm_in = self.hand_lm_interp.get_input_details()[0]
                self.hand_lm_out = self.hand_lm_interp.get_output_details()

                self.anchors = self._generate_anchors()
                print("[HandTracker] LiteRT Hand Detector & Landmark models loaded.")
            except Exception as e:
                print(f"[HandTracker] Model load error: {e}")

    def _generate_anchors(self):
        """Generate SSD anchors for 192x192 palm detector."""
        anchors = []
        for stride, num_anchors in [(8, 2), (16, 6)]:
            grid_size = 192 // stride
            for y in range(grid_size):
                for x in range(grid_size):
                    cx = (x + 0.5) * stride / 192.0
                    cy = (y + 0.5) * stride / 192.0
                    for _ in range(num_anchors):
                        anchors.append([cx, cy])
        return np.array(anchors, dtype=np.float32)

    def detect(self, frame_rgb: np.ndarray) -> dict:
        """
        Detect hands and classify gestures.

        Args:
            frame_rgb: RGB image (H, W, 3)

        Returns:
            dict with:
                - 'detected': bool
                - 'hands': list of hand dicts
                - 'active_gesture': str
                - 'gesture_triggered': bool
        """
        if self.palm_interp is None or self.hand_lm_interp is None or frame_rgb is None:
            return self._empty_result()

        h, w = frame_rgb.shape[:2]

        # 1. Run Palm Detector (192x192)
        palm_input = cv2.resize(frame_rgb, (192, 192)).astype(np.float32) / 255.0
        palm_input = np.expand_dims(palm_input, axis=0)

        self.palm_interp.set_tensor(self.palm_in["index"], palm_input)
        self.palm_interp.invoke()

        scores_raw = self.palm_interp.get_tensor(self.palm_out[1]["index"]).squeeze()
        boxes_raw = self.palm_interp.get_tensor(self.palm_out[0]["index"]).squeeze()

        # Sigmoid scoring
        scores = 1.0 / (1.0 + np.exp(-scores_raw))
        valid_idx = np.where(scores > 0.25)[0]

        if len(valid_idx) == 0:
            # Allow up to 2 missed frames before clearing history
            self._miss_count += 1
            if self._miss_count > 2:
                self._gesture_history.clear()
            return self._empty_result()

        best_idx = valid_idx[np.argmax(scores[valid_idx])]
        box = boxes_raw[best_idx]
        anchor = self.anchors[best_idx]

        # Decode palm center & dimensions
        cx = box[0] / 192.0 + anchor[0]
        cy = box[1] / 192.0 + anchor[1]
        bw = box[2] / 192.0
        bh = box[3] / 192.0

        palm_size = max(bw, bh) * 2.4
        x1 = max(0, int((cx - palm_size / 2.0) * w))
        y1 = max(0, int((cy - palm_size / 2.0) * h))
        x2 = min(w, int((cx + palm_size / 2.0) * w))
        y2 = min(h, int((cy + palm_size / 2.0) * h))

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return self._empty_result()

        crop_h, crop_w = crop.shape[:2]
        crop_resized = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
        crop_input = np.expand_dims(crop_resized, axis=0)

        # 2. Run Hand Landmarks (224x224)
        self.hand_lm_interp.set_tensor(self.hand_lm_in["index"], crop_input)
        self.hand_lm_interp.invoke()

        raw_lm = self.hand_lm_interp.get_tensor(self.hand_lm_out[0]["index"]).reshape(-1, 3)
        hand_score = float(self.hand_lm_interp.get_tensor(self.hand_lm_out[1]["index"]).ravel()[0])
        handedness_score = float(self.hand_lm_interp.get_tensor(self.hand_lm_out[2]["index"]).ravel()[0])

        if hand_score < 0.20:
            # Allow up to 2 missed frames before clearing history
            self._miss_count += 1
            if self._miss_count > 2:
                self._gesture_history.clear()
            return self._empty_result()

        # Reset miss counter on successful detection
        self._miss_count = 0

        # Transform landmarks to normalized [0, 1] frame coordinates
        lm = np.zeros_like(raw_lm)
        lm[:, 0] = (x1 + (raw_lm[:, 0] / 224.0) * crop_w) / float(w)
        lm[:, 1] = (y1 + (raw_lm[:, 1] / 224.0) * crop_h) / float(h)
        lm[:, 2] = raw_lm[:, 2] / 224.0

        handedness = "Right" if handedness_score > 0.5 else "Left"
        finger_states = self._get_finger_states(lm, handedness)
        gesture = self._classify_gesture(finger_states, lm)
        gesture_label = GESTURES.get(gesture, "")

        hands_data = [{
            "landmarks": lm,
            "handedness": handedness,
            "gesture": gesture,
            "gesture_label": gesture_label,
            "finger_states": finger_states,
        }]

        current_gesture = gesture
        gesture_triggered = False

        if current_gesture != "none":
            self._gesture_history.append(current_gesture)
        else:
            # If no gesture, pad with none to decay history
            self._gesture_history.append("none")

        now = time.time() * 1000
        
        # Trigger logic: check if we have a clear majority in the recent window
        if len(self._gesture_history) == self._gesture_history.maxlen:
            # Count occurrences of each gesture in the history window
            counts = {}
            for g in self._gesture_history:
                counts[g] = counts.get(g, 0) + 1
            
            # Find the most frequent gesture
            most_frequent = max(counts.items(), key=lambda x: x[1])
            best_gesture = most_frequent[0]
            best_count = most_frequent[1]

            # If a valid gesture appears in >= 3 of the last 5 frames, trigger it!
            if best_gesture != "none" and best_count >= 3:
                if (now - self._last_gesture_time) > GESTURE_COOLDOWN_MS:
                    gesture_triggered = True
                    self._last_gesture_time = now
                    # Clear history so we don't immediately trigger again if cooldown is disabled
                    self._gesture_history.clear()
                    
                # Update current gesture to the stable one for UI display
                current_gesture = best_gesture

        return {
            "detected": True,
            "hands": hands_data,
            "active_gesture": current_gesture,
            "gesture_triggered": gesture_triggered,
        }

    def _get_finger_states(self, lm: np.ndarray, handedness: str) -> dict:
        """Determine which fingers are extended."""
        states = {}

        # Thumb: extended if tip is far from index MCP or IP joint
        thumb_tip = lm[self.THUMB_TIP, :2]
        thumb_ip = lm[self.THUMB_IP, :2]
        index_mcp = lm[self.INDEX_MCP, :2]
        wrist = lm[self.WRIST, :2]

        thumb_extended = (
            np.linalg.norm(thumb_tip - index_mcp) > 0.07
            or thumb_tip[1] < thumb_ip[1]
        )
        states["thumb"] = thumb_extended

        # Fingers: tip Y < PIP Y means extended (Y decreases upwards)
        states["index"] = lm[self.INDEX_TIP, 1] < lm[self.INDEX_PIP, 1]
        states["middle"] = lm[self.MIDDLE_TIP, 1] < lm[self.MIDDLE_PIP, 1]
        states["ring"] = lm[self.RING_TIP, 1] < lm[self.RING_PIP, 1]
        states["pinky"] = lm[self.PINKY_TIP, 1] < lm[self.PINKY_PIP, 1]

        return states

    def _classify_gesture(self, fingers: dict, lm: np.ndarray) -> str:
        """Classify gesture from finger states and landmark coordinates."""
        index_ext = fingers["index"]
        middle_ext = fingers["middle"]
        ring_ext = fingers["ring"]
        pinky_ext = fingers["pinky"]
        thumb_ext = fingers["thumb"]

        num_ext = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        # 1. ✊ Fist: 0 main fingers extended
        if num_ext == 0:
            # Check if thumb is pointing straight up (thumbs up) vs folded over (fist)
            if lm[self.THUMB_TIP, 1] < lm[self.INDEX_MCP, 1] and lm[self.THUMB_TIP, 1] < lm[self.WRIST, 1]:
                return "thumbs_up"
            return "fist"

        # 2. 👍 Thumbs up: only thumb extended upward, 4 fingers folded
        if num_ext == 0 or (num_ext == 1 and pinky_ext is False and ring_ext is False and middle_ext is False):
            if thumb_ext and lm[self.THUMB_TIP, 1] < lm[self.WRIST, 1]:
                return "thumbs_up"

        # 3. 🖐️ Open Palm: 3 or 4 fingers extended
        if num_ext >= 3 and index_ext and middle_ext:
            return "open_palm"

        # 4. ✌️ Peace Sign: Index and middle extended, ring and pinky folded
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "peace"

        # 5. 👆 Point Up / 👇 Point Down: only index extended
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if lm[self.INDEX_TIP, 1] < lm[self.INDEX_MCP, 1]:
                return "point_up"
            else:
                return "point_down"

        return "none"

    def _empty_result(self):
        return {
            "detected": False,
            "hands": [],
            "active_gesture": "none",
            "gesture_triggered": False,
        }

    def release(self):
        """Clean up resources."""
        self.palm_interp = None
        self.hand_lm_interp = None
