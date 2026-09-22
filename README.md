# AuraDesk
**On-Device Cognitive Computing Environment for Snapdragon PCs**

AuraDesk is a real-time, privacy-first AI desktop companion that runs entirely on the Snapdragon NPU. It uses your laptop's webcam to understand your cognitive state (emotion, fatigue, focus) and gestures, adapting your computing environment in real-time.

## Features

*   **Cognitive State Monitor:** Real-time emotion, fatigue, and stress detection via webcam.
*   **Gesture Command System:** Hands-free PC control through mid-air gestures (e.g., fist for screenshot, open palm for media toggle).
*   **Gaze-Aware Smart Focus:** Eye tracking with attention heatmaps.
*   **Adaptive Wellness Engine:** Auto-adjusts environment based on your state (e.g., break reminders, deep focus mode).
*   **Privacy-First:** All inference runs on-device. No cloud dependencies.

## Architecture

AuraDesk runs a multi-model AI pipeline:
1.  **Face Detection & Landmarks:** MediaPipe Face Mesh (Fallback) / ONNX Model.
2.  **Emotion Recognition:** Geometric analysis of facial landmarks (No extra model needed).
3.  **Fatigue Detection:** Blink rate (EAR) and Yawn detection (MAR).
4.  **Hand Tracking:** MediaPipe Hands (Fallback) / ONNX Model for gesture classification.
5.  **Gaze Estimation:** Iris tracking mapped to screen coordinates.

The `ContextFusion` engine combines these signals, and the `ActionEngine` executes system commands.

## Setup & Installation

1.  **Prerequisites:** Python 3.11+.
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    python main.py
    ```

## Development & Deployment (Snapdragon)

By default, the application runs using CPU fallback (MediaPipe). To deploy on a Snapdragon-powered HP PC using the Hexagon NPU:

1.  Download the required ONNX models from the **Qualcomm AI Hub**.
2.  Place them in the `onnx_models` directory:
    *   `face_detection.onnx`
    *   `emotion_fer.onnx`
    *   `hand_landmark.onnx`
    *   `gaze_estimation.onnx`
3.  Set the environment variable before running:
    ```bash
    export AURADESK_USE_NPU=1
    # or on Windows:
    set AURADESK_USE_NPU=1
    ```
4.  Ensure `onnxruntime-qnn` is installed.

## Built With

*   **PyQt6:** Premium dark glassmorphism UI.
*   **OpenCV:** Camera capture pipeline.
*   **MediaPipe / ONNX Runtime:** AI inference.
*   **Qualcomm AI Hub:** Target deployment platform for models.
