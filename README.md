# AuraDesk
**On-Device Cognitive Computing Environment for Snapdragon PCs**

AuraDesk is a real-time, privacy-first AI desktop companion that runs entirely on the Snapdragon NPU. It uses your laptop's webcam to understand your cognitive state (emotion, fatigue, focus) and gestures, adapting your computing environment in real-time.

## Features

*   **Cognitive State Monitor:** Real-time emotion, fatigue, and stress detection via webcam.
*   **Gesture Command System:** Hands-free PC control through mid-air gestures.
*   **Gaze-Aware Smart Focus:** Eye tracking with attention heatmaps.
*   **Adaptive Wellness Engine:** Auto-adjusts environment based on your state (e.g., break reminders, deep focus mode).
*   **Privacy-First:** All inference runs on-device. No cloud dependencies.

## Judging Instructions (How to test)

To run the full AuraDesk experience on your machine:

### 1. Setup & Installation
Ensure you have Python 3.11+ installed. Open your terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### 2. Launching AuraDesk
Run the application using the startup script:
```bash
./run_auradesk.sh
```
*Note: If the script fails, you can run `python main.py` directly.*

### 3. Testing the AI Features
Once the desktop dashboard opens, try the following features:
* **Gesture Control:** Hold your hand up to the camera and try these gestures:
  * ✊ **Fist:** Takes a screenshot of your screen.
  * 🖐️ **Open Palm:** Plays/Pauses your media.
  * ☝️ **Point Up:** Increases system volume by 10%.
  * 👍 **Thumbs Up:** Triggers a system confirmation (used for accepting AI suggestions).
* **Privacy Guardian:** Have a second person step into the camera view. The app will detect "Shoulder Surfing" and instantly lock your screen with a privacy shield.
* **Cognitive Telemetry:** Watch the Stress and Fatigue bars update in real-time based on your blinking, yawning, and facial tension.

## Deployment (Snapdragon)

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
    ```
4.  Ensure `onnxruntime-qnn` is installed.

## Architecture

AuraDesk runs a multi-model AI pipeline:
1.  **Face Detection & Landmarks:** MediaPipe Face Mesh (Fallback) / ONNX Model.
2.  **Emotion Recognition:** Geometric analysis of facial landmarks.
3.  **Fatigue Detection:** Blink rate (EAR) and Yawn detection (MAR).
4.  **Hand Tracking:** MediaPipe Hands for gesture classification.

The `ContextFusion` engine combines these signals, and the `ActionEngine` executes system commands.
