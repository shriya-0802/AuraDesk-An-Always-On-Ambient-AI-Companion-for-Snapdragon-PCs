# AuraDesk
**On-Device Cognitive Computing Environment for Snapdragon PCs**

AuraDesk is a real-time, privacy-first AI desktop companion that runs entirely on the Snapdragon NPU. It uses your laptop's webcam to understand your cognitive state (emotion, fatigue, focus) and gestures, adapting your computing environment in real-time.

## Features

*   **Cognitive State Monitor:** Real-time emotion, fatigue, and stress detection via webcam.
*   **Gesture Command System:** Hands-free PC control through mid-air gestures.
*   **Gaze-Aware Smart Focus:** Eye tracking with attention heatmaps.
*   **Adaptive Wellness Engine:** Auto-adjusts environment based on your state (e.g., break reminders, deep focus mode).
*   **Privacy-First:** All inference runs on-device. No cloud dependencies.

## 👩‍⚖️ A Quick Guide for Reviewers (How to Test)

Welcome! We are so excited for you to try out AuraDesk. We've built this to be a hands-free, intelligent companion for your PC, and we want you to experience the magic firsthand. Here is a step-by-step guide to get you up and running in just a couple of minutes:

### Step 1: Let's Get Set Up
First, make sure you have Python 3.11 or higher installed on your machine. Open up your favorite terminal, navigate to this project's folder, and run:
```bash
pip install -r requirements.txt
```
*(Grab a quick sip of coffee while the dependencies install!)*

### Step 2: Wake Up AuraDesk
Once everything is installed, it's time to bring your AI companion to life. Just run our startup script:
```bash
./run_auradesk.sh
```
*(Pro tip: If for any reason the script doesn't want to cooperate, you can always just run `python main.py` directly!)*

### Step 3: Play with the Magic (Testing the Features)
As soon as the dashboard pops up, AuraDesk is already quietly analyzing the room. Go ahead and test out these three main pillars of our project:

**1. Become a Jedi (Gesture Controls)**
Make sure your hand is visible to your webcam and try these out:
* ✊ **Make a Fist:** Snap! You just took a screenshot of your screen.
* 🖐️ **Open your Palm:** This will play or pause whatever media you have running.
* ☝️ **Point your Index Finger Up:** Need it louder? This bumps up your system volume by 10%.
* 👍 **Give a Thumbs Up:** This acts as a confirmation to accept any of AuraDesk's AI wellness suggestions.

**2. Test the Privacy Guardian**
This one is fun. Grab a friend (or just hold up a picture of a face on your phone) and bring it into the camera's view over your shoulder. AuraDesk will immediately detect "Shoulder Surfing" and snap a privacy shield over your screen to protect your data. 

**3. Check your Cognitive Telemetry**
Look at the dashboard while you work. Fake a yawn, blink a few times, or furrow your brow. You'll see the **Stress** and **Fatigue** bars reacting to your micro-expressions in real-time. If you look away from the screen, watch how the Gaze-Aware engine tracks your attention!

We hope you have as much fun testing it as we had building it!

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

## 💡 The Inspiration

In today's remote-first world, we spend an average of 8+ hours a day staring at screens. This constant exposure leads to digital fatigue, eye strain, and lost focus. Most AI tools built to solve this require sending your camera feed to the cloud, creating massive privacy concerns. 

We built **AuraDesk** to solve this. By leveraging the power of the Snapdragon NPU, we created an AI that sits quietly on the edge, understanding your cognitive state in real-time without ever sending a single pixel to the cloud. It's not just an application; it's a step towards an empathetic computing environment that genuinely cares about your well-being.

## 🛠️ Tech Stack

*   **Frontend / UI:** PyQt6 (for a blazing fast, cross-platform desktop overlay)
*   **Computer Vision:** OpenCV
*   **AI / ML Pipeline:** MediaPipe (for rapid prototyping & CPU fallback), ONNX Runtime (for NPU acceleration)
*   **System Integration:** PyAutoGUI (for cross-platform system controls)
*   **Core Logic:** Python 3.11+

## 🚀 What's Next (Future Scope)

We have big plans for AuraDesk! Here is what we want to build next:
1.  **Custom Gesture Mapping:** Allow users to map their own custom hand gestures to specific application shortcuts (e.g., launching an IDE, muting Zoom).
2.  **OS-Level Deep Integration:** Connect directly with Windows Focus Assist / macOS Focus modes to automatically mute notifications when deep work is detected.
3.  **Adaptive Music Engine:** Integrate with Spotify to automatically change playlists based on your detected mood (e.g., lo-fi beats when stressed, upbeat music when fatigued).
4.  **Long-term Wellness Analytics:** Provide weekly encrypted, local reports on your cognitive load to help you build better work habits.
