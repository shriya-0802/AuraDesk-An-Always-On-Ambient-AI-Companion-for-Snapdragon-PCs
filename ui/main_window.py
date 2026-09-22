from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame,
    QPushButton, QLabel
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent, QShortcut, QKeySequence

from config.settings import WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT, PLATFORM, COLORS
from ui.styles import MAIN_STYLESHEET
from ui.camera_preview import CameraPreview
from ui.dashboard_widget import DashboardWidget
from ui.gesture_widget import GestureWidget
from ui.gaze_heatmap import GazeHeatmap
from ui.wellness_widget import WellnessWidget

from core.camera_manager import CameraThread
from core.inference_engine import InferenceEngine
from core.action_engine import ActionEngine


class MainWindow(QMainWindow):
    """Main application window for AuraDesk."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AuraDesk — Cognitive Computing Environment")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setStyleSheet(MAIN_STYLESHEET)

        # AI Engines
        self.inference_engine = InferenceEngine()
        self.action_engine = ActionEngine()

        self._setup_ui()
        self._setup_shortcuts()
        self._start_camera()

    def _setup_ui(self):
        """Build main window layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ── Left Column (Camera + Gestures + Gaze) ──
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Interactive Demo Bar
        demo_bar = self._create_demo_bar()
        left_layout.addWidget(demo_bar)

        # Camera view container
        camera_container = QWidget()
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_preview = CameraPreview()
        camera_layout.addWidget(self.camera_preview)

        left_layout.addWidget(camera_container, stretch=3)

        # Gaze Heatmap Overlay
        self.gaze_heatmap = GazeHeatmap(self.camera_preview)
        self.gaze_heatmap.resize(self.camera_preview.size())

        # Gesture feedback overlay
        self.gesture_widget = GestureWidget(self.camera_preview)
        self.gesture_widget.setFixedSize(300, 120)

        # Privacy Guardian Overlay
        self.privacy_overlay = QLabel("WARNING: SHOULDER SURFING DETECTED\n\nPrivacy Guardian Active", self.camera_preview)
        self.privacy_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.privacy_overlay.setStyleSheet("""
            background: rgba(211, 0, 0, 0.90);
            color: #FFFFFF;
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 2px;
        """)
        self.privacy_overlay.hide()

        # Wellness panel at bottom left
        self.wellness_panel = WellnessWidget()
        left_layout.addWidget(self.wellness_panel, stretch=2)

        main_layout.addWidget(left_panel, stretch=5)

        # ── Right Column (Dashboard) ──
        self.dashboard = DashboardWidget()
        main_layout.addWidget(self.dashboard, stretch=3)

    def _create_demo_bar(self) -> QWidget:
        """Create a sleek quick-action demo bar for testing system states."""
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.bg_card};
                border: 1px solid {COLORS.border_subtle};
                border-radius: 10px;
                padding: 4px 8px;
            }}
            QPushButton {{
                background: rgba(255, 255, 255, 0.08);
                color: #e0e0e0;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {COLORS.accent_cyan};
                color: #0a0a0f;
                border-color: {COLORS.accent_cyan};
            }}
        """)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        label = QLabel("DEBUG MODE:")
        label.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(label)

        btn_live = QPushButton("LIVE AI (L)")
        btn_live.setStyleSheet(f"background: rgba(255, 20, 20, 0.2); border-color: {COLORS.accent_cyan}; color: #ffffff;")
        btn_live.clicked.connect(lambda: self.set_demo_state("live"))
        layout.addWidget(btn_live)

        btn_focus = QPushButton("FOCUS (F)")
        btn_focus.clicked.connect(lambda: self.set_demo_state("focused"))
        layout.addWidget(btn_focus)

        btn_stress = QPushButton("STRESS (S)")
        btn_stress.clicked.connect(lambda: self.set_demo_state("stressed"))
        layout.addWidget(btn_stress)

        btn_yawn = QPushButton("YAWN (Y)")
        btn_yawn.clicked.connect(lambda: self.set_demo_state("yawn"))
        layout.addWidget(btn_yawn)

        btn_away = QPushButton("AWAY (A)")
        btn_away.clicked.connect(lambda: self.set_demo_state("looking_away"))
        layout.addWidget(btn_away)

        btn_norm = QPushButton("NORMAL (N)")
        btn_norm.clicked.connect(lambda: self.set_demo_state("normal"))
        layout.addWidget(btn_norm)

        layout.addStretch()
        return container

    def _setup_shortcuts(self):
        """Register global application shortcuts that always work anywhere in the window."""
        shortcuts = [
            ("L", lambda: self.set_demo_state("live")),
            ("F", lambda: self.set_demo_state("focused")),
            ("S", lambda: self.set_demo_state("stressed")),
            ("Y", lambda: self.set_demo_state("yawn")),
            ("A", lambda: self.set_demo_state("looking_away")),
            ("N", lambda: self.set_demo_state("normal")),
            ("1", lambda: self.trigger_gesture("fist")),
            ("2", lambda: self.trigger_gesture("open_palm")),
            ("3", lambda: self.trigger_gesture("point_up")),
            ("4", lambda: self.trigger_gesture("point_down")),
            ("5", lambda: self.trigger_gesture("peace")),
            ("6", lambda: self.trigger_gesture("thumbs_up")),
        ]
        self._shortcut_objs = []
        for key_str, slot in shortcuts:
            sc = QShortcut(QKeySequence(key_str), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)
            self._shortcut_objs.append(sc)

    def set_demo_state(self, mode: str):
        """Set simulation mode."""
        if hasattr(self.inference_engine, 'face_detector'):
            self.inference_engine.face_detector._demo_mode = mode
            print(f"[AuraDesk] Mode set to: {mode}")

    def trigger_gesture(self, gesture: str):
        """Manually trigger a gesture action."""
        action_result = self.action_engine.execute_gesture_action(gesture)
        if action_result["success"]:
            from config.settings import GESTURES
            self.gesture_widget.show_gesture(GESTURES.get(gesture, gesture), action_result["message"])
            log = self.action_engine.get_action_log()
            self.dashboard.set_gesture_count(len(log))

    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)

        if hasattr(self, 'gaze_heatmap') and hasattr(self, 'camera_preview'):
            self.gaze_heatmap.resize(self.camera_preview.size())

        if hasattr(self, 'gesture_widget') and hasattr(self, 'camera_preview'):
            cw = self.camera_preview.width()
            ch = self.camera_preview.height()
            gw = self.gesture_widget.width()
            gh = self.gesture_widget.height()
            self.gesture_widget.move((cw - gw) // 2, (ch - gh) // 2)

        if hasattr(self, 'privacy_overlay') and hasattr(self, 'camera_preview'):
            self.privacy_overlay.resize(self.camera_preview.size())

    def _start_camera(self):
        """Start camera thread."""
        self.camera_thread = CameraThread()
        self.camera_thread.set_inference_engine(self.inference_engine)
        self.camera_thread.frame_ready.connect(self._on_frame_ready)
        self.camera_thread.start()

    def _on_frame_ready(self, result):
        """Handle new frame result from AI pipeline."""
        # 1. Update camera preview
        self.camera_preview.update_frame(result)

        # 2. Update dashboard metrics & sparklines
        self.dashboard.update_data(result)
        if hasattr(result, 'trends') and result.trends:
            self.dashboard.update_trends(result.trends)

        # 3. Handle Gestures
        if result.gesture_triggered:
            print(f"[MainWindow] GESTURE TRIGGERED: {result.active_gesture}")
            action_result = self.action_engine.execute_gesture_action(result.active_gesture)
            if action_result["success"] and result.hands_data:
                self.gesture_widget.show_gesture(
                    result.hands_data[0]["gesture_label"],
                    action_result["message"]
                )
            elif result.hands_data:
                # Show feedback even on failure so user knows gesture was recognized
                self.gesture_widget.show_gesture(
                    result.hands_data[0]["gesture_label"],
                    action_result.get("message", "Action failed")
                )
            log = self.action_engine.get_action_log()
            self.dashboard.set_gesture_count(len(log))

        # 4. Handle Gaze Heatmap
        if result.face_detected and result.looking_at_screen:
            self.gaze_heatmap.add_gaze_point(result.gaze_point[0], result.gaze_point[1])
        else:
            self.gaze_heatmap.decay_only()

        # 5. Handle Wellness Alerts
        if hasattr(result, 'alerts'):
            self.wellness_panel.update_alerts(result.alerts)

        # 6. Hackathon Features: Privacy Guardian & AWA
        if hasattr(result, 'shoulder_surfing'):
            if result.shoulder_surfing:
                self.privacy_overlay.show()
            else:
                self.privacy_overlay.hide()

        if hasattr(result, 'awa_trigger') and result.awa_trigger:
            if not getattr(self, '_awa_triggered', False):
                self._awa_triggered = True
                self.gesture_widget.show_gesture("WELLNESS INTERVENTION", "Dark Mode & DND Activated")
                self.action_engine.apply_wellness_action("toggle_dark_mode")
                self.action_engine.apply_wellness_action("enable_dnd")

    def closeEvent(self, event):
        """Clean up resources on close."""
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
        if hasattr(self, 'inference_engine'):
            self.inference_engine.release()
        event.accept()
