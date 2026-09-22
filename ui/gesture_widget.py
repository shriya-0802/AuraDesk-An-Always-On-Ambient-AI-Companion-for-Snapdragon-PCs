"""
Gesture Feedback Widget
Displays an animated overlay when a gesture is triggered.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, pyqtProperty
from PyQt6.QtGui import QColor
from config.settings import COLORS


class GestureWidget(QFrame):
    """Overlay card that pops up when a gesture is recognized."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gestureCard")
        self._setup_ui()
        self.hide()
        self._opacity = 0.0
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)
        self._animation = QPropertyAnimation(self, b"opacity_prop")

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        self.title_label = QLabel("GESTURE DETECTED")
        self.title_label.setObjectName("sectionHeader")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self.action_label = QLabel("")
        self.action_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 700;
            color: {COLORS.text_primary};
        """)
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.action_label)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"""
            font-size: 12px;
            font-weight: 600;
            color: {COLORS.accent_cyan};
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    @pyqtProperty(float)
    def opacity_prop(self):
        return self._opacity

    @opacity_prop.setter
    def opacity_prop(self, value):
        self._opacity = value
        self.setStyleSheet(f"""
            QFrame#gestureCard {{
                background: rgba(211, 0, 0, {0.25 * value});
                border: 1px solid rgba(255, 20, 20, {0.5 * value});
                border-radius: 8px;
            }}
            QLabel {{ opacity: {value}; }}
        """)

    def show_gesture(self, gesture_label: str, action_message: str):
        """Display a gesture notification and auto-hide it after a delay."""
        self.action_label.setText(gesture_label)
        self.status_label.setText(action_message)
        
        # Stop any running animations
        self._animation.stop()
        
        # Fade in
        self.show()
        self._animation.setDuration(200)
        self._animation.setStartValue(self._opacity)
        self._animation.setEndValue(1.0)
        self._animation.start()
        
        # Schedule hide
        self._hide_timer.start(2500)

    def _fade_out(self):
        self._animation.stop()
        self._animation.setDuration(400)
        self._animation.setStartValue(self._opacity)
        self._animation.setEndValue(0.0)
        self._animation.finished.connect(self._on_fade_out_finished)
        self._animation.start()
        
    def _on_fade_out_finished(self):
        if self._opacity == 0.0:
            self.hide()
