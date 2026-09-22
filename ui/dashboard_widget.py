"""
Cognitive State Dashboard Widget
Real-time visualization of focus, fatigue, stress, emotion, and session stats.
Features animated progress bars, trend sparklines, and live metrics.
"""
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QFrame, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, pyqtProperty, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QPainterPath, QLinearGradient, QFont
from config.settings import COLORS


class SparklineWidget(QWidget):
    """Tiny inline trend chart showing recent history."""

    def __init__(self, color="#00d4ff", parent=None):
        super().__init__(parent)
        self._data = []
        self._color = QColor(color)
        self._max_points = 60
        self.setFixedHeight(32)
        self.setMinimumWidth(100)

    def set_data(self, data: list):
        self._data = data[-self._max_points:]
        self.update()

    def paintEvent(self, event):
        if len(self._data) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 4

        data = self._data
        min_val = min(data) if data else 0
        max_val = max(data) if data else 1
        val_range = max(max_val - min_val, 0.01)

        # Build path
        path = QPainterPath()
        fill_path = QPainterPath()

        for i, val in enumerate(data):
            x = margin + (i / max(len(data) - 1, 1)) * (w - 2 * margin)
            y = h - margin - ((val - min_val) / val_range) * (h - 2 * margin)
            if i == 0:
                path.moveTo(x, y)
                fill_path.moveTo(x, h - margin)
                fill_path.lineTo(x, y)
            else:
                path.lineTo(x, y)
                fill_path.lineTo(x, y)

        # Close fill path
        fill_path.lineTo(margin + ((len(data) - 1) / max(len(data) - 1, 1)) * (w - 2 * margin), h - margin)
        fill_path.closeSubpath()

        # Draw filled area with gradient
        gradient = QLinearGradient(0, 0, 0, h)
        fill_color = QColor(self._color)
        fill_color.setAlpha(40)
        gradient.setColorAt(0, fill_color)
        fill_color.setAlpha(5)
        gradient.setColorAt(1, fill_color)
        painter.fillPath(fill_path, gradient)

        # Draw line
        pen = QPen(self._color, 1.5)
        painter.setPen(pen)
        painter.drawPath(path)

        # Draw current value dot
        if data:
            last_x = w - margin
            last_y = h - margin - ((data[-1] - min_val) / val_range) * (h - 2 * margin)
            painter.setBrush(self._color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(last_x) - 3, int(last_y) - 3, 6, 6)

        painter.end()


class MetricCard(QFrame):
    """Individual metric display card with label, value, bar, and sparkline."""

    def __init__(self, title, color, icon="", parent=None):
        super().__init__(parent)
        self.setObjectName("glassCard")
        self._color = color
        self._setup_ui(title, icon)

    def _setup_ui(self, title, icon):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(6)

        if icon:
            icon_label = QLabel(icon)
            icon_label.setStyleSheet(f"font-size: 14px; font-weight: 900; color: {self._color};")
            header.addWidget(icon_label)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionHeader")
        header.addWidget(self.title_label)
        header.addStretch()

        self.value_label = QLabel("0%")
        self.value_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: 700;
            color: {self._color};
        """)
        header.addWidget(self.value_label)

        layout.addLayout(header)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: rgba(255, 255, 255, 0.05);
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                border-radius: 4px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self._color}, stop:1 {self._adjust_color(self._color, 0.7)});
            }}
        """)
        layout.addWidget(self.progress)

        # Sparkline
        self.sparkline = SparklineWidget(color=self._color)
        layout.addWidget(self.sparkline)

    def set_value(self, value: float, data: list = None):
        """Update metric value (0-1 scale)."""
        pct = int(value * 100)
        self.value_label.setText(f"{pct}%")
        self.progress.setValue(pct)
        if data:
            self.sparkline.set_data(data)

    def _adjust_color(self, hex_color: str, factor: float) -> str:
        """Darken or lighten a hex color."""
        color = QColor(hex_color)
        r = min(255, int(color.red() * factor))
        g = min(255, int(color.green() * factor))
        b = min(255, int(color.blue() * factor))
        return f"rgb({r}, {g}, {b})"


class DashboardWidget(QWidget):
    """Main cognitive state dashboard with real-time metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── Header ──
        header = QLabel("Cognitive Dashboard")
        header.setObjectName("titleLabel")
        layout.addWidget(header)

        subtitle = QLabel("Real-time cognitive state analysis")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)

        layout.addSpacing(6)

        # ── Emotion Display ──
        emotion_frame = QFrame()
        emotion_frame.setObjectName("glassCard")
        emotion_layout = QHBoxLayout(emotion_frame)
        emotion_layout.setContentsMargins(16, 12, 16, 12)

        self.emotion_emoji = QLabel("CALM")
        self.emotion_emoji.setObjectName("emojiLabel")
        self.emotion_emoji.setStyleSheet("color: #FFFFFF;")
        emotion_layout.addWidget(self.emotion_emoji)

        emotion_layout.addStretch()

        self.emotion_conf = QLabel("CONF: 0%")
        self.emotion_conf.setObjectName("subtitleLabel")
        emotion_layout.addWidget(self.emotion_conf)

        layout.addWidget(emotion_frame)

        # ── Metric Cards ──
        self.focus_card = MetricCard("FOCUS", COLORS.accent_lime, "◆")
        layout.addWidget(self.focus_card)

        self.fatigue_card = MetricCard("FATIGUE", COLORS.accent_magenta, "◆")
        layout.addWidget(self.fatigue_card)

        self.stress_card = MetricCard("STRESS", COLORS.accent_amber, "◆")
        layout.addWidget(self.stress_card)

        # ── Session Stats ──
        stats_frame = QFrame()
        stats_frame.setObjectName("glassCard")
        stats_grid = QGridLayout(stats_frame)
        stats_grid.setContentsMargins(12, 10, 12, 10)
        stats_grid.setSpacing(8)

        stats_title = QLabel("SESSION STATS")
        stats_title.setObjectName("sectionHeader")
        stats_grid.addWidget(stats_title, 0, 0, 1, 2)

        self.session_time = self._create_stat("TIME", "0:00")
        stats_grid.addWidget(self.session_time, 1, 0)

        self.blink_count = self._create_stat("BLINKS", "0")
        stats_grid.addWidget(self.blink_count, 1, 1)

        self.blink_rate_label = self._create_stat("RATE", "0/min")
        stats_grid.addWidget(self.blink_rate_label, 2, 0)

        self.gesture_count = self._create_stat("GESTURES", "0")
        stats_grid.addWidget(self.gesture_count, 2, 1)

        layout.addWidget(stats_frame)

        layout.addStretch()

    def _create_stat(self, label_text: str, default_val: str) -> QFrame:
        """Create a mini stat display."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.4); letter-spacing: 1px;")
        layout.addWidget(lbl)
        
        layout.addStretch()

        val_lbl = QLabel(default_val)
        val_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #FFFFFF;")
        val_lbl.setObjectName("statValue")
        layout.addWidget(val_lbl)

        frame._value_label = val_lbl
        return frame

    def update_data(self, result):
        """Update all dashboard metrics from a FrameResult."""
        # Emotion
        self.emotion_emoji.setText(result.emotion_emoji.upper())
        self.emotion_conf.setText(f"CONF: {int(result.emotion_confidence * 100)}%")

        # Focus
        self.focus_card.set_value(result.focus_score)

        # Fatigue
        self.fatigue_card.set_value(result.fatigue_level)

        # Stress
        self.stress_card.set_value(result.stress_level)

        # Session stats
        minutes = int(result.session_minutes)
        self.session_time._value_label.setText(f"{minutes // 60}:{minutes % 60:02d}")
        self.blink_count._value_label.setText(str(result.total_blinks))
        self.blink_rate_label._value_label.setText(f"{result.blink_rate:.0f}/min")

    def update_trends(self, trends: dict):
        """Update sparkline trends."""
        if "focus" in trends:
            self.focus_card.sparkline.set_data(trends["focus"])
        if "fatigue" in trends:
            self.fatigue_card.sparkline.set_data(trends["fatigue"])
        if "stress" in trends:
            self.stress_card.sparkline.set_data(trends["stress"])

    def set_gesture_count(self, count: int):
        """Update gesture count."""
        self.gesture_count._value_label.setText(str(count))
