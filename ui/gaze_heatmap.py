"""
Gaze Heatmap Widget
Visualizes user attention over time using a smooth, decaying translucent heatmap.
"""
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QColor, QRadialGradient
from config.settings import GAZE_HEATMAP_DECAY


class GazeHeatmap(QWidget):
    """Overlay widget that renders a smooth decaying heatmap of gaze focus."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Recent gaze points with intensities
        self._gaze_points = []  # list of (x_norm, y_norm, intensity)

    def add_gaze_point(self, x_ratio: float, y_ratio: float):
        """Add a gaze point and decay old points."""
        if not (0 <= x_ratio <= 1 and 0 <= y_ratio <= 1):
            return

        # Decay existing points
        decayed = []
        for x, y, intensity in self._gaze_points:
            new_intensity = intensity * 0.94
            if new_intensity > 0.08:
                decayed.append((x, y, new_intensity))
        self._gaze_points = decayed

        # Add new point (max 15 recent points)
        self._gaze_points.append((x_ratio, y_ratio, 1.0))
        if len(self._gaze_points) > 15:
            self._gaze_points.pop(0)

        self.update()

    def decay_only(self):
        """Decay heatmap without adding new points (e.g. when looking away)."""
        if not self._gaze_points:
            return
        decayed = []
        for x, y, intensity in self._gaze_points:
            new_intensity = intensity * 0.90
            if new_intensity > 0.08:
                decayed.append((x, y, new_intensity))
        self._gaze_points = decayed
        self.update()

    def paintEvent(self, event):
        if not self._gaze_points:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = float(self.width())
        h = float(self.height())

        for x_norm, y_norm, intensity in self._gaze_points:
            cx = x_norm * w
            cy = y_norm * h
            radius = min(w, h) * 0.18

            # Soft radial glow
            gradient = QRadialGradient(QPointF(cx, cy), radius)
            # Center color: warm white core fading into Snapdragon Red
            alpha_core = int(120 * intensity)
            alpha_mid = int(80 * intensity)
            gradient.setColorAt(0.0, QColor(255, 255, 255, alpha_core))
            gradient.setColorAt(0.2, QColor(255, 20, 20, alpha_mid))
            gradient.setColorAt(0.6, QColor(211, 0, 0, int(30 * intensity)))
            gradient.setColorAt(1.0, QColor(100, 0, 0, 0))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

        painter.end()

    def clear(self):
        self._gaze_points.clear()
        self.update()
