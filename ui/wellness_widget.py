"""
Wellness Widget
Displays proactive wellness recommendations based on cognitive state.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt
from config.settings import COLORS


class AlertCard(QFrame):
    """A single alert/recommendation card."""

    def __init__(self, alert_data: dict, parent=None):
        super().__init__(parent)
        self._setup_ui(alert_data)

    def _setup_ui(self, alert: dict):
        # Set styling based on severity
        severity = alert.get("severity", "info")
        if severity == "warning":
            self.setObjectName("alertCardDanger")
            title_color = COLORS.accent_coral
        elif severity == "info":
            self.setObjectName("alertCardInfo")
            title_color = COLORS.accent_cyan
        else:
            self.setObjectName("alertCard")
            title_color = COLORS.accent_amber

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # Header
        header = QHBoxLayout()
        header.setSpacing(8)
        
        icon = QLabel(alert.get("icon", "ℹ️"))
        icon.setStyleSheet("font-size: 16px;")
        header.addWidget(icon)
        
        title = QLabel(alert.get("title", "Notice"))
        title.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {title_color};")
        header.addWidget(title)
        header.addStretch()
        
        layout.addLayout(header)

        # Message
        msg = QLabel(alert.get("message", ""))
        msg.setWordWrap(True)
        msg.setStyleSheet("font-size: 12px; color: rgba(255, 255, 255, 0.8);")
        layout.addWidget(msg)

        # Action Button
        action_text = alert.get("action")
        if action_text:
            btn = QPushButton(action_text)
            if severity == "warning":
                btn.setObjectName("dangerBtn")
            else:
                btn.setObjectName("actionBtn")
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)


class WellnessWidget(QWidget):
    """Panel displaying active wellness alerts and recommendations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("Wellness Engine")
        title.setObjectName("titleLabel")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.status_badge = QLabel("ACTIVE")
        self.status_badge.setObjectName("statusLabel")
        self.status_badge.setStyleSheet(f"background: rgba(57, 255, 20, 0.15); color: {COLORS.accent_lime};")
        header_layout.addWidget(self.status_badge)
        
        layout.addLayout(header_layout)

        subtitle = QLabel("Adaptive environment recommendations")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)
        
        layout.addSpacing(10)

        # Scroll area for alerts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("alertScroll")
        
        self.scroll_content = QWidget()
        self.alerts_layout = QVBoxLayout(self.scroll_content)
        self.alerts_layout.setContentsMargins(0, 0, 10, 0)
        self.alerts_layout.setSpacing(10)
        self.alerts_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Empty state
        self.empty_label = QLabel("No active recommendations.\nYou're doing great! 🌟")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: rgba(255, 255, 255, 0.4); margin-top: 40px;")
        self.alerts_layout.addWidget(self.empty_label)
        
        scroll.setWidget(self.scroll_content)
        layout.addWidget(scroll)

    def update_alerts(self, alerts: list):
        """Update the displayed alerts."""
        # Clear existing
        while self.alerts_layout.count():
            item = self.alerts_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if not alerts:
            self.empty_label = QLabel("No active recommendations.\nYou're doing great! 🌟")
            self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.empty_label.setStyleSheet("color: rgba(255, 255, 255, 0.4); margin-top: 40px;")
            self.alerts_layout.addWidget(self.empty_label)
            return
            
        # Add new alerts
        for alert_data in alerts:
            card = AlertCard(alert_data)
            self.alerts_layout.addWidget(card)
