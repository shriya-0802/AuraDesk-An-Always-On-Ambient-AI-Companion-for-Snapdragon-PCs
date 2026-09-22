"""
AuraDesk Stylesheet — Dark Glassmorphism Theme
Premium, futuristic design with neon accents, glass panels, and glow effects.
"""

MAIN_STYLESHEET = """
/* ═══════════════════════════════════════════
   AuraDesk — Snapdragon Carbon Theme
   ═══════════════════════════════════════════ */

/* ── Global ── */
QWidget {
    background-color: transparent;
    color: #FFFFFF;
    font-family: -apple-system, 'Helvetica Neue', 'Segoe UI', sans-serif;
    font-size: 13px;
}

QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #050505, stop:0.5 #080809, stop:1 #0B0B0C);
}

/* ── Glass Panel Base ── */
QFrame#glassPanel {
    background: rgba(16, 16, 18, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
}

QFrame#glassCard {
    background: rgba(24, 24, 28, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 12px;
}

/* ── Sidebar ── */
QFrame#sidebar {
    background: rgba(8, 8, 10, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0px;
}

QPushButton#sidebarBtn {
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 14px;
    font-size: 20px;
    min-width: 48px;
    min-height: 48px;
    color: rgba(255, 255, 255, 0.4);
}

QPushButton#sidebarBtn:hover {
    background: rgba(255, 20, 20, 0.1);
    color: #FF1414;
}

QPushButton#sidebarBtn:checked {
    background: rgba(211, 0, 0, 0.15);
    color: #FF1414;
    border: 1px solid rgba(255, 20, 20, 0.3);
}

/* ── Labels ── */
QLabel#titleLabel {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.5px;
}

QLabel#subtitleLabel {
    font-size: 11px;
    font-weight: 400;
    color: rgba(255, 255, 255, 0.5);
    letter-spacing: 1px;
    text-transform: uppercase;
}

QLabel#valueLabel {
    font-size: 28px;
    font-weight: 700;
    color: #FF1414; /* Snapdragon Red */
}

QLabel#unitLabel {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

QLabel#emojiLabel {
    font-size: 24px; /* Reduced from 42px since we use text icons now */
    font-weight: 700;
}

QLabel#statusLabel {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.8);
    padding: 4px 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    letter-spacing: 0.5px;
}

QLabel#alertTitle {
    font-size: 13px;
    font-weight: 700;
    color: #FF4400;
}

QLabel#alertMessage {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.7);
}

QLabel#gestureLabel {
    font-size: 14px;
    font-weight: 700;
    color: #FFFFFF;
    padding: 10px 20px;
    background: rgba(211, 0, 0, 0.25);
    border: 1px solid rgba(255, 20, 20, 0.4);
    border-radius: 8px;
    letter-spacing: 1px;
}

QLabel#fpsLabel {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.5);
    padding: 4px 8px;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 4px;
    letter-spacing: 1px;
}

/* ── Progress Bars ── */
QProgressBar {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    height: 8px;
    text-align: center;
    font-size: 1px;
}

QProgressBar::chunk {
    border-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #A00000, stop:1 #FF1414);
}

QProgressBar#focusBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #FFFFFF, stop:1 #FF1414); /* White to Snapdragon Red */
}

QProgressBar#fatigueBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #D30000, stop:1 #FF1414);
}

QProgressBar#stressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #D30000, stop:1 #FF4400);
}

/* ── Buttons ── */
QPushButton#actionBtn {
    background: rgba(255, 20, 20, 0.15);
    border: 1px solid rgba(255, 20, 20, 0.3);
    border-radius: 6px;
    padding: 8px 20px;
    color: #FF1414;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
}

QPushButton#actionBtn:hover {
    background: rgba(255, 20, 20, 0.25);
    border-color: rgba(255, 20, 20, 0.5);
}

QPushButton#actionBtn:pressed {
    background: rgba(255, 20, 20, 0.4);
}

QPushButton#dangerBtn {
    background: rgba(211, 0, 0, 0.2);
    border: 1px solid rgba(211, 0, 0, 0.4);
    border-radius: 6px;
    padding: 8px 20px;
    color: #FFFFFF;
    font-weight: 700;
}

/* ── Scroll Area ── */
QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 4px 0;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 3px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(255, 255, 255, 0.35);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* ── Tooltips ── */
QToolTip {
    background: rgba(15, 15, 18, 0.95);
    color: #FFFFFF;
    border: 1px solid rgba(255, 20, 20, 0.4);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 600;
}

/* ── Camera Preview ── */
QLabel#cameraView {
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
}

/* ── Tab-like section headers ── */
QLabel#sectionHeader {
    font-size: 10px;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 4px 0;
}

/* ── Alert Card ── */
QFrame#alertCard {
    background: rgba(255, 68, 0, 0.1);
    border: 1px solid rgba(255, 68, 0, 0.25);
    border-radius: 8px;
    padding: 10px;
}

QFrame#alertCardDanger {
    background: rgba(211, 0, 0, 0.15);
    border: 1px solid rgba(211, 0, 0, 0.3);
    border-radius: 8px;
    padding: 10px;
}

QFrame#alertCardInfo {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 8px;
    padding: 10px;
}

/* ── Gesture Card ── */
QFrame#gestureCard {
    background: rgba(255, 20, 20, 0.08);
    border: 1px solid rgba(255, 20, 20, 0.2);
    border-radius: 8px;
    padding: 10px;
}

QFrame#gestureCardActive {
    background: rgba(211, 0, 0, 0.2);
    border: 1px solid rgba(255, 20, 20, 0.4);
    border-radius: 8px;
    padding: 10px;
}
"""
