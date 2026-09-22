"""
AuraDesk — Main Entry Point
"""
import sys
import os

# Ensure the app path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Configure app settings
    app.setApplicationName("AuraDesk")
    app.setOrganizationName("HackathonTeam")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
