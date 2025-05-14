from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget
)
from ui.detection_tab import DetectionTab
from ui.rerun_morphometrics_tab import RerunMorphometricsTab


class MainWindow(QMainWindow):
    """
    Main application window that holds the tab interface.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Morphometric Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget for interface organization
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Add the detection tab
        self.detection_tab = DetectionTab()
        self.tab_widget.addTab(self.detection_tab, "Detection")

        # Add the rerun morphometrics tab
        self.rerun_tab = RerunMorphometricsTab()
        self.tab_widget.addTab(self.rerun_tab, "Re-run Morphometrics")

        # New tabs can be added here in the future