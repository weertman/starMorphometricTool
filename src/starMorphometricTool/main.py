from ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication
import sys
import logging

# Configure logging
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')
# Empty the log file on start
open('debug_log.txt', 'w').close()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()