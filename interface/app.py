"""
Entry point for the Drone Swarm Simulator GUI.

Run from the davids_drones/ directory:
    python -m interface.app
or:
    python interface/app.py
"""
from pathlib import Path
import sys
import importlib
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _preload_torch():
    """
    Import torch before Qt to prevent a DLL conflict on Windows.
    Returns an error string on failure (non-fatal), or None on success.
    """
    try:
        importlib.import_module("torch")
        return None
    except Exception as exc:
        return str(exc)


_torch_error = _preload_torch()

# Qt imports MUST come after the torch preload above.
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QProgressDialog, QMessageBox
from PyQt6.QtCore import QTimer, Qt
from interface.pages.landing_page import LandingPage       
from interface.pages.simulation_page import SimulationPage 


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Swarm Simulator")
        self.setGeometry(50, 50, 1200, 650)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create pages
        self.landing_page = LandingPage(self.switch_to_simulation)
        # simulation_page is created on demand with config from landing_page; see switch_to_simulation()
        self.simulation_page: SimulationPage | None = None

        self.stacked_widget.addWidget(self.landing_page)
        self.stacked_widget.setCurrentWidget(self.landing_page)
        
    def switch_to_simulation(self, config: dict):
        """Create the SimulationPage with the given config an switch to it."""
        self.loading_dialog = QProgressDialog("Loading simulation...", None, 0, 100, self)
        self.loading_dialog.setWindowTitle("Please Wait")
        self.loading_dialog.setCancelButton(None)
        self.loading_dialog.setMinimumDuration(0)
        self.loading_dialog.show()
        
        # Start a timer to animate the progress
        self.progress_value = 0
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(5) 
        QApplication.processEvents()
        
        self.pending_config = config
        QTimer.singleShot(50, self._create_simulation_page)

    def update_progress(self):
        if self.progress_value < 90:
            self.progress_value += 3.6
            self.loading_dialog.setValue(int(self.progress_value))
        QApplication.processEvents()

    def _create_simulation_page(self):
        """Actually create the simulation page."""
        try:
            self.loading_dialog.setValue(100)
            self.simulation_page = SimulationPage(self.pending_config)
            QApplication.processEvents()
            QTimer.singleShot(1000, self._finish_loading)
            
        except Exception as e:
            self.progress_timer.stop()
            self.loading_dialog.close()
            QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

    def _finish_loading(self):
        """Complete the loading and switch pages."""
        self.progress_timer.stop()
        self.stacked_widget.addWidget(self.simulation_page)
        self.stacked_widget.setCurrentWidget(self.simulation_page)
        self.loading_dialog.close()
    
    def switch_to_landing(self):
        """Switch back to the LandingPage and remove the SimulationPage if it exists."""
        self.stacked_widget.setCurrentWidget(self.landing_page)
        if self.simulation_page is not None:
            self.stacked_widget.removeWidget(self.simulation_page)
            self.simulation_page = None


def main():
    if _torch_error:
        print(f"[Warning] Could not preload torch: {_torch_error}")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
