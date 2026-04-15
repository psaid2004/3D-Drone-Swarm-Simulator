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
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget  
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

    def closeEvent(self, event):
        """Ensure the simulation is stopped before the window closes."""
        if self.simulation_page is not None:
            self.simulation_page.close()
        super().closeEvent(event)
    
    def switch_to_simulation(self, config: dict):
        """Create the SimulationPage with the given config an switch to it."""
        self.simulation_page = SimulationPage(config)
        self.stacked_widget.addWidget(self.simulation_page)
        self.stacked_widget.setCurrentWidget(self.simulation_page)

    def switch_to_landing(self):
        """Switch back to the LandingPage and remove the SimulationPage if it exists."""
        self.stacked_widget.setCurrentWidget(self.landing_page)
        if self.simulation_page is not None:
            self.simulation_page.close()  # stop env runner & release PyBullet GUI
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
