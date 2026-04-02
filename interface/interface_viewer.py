from pathlib import Path
import sys
import importlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

## Preload torch before initializing Qt to prevent scary error 
def _preload_torch_before_qt():
    try:
        importlib.import_module("torch")
        return None
    except Exception as exc:
        return str(exc)
    
torch_error = _preload_torch_before_qt()

# !!! MUST KEEP AFTER TORCH PRELOAD OTHERWISE BBE OCCURS (BIG BAD ERROR) !!!
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStackedWidget,
)
from interface.configs_viewer import LandingPage
from interface.simulation_viewer import SimulationViewer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Swarm Simulator")
        self.setGeometry(50, 50, 1200, 650)
        
        # Create stacked widget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create pages
        self.landing_page = LandingPage(self.switch_to_simulation)
        self.simulation_viewer = None  # Will be created when needed
        
        # Add pages to stack
        self.stacked_widget.addWidget(self.landing_page)
        
        # Show landing page first
        self.stacked_widget.setCurrentWidget(self.landing_page)
    
    def switch_to_simulation(self, config):
        # Create simulation viewer with config
        self.simulation_viewer = SimulationViewer(config)
        self.stacked_widget.addWidget(self.simulation_viewer)
        self.stacked_widget.setCurrentWidget(self.simulation_viewer)
    
    def switch_to_landing(self):
        self.stacked_widget.setCurrentWidget(self.landing_page)
        # Clean up simulation viewer if needed
        if self.simulation_viewer:
            self.stacked_widget.removeWidget(self.simulation_viewer)
            self.simulation_viewer = None


def main():
    if torch_error:
        print(f"{torch_error}")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()