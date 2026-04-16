# -*- coding: utf-8 -*-
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QLineEdit
)

OBJECT_COLORS = [
    "#E6F5D4",  # Verde lima muy claro
    "#D9F0C3",  # Verde menta suave
    "#CCEBB2",  # Verde pastel
    "#BFE6A1",  # Verde manzana claro
    "#B2E190",  # Verde primavera
    "#A5DC7F",  # Verde salvia claro
    "#98D76E",  # Verde hierba suave
    "#8BD25D",  # Verde pistacho
    "#7ECD4C",  # Verde musgo claro
    "#C8EDB8",  # Verde té verde
    "#BEE8AB",  # Verde pera
    "#B4E39E",  # Verde albahaca clara
    "#AADE91",  # Verde esmeralda claro
    "#A0D984",  # Verde oliva claro
    "#96D477",  # Verde césped suave
    "#DCF5CC",  # Verde nieve
    "#D2F0BF",  # Verde junco
    "#F0FAD8",  # Verde muy pálido
    "#EAF9D0",  # Verde lima pálido
    "#E2F7C8",  # Verde brillante claro
]


class LandingPage(QWidget):
    def __init__(self, switch_to_simulation_callback):
        super().__init__()
        self.switch_to_simulation = switch_to_simulation_callback
        self._model_options = self._discover_models()
        self._init_ui()

    # ---- UI ----

    def _init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Drone Swarm Simulator")
        title.setStyleSheet("font-size: 32px; font-weight: bold; margin-bottom: 30px;")

        subtitle = QLabel("Multi-Agent Reinforcement Learning Environment")
        subtitle.setStyleSheet("font-size: 16px; color: gray; margin-bottom: 50px;")

        # ---- Quick Configuration panel ----
        config_frame = QFrame()
        config_frame.setFrameShape(QFrame.Shape.Box)
        config_frame.setMaximumWidth(500)
        config_frame.setStyleSheet(
            "QFrame {"
            "  background-color: #E6F5D4;"
            "  border: 2px;"
            "  border-radius: 5px;"
            "}"
        )
        config_layout = QVBoxLayout(config_frame)


        config_title = QLabel("Quick Configuration")
        config_title.setStyleSheet("font-weight: bold; font-size: 18px;")
        config_layout.addWidget(config_title)

        # Project Name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Project Name")
        config_layout.addLayout(self._labeled_row("Name of Project:", self.name_input))

        # Number of swarms
        self.swarms_spinbox = QSpinBox()
        self.swarms_spinbox.setRange(1, 4)
        self.swarms_spinbox.setValue(1)
        config_layout.addLayout(self._labeled_row("Number of Swarms:", self.swarms_spinbox))

        # Number of drones in a swarm
        self.drone_spinbox = QSpinBox()
        self.drone_spinbox.setRange(1, 10)
        self.drone_spinbox.setValue(2)
        config_layout.addLayout(self._labeled_row("Number of Drones:", self.drone_spinbox))


        # Number of obstacles
        self.obstacle_spinbox = QSpinBox()
        self.obstacle_spinbox.setRange(0, 50)
        self.obstacle_spinbox.setValue(15)
        config_layout.addLayout(self._labeled_row("Number of Obstacles:", self.obstacle_spinbox))

        # Steps
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(100, 10000)
        self.steps_spinbox.setValue(5000)
        self.steps_spinbox.setSingleStep(100)
        config_layout.addLayout(self._labeled_row("Max Steps:", self.steps_spinbox))

        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self._model_options.keys()))
        config_layout.addLayout(self._labeled_row("Model:", self.model_combo))

        config_layout.addStretch()

        # ---- Experimental Settings panel ----
        experimental_frame = QFrame()
        experimental_frame.setFrameShape(QFrame.Shape.Box)
        experimental_frame.setMaximumWidth(500)
        experimental_frame.setStyleSheet(
            "QFrame {"
            "  background-color: #E6F5D4;"
            "  border: 2px;"
            "  border-radius: 5px;"
            "}"
        )
        experimental_layout = QVBoxLayout(experimental_frame)

        experimental_title = QLabel("Experimental Settings")
        experimental_title.setStyleSheet(
            "font-weight: bold; font-size: 18px; margin-bottom: 10px;"
        )
        experimental_layout.addWidget(experimental_title)

        self.world_scale_spinbox = QSpinBox()
        self.world_scale_spinbox.setRange(1, 100)
        self.world_scale_spinbox.setValue(5)
        experimental_layout.addLayout(self._labeled_row("World Scale:", self.world_scale_spinbox))

        self.use_seed_checkbox = QCheckBox("Use Custom Seed")
        experimental_layout.addWidget(self.use_seed_checkbox)

        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 9_999_999)
        self.seed_spinbox.setValue(1)
        self.seed_spinbox.setEnabled(False)
        self.use_seed_checkbox.toggled.connect(self.seed_spinbox.setEnabled)
        experimental_layout.addLayout(self._labeled_row("Seed:", self.seed_spinbox))

        experimental_layout.addStretch()

        # ---- Launch button ----
        self.start_btn = QPushButton("Launch Simulation")
        self.start_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #7ECD4C; color: white; font-size: 16px;"
            "  padding: 10px; min-width: 200px; font-weight: bold;"
            "}"
            "QPushButton:hover { background-color: #6BA03D; }"
        )
        self.start_btn.clicked.connect(self._on_start_clicked)

        if not self._model_options:
            self.model_combo.addItem("(no models found)")
            self.model_combo.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.start_btn.setToolTip("No model .zip files found in the models/ directory")

        # ---- Final layout ----
        layout.addWidget(title)
        layout.addWidget(subtitle)
        settings_row = QHBoxLayout()
        settings_row.addWidget(config_frame)
        settings_row.addWidget(experimental_frame)
        layout.addLayout(settings_row)
        layout.addSpacing(30)
        layout.addWidget(self.start_btn)
        self.setLayout(layout)

    # ---- Helpers ----

    @staticmethod
    def _labeled_row(label_text: str, widget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        row.addWidget(widget)
        row.addStretch()
        return row

    def _discover_models(self) -> dict:
        # parents[2] = davids_drones/
        project_root = Path(__file__).resolve().parents[2]
        models_dir = project_root / "models"
        options = {}
        if not models_dir.is_dir():
            return options
        for model_file in sorted(models_dir.glob("*.zip")):
            options[model_file.name] = str(model_file)
        return options

    def _on_start_clicked(self):
        selected_label = self.model_combo.currentText()
        selected_path = self._model_options.get(selected_label, "")
        t = self.name_input.text()
        config = {
            "project_name": (t if t else "obj"),
            "num_drones": self.drone_spinbox.value(),
            "num_obstacles": self.obstacle_spinbox.value(),
            "max_steps": self.steps_spinbox.value(),
            "render_mode": "human",
            "lidar": True,
            "model_path": selected_path,
            "policy_model_path": selected_path,
            "world_scale": self.world_scale_spinbox.value(),
            "num_swarms": self.swarms_spinbox.value(),
            "seed": self.seed_spinbox.value() if self.use_seed_checkbox.isChecked() else None,
        }
        self.switch_to_simulation(config)

