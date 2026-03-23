import os
import sys
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
  )
class LandingPage(QWidget):
    def __init__(self, switch_to_simulation_callback):
        super().__init__()
        self.switch_to_simulation = switch_to_simulation_callback
        self.init_ui()
        
       ###### UI ######
    def init_ui(self):
        layout = QVBoxLayout()
        #layout.setAlignment(Qt.AlignCenter)
        
        # Title
        title = QLabel("Drone Swarm Simulator")
        title.setStyleSheet("font-size: 32px; font-weight: bold; margin-bottom: 30px;")
        #title.setAlignment(Qt.AlignCenter)
        
        # Subtitle
        subtitle = QLabel("Multi-Agent Reinforcement Learning Environment")
        subtitle.setStyleSheet("font-size: 16px; color: gray; margin-bottom: 50px;")
        #subtitle.setAlignment(Qt.AlignCenter)
        


        ################## 1. Settings panel ##################

        config_frame = QFrame()
        config_frame.setFrameShape(QFrame.Shape.Box)
        config_frame.setMaximumWidth(500)
        config_layout = QVBoxLayout(config_frame)
        
        config_title = QLabel("Quick Configuration")
        config_title.setStyleSheet("font-weight: bold; font-size: 18px;")
        config_layout.addWidget(config_title)
        
        # Number of drones
        drone_layout = QHBoxLayout()
        drone_layout.addWidget(QLabel("Number of Drones:"))
        self.drone_spinbox = QSpinBox()
        self.drone_spinbox.setRange(1, 10)
        self.drone_spinbox.setValue(2)
        drone_layout.addWidget(self.drone_spinbox)
        drone_layout.addStretch()
        config_layout.addLayout(drone_layout)
        
        # Number of obstacles
        obstacle_layout = QHBoxLayout()
        obstacle_layout.addWidget(QLabel("Number of Obstacles:"))
        self.obstacle_spinbox = QSpinBox()
        self.obstacle_spinbox.setRange(0, 50)
        self.obstacle_spinbox.setValue(15)
        obstacle_layout.addWidget(self.obstacle_spinbox)
        obstacle_layout.addStretch()
        config_layout.addLayout(obstacle_layout)
        
        # Max steps
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Max Steps:"))
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(100, 10000)
        self.steps_spinbox.setValue(5000)
        self.steps_spinbox.setSingleStep(100)
        steps_layout.addWidget(self.steps_spinbox)
        steps_layout.addStretch()
        config_layout.addLayout(steps_layout)
        
        # Render mode
        render_layout = QHBoxLayout()
        render_layout.addWidget(QLabel("Render Mode:"))
        self.render_combo = QComboBox()
        self.render_combo.addItems(["human", "none"])
        render_layout.addWidget(self.render_combo)
        render_layout.addStretch()
        config_layout.addLayout(render_layout)
        
        # Lidar option
        self.lidar_checkbox = QCheckBox("Enable Lidar")
        config_layout.addWidget(self.lidar_checkbox)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self._model_options = self._discover_models()
        self.model_combo.addItems(list(self._model_options.keys()))
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        config_layout.addLayout(model_layout)

        config_layout.addStretch()



        ################ 2. Experimental settings panel ################

        experimental_frame = QFrame()
        experimental_frame.setFrameShape(QFrame.Shape.Box)
        experimental_frame.setMaximumWidth(500)
        experimental_layout = QVBoxLayout(experimental_frame)

        experimental_title = QLabel("Experimental Settings")
        experimental_title.setStyleSheet("font-weight: bold; font-size: 18px; margin-bottom: 10px;")  
        experimental_layout.addWidget(experimental_title)

        # World scale
        world_layout = QHBoxLayout()
        world_layout.addWidget(QLabel("World Scale:"))
        self.world_scale_spinbox = QSpinBox()
        self.world_scale_spinbox.setRange(1, 100)
        self.world_scale_spinbox.setValue(10)
        world_layout.addWidget(self.world_scale_spinbox)
        world_layout.addStretch()
        experimental_layout.addLayout(world_layout)

        self.start_btn = QPushButton("Launch Simulation")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                min-width: 200px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_btn.clicked.connect(self.on_start_clicked)
        

        ################# Final layout  ################

        layout.addWidget(title)
        layout.addWidget(subtitle)
        settings_row = QHBoxLayout()
        settings_row.addWidget(config_frame)
        settings_row.addWidget(experimental_frame)
        layout.addLayout(settings_row)
        layout.addSpacing(30)
        layout.addWidget(self.start_btn)
        
        self.setLayout(layout)



    ################ Helper methods ################ 

    def _discover_models(self):
        project_root = Path(__file__).resolve().parents[1]
        models_dir = project_root / "models"
        options = {"No model": ""}
        if not models_dir.is_dir():
            return options
        model_files = sorted(models_dir.glob("*.zip"))
        for model_file in model_files:
            options[model_file.name] = str(model_file)

        return options
    
    def on_start_clicked(self):
        selected_model_label = self.model_combo.currentText()
        selected_model_path = self._model_options.get(selected_model_label, "")

        config = {
            'num_drones': self.drone_spinbox.value(),
            'num_obstacles': self.obstacle_spinbox.value(),
            'max_steps': self.steps_spinbox.value(),
            'render_mode': self.render_combo.currentText(),
            'lidar': self.lidar_checkbox.isChecked(),
            'policy_model_path': selected_model_path,
        }
        self.switch_to_simulation(config)