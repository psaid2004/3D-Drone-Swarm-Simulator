import queue
import threading
import json
from datetime import datetime
from functools import partial
from pathlib import Path


from interface.environment_viewer import EnvironmentViewer
from interface.qt_terminal import DualTerminal

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    )


class _NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()

from envs.createWorld import SUPPORTED_SHAPES 
# Constants for object configuration
KINDS = ("static", "dynamic")
AXIS_LABEL_TO_INT = {"x": 0, "y": 1, "z": 2}
AXIS_INT_TO_LABEL = {0: "x", 1: "y", 2: "z"}
LEFT_SIDEBAR_WIDTH = 400


class SimulationViewer(QWidget):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.console = None
        self._console_visible = False

        # Initialize all attributes
        self._object_configs = []
        self._config_lock = threading.Lock()
        self._running = False
        self._episode = 0
        self._step = 0
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._state_queue = queue.Queue(maxsize=1)
        self._sim_thread = None
        self._is_scrubbing = False
        self._is_replay_playing = False
        self._last_live_state = {"episode": 0, "step": 0, "drones": 0, "obstacles": 0}
        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._advance_replay)

        # Initialize with default objects
        default_objects = [
            {
                "name": "Static Box",
                "kind": "static",
                "shape": "box",
                "x": 2.0,
                "y": 0.0,
                "z": 0.5,
                "size": 0.5,
                "yaw": 0.0,
                "axis": 0,
                "min": -2.0,
                "max": 2.0,
                "speed": 0.2,
                "direction": 1
            },
            {
                "name": "Dynamic Sphere",
                "kind": "dynamic",
                "shape": "sphere",
                "x": -2.0,
                "y": 0.0,
                "z": 0.5,
                "size": 0.5,
                "yaw": 0.0,
                "axis": 1,
                "min": -2.0,
                "max": 2.0,
                "speed": 0.3,
                "direction": -1
            }
        ]
        
        # Use config if provided, otherwise use defaults
        if config and config.get('num_obstacles', 0) > 0:
            for i in range(config.get('num_obstacles', 2)):
                obj = default_objects[i % len(default_objects)].copy()
                obj["name"] = f"Object {i + 1}"
                self._object_configs.append(obj)
        else:
            self._object_configs = default_objects.copy()
        
        # Start a timer to consume state updates
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._consume_state_updates)
        self._update_timer.start(100)
        
        self._build_layout()

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._consume_state_updates)
        self._ui_timer.start(50)

        self._env_viewer = EnvironmentViewer(
            object_configs_getter=self._get_object_configs_snapshot,
            on_state=self._enqueue_state,
            on_stopped=self._enqueue_stopped,
            on_error=self._enqueue_error,
            num_drones=int(self.config.get("num_drones", 8)),
            max_steps=int(self.config.get("max_steps", 5000)),
            lidar=bool(self.config.get("lidar", False)),
            render_mode=str(self.config.get("render_mode", "human")),
            model_path=str(self.config.get("model_path", "")),
        )
    ####### UI Building Methods #######


    def _build_layout(self):
        root = QWidget()
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().addWidget(root)
        root_layout = QVBoxLayout(root)
        self._root = root
        self._root_layout = root_layout
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        top_bar = QFrame()
        top_bar.setFrameShape(QFrame.Shape.Box)
        top_bar.setFixedHeight(52)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("Simulation Controls")
        title.setStyleSheet("font-weight: 700;")

        self.status_lbl = QLabel("Stopped")

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedWidth(90)
        self.start_btn.clicked.connect(self._start)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedWidth(90)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)

        self.console_btn = QPushButton("Open console")
        self.console_btn.setFixedWidth(120)
        self.console_btn.clicked.connect(self._toggle_console)

        self.change_model_btn = QPushButton("Change model")
        self.change_model_btn.setFixedWidth(120)
        self.change_model_btn.clicked.connect(self._change_model)

        self.save_config_btn = QPushButton("Save config")
        self.save_config_btn.setFixedWidth(120)
        self.save_config_btn.clicked.connect(self._save_config)

        self.load_config_btn = QPushButton("Load config")
        self.load_config_btn.setFixedWidth(120)
        self.load_config_btn.clicked.connect(self._load_config)

        timeline_frame = QFrame()
        timeline_frame.setFrameShape(QFrame.Shape.Box)
        timeline_layout = QHBoxLayout(timeline_frame)
        timeline_layout.setContentsMargins(12, 6, 12, 6)

        timeline_scroll = QScrollArea()
        timeline_scroll.setWidgetResizable(True)
        timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        timeline_scroll.setWidget(timeline_frame)
        
        self.time_start_lbl = QLabel("00:00")
        self.time_now_lbl = QLabel("00:00")

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setTracking(True)
        self.timeline_slider.sliderPressed.connect(self._on_timeline_pressed)
        self.timeline_slider.sliderReleased.connect(self._on_timeline_released)
        self.timeline_slider.valueChanged.connect(self._on_timeline_value_changed)

        self.replay_btn = QPushButton("Replay")
        self.replay_btn.setFixedWidth(120)
        self.replay_btn.setEnabled(False)
        self.replay_btn.clicked.connect(self._toggle_replay)

        speed_title = QLabel("Speed")
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(10, 200)
        self.speed_spinbox.setSingleStep(10)
        self.speed_spinbox.setSuffix("%")
        self.speed_spinbox.setValue(100)
        self.speed_spinbox.valueChanged.connect(self._on_speed_changed)

        top_layout.addWidget(title)
        top_layout.addStretch(1)
        timeline_layout.addWidget(self.time_start_lbl)
        timeline_layout.addWidget(self.timeline_slider, 1)
        timeline_layout.addWidget(self.time_now_lbl)
        top_layout.addWidget(self.status_lbl)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.change_model_btn)
        top_layout.addSpacing(10)
        top_layout.addWidget(speed_title)
        top_layout.addWidget(self.speed_spinbox)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.console_btn)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.replay_btn)
        top_layout.addWidget(self.stop_btn)
        top_layout.addWidget(self.start_btn)
        top_layout.addWidget(self.save_config_btn)
        top_layout.addWidget(self.load_config_btn)
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        self.body_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.body_splitter.setHandleWidth(8)

        self.left_sidebar = QFrame()
        self.left_sidebar.setFrameShape(QFrame.Shape.Box)
        self.left_sidebar.setMinimumWidth(0)
        self.left_sidebar.setMaximumWidth(LEFT_SIDEBAR_WIDTH)
        left_layout = QVBoxLayout(self.left_sidebar)
        left_layout.setContentsMargins(12, 12, 12, 10)

        side_title = QLabel("Objects")
        side_title.setStyleSheet("font-weight: 700;")

        count_label = QLabel("Object count")
        self.object_spinbox = QSpinBox()
        self.object_spinbox.setRange(0, 100) 
        self.object_spinbox.setValue(len(self._object_configs))
        self.object_spinbox.valueChanged.connect(self._on_object_spinbox)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_host = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_host)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(8)
        self.scroll_area.setWidget(self.scroll_host)

        left_layout.addWidget(side_title)
        left_layout.addWidget(count_label)
        left_layout.addWidget(self.object_spinbox)
        left_layout.addWidget(self.scroll_area)

        self.info_panel = QFrame()
        self.info_panel.setFrameShape(QFrame.Shape.Box)
        info_layout = QVBoxLayout(self.info_panel)
        info_layout.setContentsMargins(14, 14, 14, 12)

        panel_title = QLabel("PyBullet Live View")
        panel_title.setStyleSheet("font-weight: 700;")

        panel_text = QLabel(
            "The sidebar controls each object individually: name, type, shape, position, size, yaw, and dynamic motion.\n"
            "Any changes while running automatically restart the episode with the new object setup."
        )
        panel_text.setWordWrap(True)

        self.telemetry_lbl = QLabel(self._telemetry_text(episode=0, step=0, drones=0, obstacles=len(self._object_configs)))
        self.telemetry_lbl.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")

        info_layout.addWidget(panel_title)
        info_layout.addWidget(panel_text)
        info_layout.addWidget(self.telemetry_lbl)
        info_layout.addStretch(1)

        self.body_splitter.addWidget(self.left_sidebar)
        self.body_splitter.addWidget(self.info_panel)
        self.body_splitter.setCollapsible(0, True)
        self.body_splitter.setCollapsible(1, False)
        self.body_splitter.setStretchFactor(0, 0)
        self.body_splitter.setStretchFactor(1, 1)
        self.body_splitter.setSizes([LEFT_SIDEBAR_WIDTH, max(600, self.width() - LEFT_SIDEBAR_WIDTH)])

        body_layout.addWidget(self.body_splitter)

        root_layout.addWidget(top_bar)
        root_layout.addWidget(timeline_scroll)
        root_layout.addWidget(body, 1)

        self._rebuild_object_controls()

    def _add_labeled_entry(self, parent: QWidget, label: str, value, on_apply):
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel(label)
        title.setFixedWidth(38)
        entry = QLineEdit(str(value))

        entry.returnPressed.connect(lambda e=entry: on_apply(e.text()))
        entry.editingFinished.connect(lambda e=entry: on_apply(e.text()))

        layout.addWidget(title)
        layout.addWidget(entry, 1)
        return row

    def _add_option_menu(self, parent: QWidget, label: str, current: str, options, on_apply):
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel(label)
        title.setFixedWidth(38)
        combo = _NoScrollComboBox()
        combo.addItems(list(options))
        combo.setCurrentText(str(current))
        combo.currentTextChanged.connect(on_apply)

        layout.addWidget(title)
        layout.addWidget(combo, 1)
        return row

    def _rebuild_object_controls(self):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        with self._config_lock:
            configs = [dict(cfg) for cfg in self._object_configs]

        if not configs:
            empty = QLabel("No objects configured")
            empty.setStyleSheet("color: #666;")
            self.scroll_layout.addWidget(empty)
            self.scroll_layout.addStretch(1)
            return

        for i, raw_cfg in enumerate(configs):
            # Ensure config has all required fields
            cfg = self._ensure_config_fields(raw_cfg)
            
            card = QGroupBox(f"Object {i + 1}")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(4)

            header_row = QWidget(card)
            header_layout = QHBoxLayout(header_row)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(0)
            header_layout.addStretch(1)
            header_layout.addWidget(self._remove_object_button(i), 0, Qt.AlignmentFlag.AlignRight)
            card_layout.addWidget(header_row)
            # Name field
            card_layout.addWidget(self._add_labeled_entry(
                card, "Name", cfg["name"], 
                partial(self._set_object_field, i, "name")
            ))
            
            # Type dropdown
            card_layout.addWidget(self._add_option_menu(
                card, "Type", cfg["kind"], KINDS, 
                partial(self._set_object_field, i, "kind")
            ))
            
            # Shape dropdown
            card_layout.addWidget(self._add_option_menu(
                card, "Shape", cfg["shape"], SUPPORTED_SHAPES, 
                partial(self._set_object_field, i, "shape")
            ))

            # Position fields
            card_layout.addWidget(self._add_labeled_entry(
                card, "X", cfg["x"], 
                partial(self._set_object_field, i, "x")
            ))
            card_layout.addWidget(self._add_labeled_entry(
                card, "Y", cfg["y"], 
                partial(self._set_object_field, i, "y")
            ))

            card_layout.addWidget(self._add_labeled_entry(
                card, "Z", cfg["z"], 
                partial(self._set_object_field, i, "z")
            ))
            
            # Size and Yaw
            card_layout.addWidget(self._add_labeled_entry(
                card, "Size", cfg["size"], 
                partial(self._set_object_field, i, "size")
            ))
            card_layout.addWidget(self._add_labeled_entry(
                card, "Yaw", cfg["yaw"], 
                partial(self._set_object_field, i, "yaw")
            ))

            # Dynamic motion fields
            axis_label = AXIS_INT_TO_LABEL.get(int(cfg["axis"]), "x")
            card_layout.addWidget(self._add_option_menu(
                card, "Axis", axis_label, ("x", "y", "z"), 
                partial(self._set_object_field, i, "axis")
            ))
            card_layout.addWidget(self._add_labeled_entry(
                card, "Min", cfg["min"], 
                partial(self._set_object_field, i, "min")
            ))
            card_layout.addWidget(self._add_labeled_entry(
                card, "Max", cfg["max"], 
                partial(self._set_object_field, i, "max")
            ))
            card_layout.addWidget(self._add_labeled_entry(
                card, "Speed", cfg["speed"], 
                partial(self._set_object_field, i, "speed")
            ))
            card_layout.addWidget(self._add_labeled_entry(
                card, "Dir", cfg["direction"], 
                partial(self._set_object_field, i, "direction")
            ))
            self.scroll_layout.addWidget(card)
        self.scroll_layout.addStretch(1)

    def _remove_object_button(self, index: int):
        btn = QPushButton("X")
        btn.setFixedSize(11, 11)
        btn.setToolTip("Remove object")
        btn.setStyleSheet("""
            QPushButton {
                background-color: #ff0000;
                color: white;
                font-size: 5px;
                font-weight: 700;
                padding: 0px;
                border-radius: 6px;
            }""")
        btn.clicked.connect(lambda: self._remove_object(index))
        return btn                      
    
    def _telemetry_text(self, episode: int, step: int, drones: int, obstacles: int) -> str:
        return (
            f"Episode     : {episode}\n"
            f"Step        : {step}\n"
            f"Live drones : {drones}\n"
            f"Obstacles   : {obstacles}\n"
        )

######################################################################################################################################################################################################################
######################################################################################################################################################################################################################
################################################################################# Methodds ##################################################################################################################################### 
##################################################################################################################################################################################################################### 
###################################################################################################################################################################################################################### 

    def _enqueue_state(self, state):
        while not self._state_queue.empty():
            try:
                self._state_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._state_queue.put_nowait(state)
        except queue.Full:
            pass

    def _enqueue_stopped(self):
        self._enqueue_state({"stopped": True})

    def _enqueue_error(self, error_text: str):
        self._enqueue_state({"error": error_text})

    def _coerce_float(self, text: str, fallback: float):
        try:
            return float(text)
        except Exception:
            return float(fallback)

    def _coerce_int(self, text: str, fallback: int):
        try:
            return int(float(text))
        except Exception:
            return int(fallback)

    def _consume_state_updates(self):
        latest = None
        while not self._state_queue.empty():
            try:
                latest = self._state_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return

        if latest.get("stopped"):
            if self._running:
                self._running = False
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.status_lbl.setText("Stopped")
            self._refresh_replay_controls()
            return

        if "error" in latest:
            self.status_lbl.setText(f"Error: {latest['error']}")
            self._stop()
            return

        self._update_telemetry(latest)
        self._refresh_replay_controls()

    def _update_telemetry(self, state):
        self._last_live_state = {
            "episode": int(state.get("episode", self._last_live_state.get("episode", 0))),
            "step": int(state.get("step", self._last_live_state.get("step", 0))),
            "drones": int(state.get("drones", self._last_live_state.get("drones", 0))),
            "obstacles": int(state.get("obstacles", self._last_live_state.get("obstacles", 0))),
        }

        if self._running and not self._is_scrubbing:
            replay_index = int(state.get("replay_index", self._env_viewer.get_replay_index()))
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(replay_index)
            self.timeline_slider.blockSignals(False)

        speed = float(state.get("speed_multiplier", self._env_viewer.get_speed_multiplier()))
        self.telemetry_lbl.setText(
            self._telemetry_text(
                episode=int(self._last_live_state["episode"]),
                step=int(self._last_live_state["step"]),
                drones=int(self._last_live_state["drones"]),
                obstacles=int(self._last_live_state["obstacles"]),
            )
            + f"Speed       : {speed:.2f}x\n"
        )

    def _toggle_console(self):
        if self.console is None:
            self.console = DualTerminal(self._root)
            self._root_layout.addWidget(self.console)

        self._console_visible = not self._console_visible
        self.console.setVisible(self._console_visible)
        self.console_btn.setText("Hide console" if self._console_visible else "Open console")

    def _change_model(self):
        models_dir = Path(__file__).resolve().parents[1] / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select policy model",
            str(models_dir),
            "Model Files (*.zip);;All Files (*)",
        )
        if not file_name:
            return

        ok, message = self._env_viewer.set_policy_model_path(file_name)
        if not ok:
            self.status_lbl.setText(f"Error: {message}")
            return

        self.config["policy_model_path"] = file_name
        model_name = Path(file_name).name
        self.status_lbl.setText(f"Model changed to {model_name}")
        if self._running:
            self._request_restart(f"Restarting with model: {model_name}")



    ## Control methods
    def _start(self):
        if self._running:
            return

        self._is_replay_playing = False
        self._replay_timer.stop()
        self.replay_btn.setText("Play Replay")
        self.replay_btn.setEnabled(False)
        self._running = True
        self._on_speed_changed(int(self.speed_spinbox.value()))
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        mode = str(self.config.get("render_mode", "human"))
        mode_label = "Headless capture" if mode == "none" else "PyBullet GUI"
        self.status_lbl.setText(f"Running ({mode_label}) @ {self._env_viewer.get_speed_multiplier():.2f}x")
        self._env_viewer.start()

    def _stop(self):
        if not self._running:
            return

        self._running = False
        self._env_viewer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("Stopped")
        self._refresh_replay_controls()

    def closeEvent(self, event):
        self._replay_timer.stop()
        self._env_viewer.close()
        super().closeEvent(event)

    def _request_restart(self, message: str):
        if self._running:
            self._env_viewer.request_restart()
            self.status_lbl.setText(message)

    def go_back(self):
        main_window = self.window()
        if hasattr(main_window, "switch_to_landing"):
            main_window.switch_to_landing()

    ## Control methods


    ### Object config methods

    def _remove_object(self, index: int):
        object_name = self._object_name(index)
        answer = QMessageBox.question(
            self,
            "Confirm deletion",
            f"Remove {object_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        with self._config_lock:
            if 0 <= index < len(self._object_configs):
                del self._object_configs[index]
            else:
                return

        self.object_spinbox.blockSignals(True)
        self.object_spinbox.setValue(len(self._object_configs))
        self.object_spinbox.blockSignals(False)
        self._rebuild_object_controls()
        self._request_restart(f"Restarting after removing {object_name}...")

    def _get_object_configs_snapshot(self):
        with self._config_lock:
            return [dict(cfg) for cfg in self._object_configs]

    def _default_object_config(self, index: int):
        return {
            "name": f"Object {index + 1}",
            "kind": "static",
            "shape": "box",
            "x": round((index % 4) * 0.8 - 1.2, 2),
            "y": round((index // 4) * 0.8 - 1.2, 2),
            "z": 0.8,
            "size": 1.0,
            "yaw": 0.0,
            "axis": 0,
            "min": -1.0,
            "max": 1.0,
            "speed": 0.2,
            "direction": 1,
        }

    def _on_speed_changed(self, value: int):
        return value

    def _on_object_spinbox(self, value: int):
        """Handle spin box value changes"""
        self._sync_object_config_count(int(value))
        self._rebuild_object_controls()
        self._request_restart(f"Restarting with updated object count: {value}")
        
    def _ensure_config_fields(self, config):
        """Ensure config has all required fields with defaults"""
        defaults = {
            "name": "Object",
            "kind": "static",
            "shape": "box",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "size": 0.5,
            "yaw": 0.0,
            "axis": 0,
            "min": -1.0,
            "max": 1.0,
            "speed": 0.2,
            "direction": 1
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        return config

    def _sync_object_config_count(self, count: int):
        current_count = len(self._object_configs)
        if count > current_count:
            # Add new configs with all default fields
            for i in range(current_count, count):
                new_config = {
                    "name": f"Object {i + 1}",
                    "kind": "static",
                    "shape": "box",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "size": 0.5,
                    "yaw": 0.0,
                    "axis": 0,
                    "min": -1.0,
                    "max": 1.0,
                    "speed": 0.2,
                    "direction": 1
                }
                self._object_configs.append(new_config)
        elif count < current_count:
            # Remove excess configs
            self._object_configs = self._object_configs[:count]

    def _set_object_field(self, index: int, key: str, raw_value):
        with self._config_lock:
            if not (0 <= index < len(self._object_configs)):
                return
            old = self._object_configs[index].get(key)
            if isinstance(old, float):
                value = self._coerce_float(raw_value, old)
            elif isinstance(old, int):
                value = self._coerce_int(raw_value, old)
            else:
                value = str(raw_value).strip() if str(raw_value).strip() else str(old)
            
            if key == "shape" and value not in SUPPORTED_SHAPES:
                value = old
            if key == "kind" and value not in KINDS:
                value = old
            if key == "axis":
                if isinstance(value, str):
                    value = AXIS_LABEL_TO_INT.get(value, old)
                value = max(0, min(2, int(value)))
            if key == "size":
                value = max(0.1, float(value))
            if key == "speed":
                value = max(0.0, float(value))
            if key == "direction":
                value = -1 if int(value) < 0 else 1

            self._object_configs[index][key] = value

        self._request_restart(f"Restarting after updating {self._object_name(index)}...")

    def _object_name(self, index: int) -> str:
        with self._config_lock:
            if 0 <= index < len(self._object_configs):
                return str(self._object_configs[index].get("name", f"Object {index + 1}"))
        return f"Object {index + 1}"

    def _save_config(self):
        self._object_configs = self._get_object_configs_snapshot()
        configs_dir = Path(__file__).resolve().parent / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = configs_dir / f"obj_{timestamp}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(self._object_configs, f, indent=4)

    def _load_config(self):
        configs_dir = Path(__file__).resolve().parent / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load object config",
            str(configs_dir),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_name:
            return
        file_path = Path(file_name)
        if not file_path.is_file():
            return
        try:
            with file_path.open("r", encoding="utf-8") as f:
                configs = json.load(f)
            if isinstance(configs, list):
                normalized_configs = [self._ensure_config_fields(dict(cfg)) for cfg in configs if isinstance(cfg, dict)]
                with self._config_lock:
                    self._object_configs = normalized_configs
                self.object_spinbox.blockSignals(True)
                self.object_spinbox.setValue(len(normalized_configs))
                self.object_spinbox.blockSignals(False)
                self._rebuild_object_controls()
                self.telemetry_lbl.setText(
                    self._telemetry_text(
                        episode=0,
                        step=0,
                        drones=0,
                        obstacles=len(normalized_configs),) + f"Speed: {self._env_viewer.get_speed_multiplier():.2f}x\n"
                )

                if self._running:
                    self._request_restart(f"Restarting with: {file_path.name}")
                else:
                    self.status_lbl.setText(f"Loaded config: {file_path.name}")
            else:
                self.status_lbl.setText("Load failed")
        except Exception as exc:
            print(f"Failed to load config from {file_path}: {exc}")
            self.status_lbl.setText("Load failed")


    ### Object config methods
            

    ### Replay methods 

    def _format_clock(self, seconds: float) -> str:
        total = max(0, int(seconds))
        minutes = total // 60
        secs = total % 60
        return f"{minutes:02d}:{secs:02d}"

    def _refresh_replay_controls(self):
        frame_count = self._env_viewer.get_replay_frame_count()
        max_index = max(0, frame_count - 1)
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMaximum(max_index)
        self.timeline_slider.setEnabled(frame_count > 0)
        self.timeline_slider.blockSignals(False)

        self.replay_btn.setEnabled((not self._running) and frame_count > 1)

        current_index = min(self.timeline_slider.value(), max_index)
        if current_index != self.timeline_slider.value():
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(current_index)
            self.timeline_slider.blockSignals(False)

        total_secs = max_index / 30.0
        now_secs = current_index / 30.0
        self.time_start_lbl.setText("00:00")
        self.time_now_lbl.setText(f"{self._format_clock(now_secs)} / {self._format_clock(total_secs)}")

    def _seek_to_frame(self, frame_index: int):
        if self._running:
            return

        if not self._env_viewer.seek_replay_frame(frame_index):
            return

        meta = self._env_viewer.get_replay_frame_meta(frame_index) or {}
        replay_state = dict(self._last_live_state)
        replay_state["episode"] = int(meta.get("episode", replay_state.get("episode", 0)))
        replay_state["step"] = int(meta.get("step", replay_state.get("step", 0)))
        replay_state["speed_multiplier"] = self._env_viewer.get_speed_multiplier()
        self._update_telemetry(replay_state)
        self._refresh_replay_controls()

    def _on_timeline_pressed(self):
        self._is_scrubbing = True

    def _on_timeline_released(self):
        self._is_scrubbing = False
        self._seek_to_frame(self.timeline_slider.value())

    def _on_timeline_value_changed(self, value: int):
        if self._running:
            return
        self._seek_to_frame(int(value))

    def _toggle_replay(self):
        if self._running:
            return

        frame_count = self._env_viewer.get_replay_frame_count()
        if frame_count <= 1:
            return

        self._is_replay_playing = not self._is_replay_playing
        if self._is_replay_playing:
            self.replay_btn.setText("Pause Replay")
            self._on_speed_changed(int(self.speed_spinbox.value()))
            if self._replay_timer.interval() <= 0:
                self._replay_timer.setInterval(33)
            self._replay_timer.start()
            self.status_lbl.setText("Replay playing")
        else:
            self._replay_timer.stop()
            self.replay_btn.setText("Play Replay")
            self.status_lbl.setText("Replay paused")

    def _advance_replay(self):
        if self._running:
            self._replay_timer.stop()
            return

        max_index = max(0, self._env_viewer.get_replay_frame_count() - 1)
        if max_index <= 0:
            self._replay_timer.stop()
            self._is_replay_playing = False
            self.replay_btn.setText("Play Replay")
            return

        next_index = self.timeline_slider.value() + 1
        if next_index > max_index:
            next_index = 0

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(next_index)
        self.timeline_slider.blockSignals(False)
        self._seek_to_frame(next_index)

    ### Replay methopds