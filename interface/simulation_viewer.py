# -*- coding: utf-8 -*-
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
SIDEBAR_MODES = ("object_spawns", "drone_spawns", "target_spawns")
LEFT_SIDEBAR_WIDTH = 550
SWARM_COLOR_PRESETS = {
    "Green": [0.0, 1.0, 0.0, 0.9],
    "Blue": [0.0, 0.0, 1.0, 0.9],
    "Yellow": [1.0, 1.0, 0.0, 0.9],
    "Magenta": [1.0, 0.0, 1.0, 0.9],
    "Cyan": [0.0, 1.0, 1.0, 0.9],
    "Orange": [1.0, 0.5, 0.0, 0.9],
    "Purple": [0.5, 0.0, 1.0, 0.9],
}


class SimulationViewer(QWidget):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.console = None
        self._console_visible = False

        # Initialize all attributes

        # Spawn attributes
        self._object_configs = []
        self._drone_spawn_configs = []
        self._swarm_configs = []
        self._expanded_swarms = set()
        self._target_spawn_configs = []
        self._sidebar_mode = "object_spawns"
        self._num_swarms = max(1, int(self.config.get("num_swarms", 1)))
        self._drones_per_swarm = max(1, int(self.config.get("num_drones", 3)))

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

        # Initialize with default object
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
                "direction": 1,
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
                "direction": -1,
            },
        ]
        # Use config if provided, otherwise use defaults
        if config and config.get("num_obstacles", 0) >= 0:
            for i in range(config.get("num_obstacles", 0)):
                obj = default_objects[i % len(default_objects)].copy()
                obj["name"] = f"Object {i + 1}"
                self._object_configs.append(obj)
        else:
            self._object_configs = default_objects.copy()

        self._init_swarm_spawn_configs()
        self._target_spawn_configs = [
            self._default_target_spawn_config(i)
            for i in range(max(1, self._num_swarms))
        ]

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
            drone_spawn_configs_getter=self._get_drone_spawn_configs_snapshot,
            target_spawn_configs_getter=self._get_target_spawn_configs_snapshot,
            on_state=self._enqueue_state,
            on_stopped=self._enqueue_stopped,
            on_error=self._enqueue_error,
            num_drones=self._drones_per_swarm,
            max_steps=int(self.config.get("max_steps", 5000)),
            lidar=bool(self.config.get("lidar", False)),
            render_mode=str(self.config.get("render_mode", "human")),
            model_path=str(self.config.get("model_path", "")),
            world_scale=float(self.config.get("world_scale", 5)),
            num_swarms=self._num_swarms,
            seed=self.config.get("seed", None),
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
        self.start_btn.setFixedWidth(100)
        self.start_btn.clicked.connect(self._start)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedWidth(100)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)

        self.console_btn = QPushButton("Open console")
        self.console_btn.setFixedWidth(100)
        self.console_btn.clicked.connect(self._toggle_console)

        self.change_model_btn = QPushButton("Change model")
        self.change_model_btn.setFixedWidth(100)
        self.change_model_btn.clicked.connect(self._change_model)

        self.save_config_btn = QPushButton("Save config")
        self.save_config_btn.setFixedWidth(100)
        self.save_config_btn.clicked.connect(self._save_config)

        self.load_config_btn = QPushButton("Load config")
        self.load_config_btn.setFixedWidth(100)
        self.load_config_btn.clicked.connect(self._load_config)

        self.back_btn = QPushButton("Back to config")
        self.back_btn.setFixedWidth(100)
        self.back_btn.clicked.connect(self.go_back)

        timeline_frame = QFrame()
        timeline_frame.setFrameShape(QFrame.Shape.Box)
        timeline_layout = QHBoxLayout(timeline_frame)
        timeline_layout.setContentsMargins(12, 6, 12, 6)

        timeline_scroll = QScrollArea()
        timeline_scroll.setWidgetResizable(True)
        timeline_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        timeline_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
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
        top_layout.addWidget(self.back_btn)
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
        self.side_title = side_title

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)

        self.mode_btn_objects = QPushButton("Obstacles")
        self.mode_btn_drones = QPushButton("Drones")
        self.mode_btn_targets = QPushButton("Targets")
        for btn in (self.mode_btn_objects, self.mode_btn_drones, self.mode_btn_targets):
            btn.setCheckable(True)
            mode_layout.addWidget(btn)
        self.mode_btn_objects.clicked.connect(
            lambda: self._set_sidebar_mode("object_spawns")
        )
        self.mode_btn_drones.clicked.connect(
            lambda: self._set_sidebar_mode("drone_spawns")
        )
        self.mode_btn_targets.clicked.connect(
            lambda: self._set_sidebar_mode("target_spawns")
        )

        count_label = QLabel("Object count")
        self.count_label = count_label
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
        left_layout.addWidget(mode_row)
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
            "Any changes while running automatically restart the episode with the new object setup.\n"
            "Statistics:"
        )
        panel_text.setWordWrap(True)

        self.telemetry_lbl = QLabel(
            self._telemetry_text(
                episode=0, step=0, drones=0, obstacles=len(self._object_configs)
            )
        )
        self.telemetry_lbl.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace;"
        )

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
        self.body_splitter.setSizes(
            [LEFT_SIDEBAR_WIDTH, max(600, self.width() - LEFT_SIDEBAR_WIDTH)]
        )

        body_layout.addWidget(self.body_splitter)

        root_layout.addWidget(top_bar)
        root_layout.addWidget(timeline_scroll)
        root_layout.addWidget(body, 1)

        self._set_sidebar_mode("object_spawns")

    def _add_labeled_entry(self, parent: QWidget, label: str, value, on_apply):
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel(label)
        title.setFixedWidth(70)
        entry = QLineEdit(str(value))

        entry.returnPressed.connect(lambda e=entry: on_apply(e.text()))
        entry.editingFinished.connect(lambda e=entry: on_apply(e.text()))

        layout.addWidget(title)
        layout.addWidget(entry, 1)
        return row

    def _add_option_menu(
        self, parent: QWidget, label: str, current: str, options, on_apply
    ):
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel(label)
        title.setFixedWidth(70)
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
            if self._sidebar_mode == "object_spawns":
                configs = [dict(cfg) for cfg in self._object_configs]
            elif self._sidebar_mode == "drone_spawns":
                configs = [dict(cfg) for cfg in self._drone_spawn_configs]
            else:
                configs = [dict(cfg) for cfg in self._target_spawn_configs]

        if not configs:
            empty = QLabel("No entries configured")
            empty.setStyleSheet("color: #666;")
            self.scroll_layout.addWidget(empty)
            self.scroll_layout.addStretch(1)
            return

        if self._sidebar_mode == "drone_spawns":
            with self._config_lock:
                swarms = [dict(cfg) for cfg in self._swarm_configs]
                drones = [dict(cfg) for cfg in self._drone_spawn_configs]

            for swarm_idx, swarm_cfg in enumerate(swarms):
                card = QGroupBox(f"Swarm {swarm_idx + 1}")
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(8, 8, 8, 8)
                card_layout.setSpacing(6)

                toggle_btn = QPushButton(
                    "Collapse drones"
                    if swarm_idx in self._expanded_swarms
                    else "Expand drones"
                )
                toggle_btn.clicked.connect(
                    partial(self._toggle_swarm_expand, swarm_idx)
                )
                card_layout.addWidget(toggle_btn)

                card_layout.addWidget(
                    self._add_labeled_entry(
                        card,
                        "Name",
                        swarm_cfg.get("name", f"Swarm {swarm_idx + 1}"),
                        partial(self._set_swarm_field, swarm_idx, "name"),
                    )
                )
                card_layout.addWidget(
                    self._add_labeled_entry(
                        card,
                        "Center X",
                        swarm_cfg.get("x", 0.0),
                        partial(self._set_swarm_field, swarm_idx, "x"),
                    )
                )
                card_layout.addWidget(
                    self._add_labeled_entry(
                        card,
                        "Center Y",
                        swarm_cfg.get("y", 0.0),
                        partial(self._set_swarm_field, swarm_idx, "y"),
                    )
                )
                card_layout.addWidget(
                    self._add_labeled_entry(
                        card,
                        "Center Z",
                        swarm_cfg.get("z", 0.8),
                        partial(self._set_swarm_field, swarm_idx, "z"),
                    )
                )
                card_layout.addWidget(
                    self._add_option_menu(
                        card,
                        "Color",
                        str(swarm_cfg.get("color_name", "Green")),
                        tuple(SWARM_COLOR_PRESETS.keys()),
                        partial(self._set_swarm_field, swarm_idx, "color_name"),
                    )
                )

                if swarm_idx in self._expanded_swarms:
                    for drone_local_idx in range(self._drones_per_swarm):
                        drone_index = (
                            swarm_idx * self._drones_per_swarm + drone_local_idx
                        )
                        if not (0 <= drone_index < len(drones)):
                            continue
                        drone_cfg = drones[drone_index]

                        drone_card = QGroupBox(f"Drone {drone_local_idx + 1}")
                        drone_layout = QVBoxLayout(drone_card)
                        drone_layout.setContentsMargins(8, 8, 8, 8)
                        drone_layout.setSpacing(4)

                        drone_layout.addWidget(
                            self._add_labeled_entry(
                                drone_card,
                                "Offset X",
                                drone_cfg.get("local_x", 0.0),
                                partial(
                                    self._set_drone_offset_field, drone_index, "local_x"
                                ),
                            )
                        )
                        drone_layout.addWidget(
                            self._add_labeled_entry(
                                drone_card,
                                "Offset Y",
                                drone_cfg.get("local_y", 0.0),
                                partial(
                                    self._set_drone_offset_field, drone_index, "local_y"
                                ),
                            )
                        )
                        drone_layout.addWidget(
                            self._add_labeled_entry(
                                drone_card,
                                "Offset Z",
                                drone_cfg.get("local_z", 0.0),
                                partial(
                                    self._set_drone_offset_field, drone_index, "local_z"
                                ),
                            )
                        )
                        card_layout.addWidget(drone_card)

                self.scroll_layout.addWidget(card)

            self.scroll_layout.addStretch(1)
            return

        if self._sidebar_mode == "target_spawns":
            for i, raw_cfg in enumerate(configs):
                cfg = self._ensure_spawn_fields(raw_cfg, "Target")
                card = QGroupBox(f"Target {i + 1}")
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(8, 8, 8, 8)
                card_layout.setSpacing(4)

                card_layout.addWidget(
                    self._add_labeled_entry(
                        card,
                        "Name",
                        cfg["name"],
                        partial(self._set_object_field, i, "name"),
                    )
                )
                card_layout.addWidget(
                    self._add_labeled_entry(
                        card,
                        "Coordinate X",
                        cfg["x"],
                        partial(self._set_object_field, i, "x"),
                    )
                )
                card_layout.addWidget(
                    self._add_labeled_entry(
                        card,
                        "Coordinate Y",
                        cfg["y"],
                        partial(self._set_object_field, i, "y"),
                    )
                )
                card_layout.addWidget(
                    self._add_labeled_entry(
                        card,
                        "Coordinate Z",
                        cfg["z"],
                        partial(self._set_object_field, i, "z"),
                    )
                )
                self.scroll_layout.addWidget(card)

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
            header_layout.addWidget(
                self._remove_object_button(i), 0, Qt.AlignmentFlag.AlignRight
            )
            card_layout.addWidget(header_row)
            # Name field
            card_layout.addWidget(
                self._add_labeled_entry(
                    card,
                    "Name",
                    cfg["name"],
                    partial(self._set_object_field, i, "name"),
                )
            )

            # Type dropdown
            card_layout.addWidget(
                self._add_option_menu(
                    card,
                    "Type",
                    cfg["kind"],
                    KINDS,
                    partial(self._set_object_field, i, "kind"),
                )
            )

            # Shape dropdown
            card_layout.addWidget(
                self._add_option_menu(
                    card,
                    "Shape",
                    cfg["shape"],
                    SUPPORTED_SHAPES,
                    partial(self._set_object_field, i, "shape"),
                )
            )

            # Position fields
            card_layout.addWidget(
                self._add_labeled_entry(
                    card,
                    "Coordinate X",
                    cfg["x"],
                    partial(self._set_object_field, i, "x"),
                )
            )
            card_layout.addWidget(
                self._add_labeled_entry(
                    card,
                    "Coordinate Y",
                    cfg["y"],
                    partial(self._set_object_field, i, "y"),
                )
            )

            card_layout.addWidget(
                self._add_labeled_entry(
                    card,
                    "Coordinate Z",
                    cfg["z"],
                    partial(self._set_object_field, i, "z"),
                )
            )

            # Size and Yaw
            card_layout.addWidget(
                self._add_labeled_entry(
                    card,
                    "Size",
                    cfg["size"],
                    partial(self._set_object_field, i, "size"),
                )
            )
            card_layout.addWidget(
                self._add_labeled_entry(
                    card, "Yaw", cfg["yaw"], partial(self._set_object_field, i, "yaw")
                )
            )

            if cfg["kind"] != "dynamic":
                self.scroll_layout.addWidget(card)
                continue

            # Dynamic motion fields
            axis_label = AXIS_INT_TO_LABEL.get(int(cfg["axis"]), "x")
            card_layout.addWidget(
                self._add_option_menu(
                    card,
                    "Axis",
                    axis_label,
                    ("x", "y", "z"),
                    partial(self._set_object_field, i, "axis"),
                )
            )
            card_layout.addWidget(
                self._add_labeled_entry(
                    card, "Min", cfg["min"], partial(self._set_object_field, i, "min")
                )
            )
            card_layout.addWidget(
                self._add_labeled_entry(
                    card, "Max", cfg["max"], partial(self._set_object_field, i, "max")
                )
            )
            card_layout.addWidget(
                self._add_labeled_entry(
                    card,
                    "Speed",
                    cfg["speed"],
                    partial(self._set_object_field, i, "speed"),
                )
            )
            card_layout.addWidget(
                self._add_labeled_entry(
                    card,
                    "Dir",
                    cfg["direction"],
                    partial(self._set_object_field, i, "direction"),
                )
            )
            self.scroll_layout.addWidget(card)
        self.scroll_layout.addStretch(1)

    def _set_sidebar_mode(self, mode: str):
        if mode not in SIDEBAR_MODES:
            return
        self._sidebar_mode = mode

        self.mode_btn_objects.setChecked(mode == "object_spawns")
        self.mode_btn_drones.setChecked(mode == "drone_spawns")
        self.mode_btn_targets.setChecked(mode == "target_spawns")

        if mode == "object_spawns":
            self.side_title.setText("Objects")
            self.count_label.setText("Object count")
            count = len(self._object_configs)
            spin_enabled = True
        elif mode == "drone_spawns":
            self.side_title.setText("Swarms")
            self.count_label.setText("Swarm count")
            count = len(self._swarm_configs)
            spin_enabled = True
        else:
            self.side_title.setText("Targets")
            self.count_label.setText("Target count")
            count = len(self._target_spawn_configs)
            spin_enabled = True

        self.object_spinbox.blockSignals(True)
        self.object_spinbox.setEnabled(spin_enabled)
        self.object_spinbox.setValue(count)
        self.object_spinbox.blockSignals(False)
        self._rebuild_object_controls()

    def _ensure_spawn_fields(self, config, prefix: str):
        defaults = {
            "name": f"{prefix} count",
            "x": 0.0,
            "y": 0.0,
            "z": 0.5,
        }
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        return config

    def _remove_object_button(self, index: int):
        btn = QPushButton("X")
        btn.setFixedSize(11, 11)
        btn.setToolTip("Remove object")
        btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff0000;
                color: white;
                font-size: 5px;
                font-weight: 700;
                padding: 0px;
                border-radius: 6px;
            }"""
        )
        btn.clicked.connect(lambda: self._remove_object(index))
        return btn

    def _telemetry_text(
        self, episode: int, step: int, drones: int, obstacles: int
    ) -> str:
        return (
            f"Episode     : {episode}\n"
            f"Step        : {step}\n"
            f"Live drones : {drones}\n"
            f"Obstacles   : {obstacles}\n"
        )

    ######################################################################################################################################################################################################################
    ######################################################################################################################################################################################################################
    ################################################################################# Methods ###########################################################################################################################
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
            "episode": int(
                state.get("episode", self._last_live_state.get("episode", 0))
            ),
            "step": int(state.get("step", self._last_live_state.get("step", 0))),
            "drones": int(state.get("drones", self._last_live_state.get("drones", 0))),
            "obstacles": int(
                state.get("obstacles", self._last_live_state.get("obstacles", 0))
            ),
        }

        if self._running and not self._is_scrubbing:
            replay_index = int(
                state.get("replay_index", self._env_viewer.get_replay_index())
            )
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(replay_index)
            self.timeline_slider.blockSignals(False)

        speed = float(
            state.get("speed_multiplier", self._env_viewer.get_speed_multiplier())
        )
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
        self.console_btn.setText(
            "Hide console" if self._console_visible else "Open console"
        )

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

        ok, message = self._env_viewer.set_model_path(file_name)
        if not ok:
            self.status_lbl.setText(f"Error: {message}")
            return

        self.config["model_path"] = file_name
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
        self.status_lbl.setText(
            f"Running ({mode_label}) @ {self._env_viewer.get_speed_multiplier():.2f}x"
        )
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
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        with self._config_lock:
            if self._sidebar_mode == "object_spawns":
                if 0 <= index < len(self._object_configs):
                    del self._object_configs[index]
                else:
                    return
            elif self._sidebar_mode == "drone_spawns":
                if 0 <= index < len(self._drone_spawn_configs):
                    del self._drone_spawn_configs[index]
                else:
                    return
            else:
                if 0 <= index < len(self._target_spawn_configs):
                    del self._target_spawn_configs[index]
                else:
                    return

        self.object_spinbox.blockSignals(True)
        if self._sidebar_mode == "object_spawns":
            self.object_spinbox.setValue(len(self._object_configs))
        elif self._sidebar_mode == "drone_spawns":
            self.object_spinbox.setValue(len(self._drone_spawn_configs))
        else:
            self.object_spinbox.setValue(len(self._target_spawn_configs))
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

    def _default_drone_spawn_config(self, index: int):
        return {
            "name": f"Drone Spawn {index + 1}",
            "x": round((index % 3) * 0.9 - 0.1, 2),
            "y": round((index // 3) * 0.9 - 0.1, 2),
            "z": 0.8,
        }

    def _default_target_spawn_config(self, index: int):
        return {
            "name": f"Target Spawn {index + 1}",
            "x": 0.0,
            "y": 2.0,
            "z": 0.5,
        }

    # TODO: Add speed change
    def _on_speed_changed(self, value: int):
        return value

    def _on_object_spinbox(self, value: int):
        """Handle spin box value changes"""
        if self._sidebar_mode == "object_spawns":
            self._sync_object_config_count(int(value))
        else:
            self._set_swarm_target_count(int(value))
        self._rebuild_object_controls()
        self._request_restart(f"Restarting with updated count: {value}")

    def _set_swarm_target_count(self, count: int):
        count = max(1, int(count))
        with self._config_lock:
            self._num_swarms = count
            self.config["num_swarms"] = count

            current_swarm_count = len(self._swarm_configs)
            if count > current_swarm_count:
                for i in range(current_swarm_count, count):
                    self._swarm_configs.append(self._default_swarm_config(i))
            elif count < current_swarm_count:
                self._swarm_configs = self._swarm_configs[:count]
                self._expanded_swarms = {i for i in self._expanded_swarms if i < count}

            current_target_count = len(self._target_spawn_configs)
            if count > current_target_count:
                for i in range(current_target_count, count):
                    self._target_spawn_configs.append(
                        self._default_target_spawn_config(i)
                    )
            elif count < current_target_count:
                self._target_spawn_configs = self._target_spawn_configs[:count]

            for i, target_cfg in enumerate(self._target_spawn_configs):
                if "name" not in target_cfg:
                    target_cfg["name"] = f"Target Spawn {i + 1}"
                target_cfg["color"] = list(
                    self._swarm_configs[i].get("color", SWARM_COLOR_PRESETS["Green"])
                )

            self._rebuild_drone_spawn_configs_locked()

        self._env_viewer.set_num_swarms(count)

    def _default_swarm_color_name(self, index: int) -> str:
        names = list(SWARM_COLOR_PRESETS.keys())
        return names[index % len(names)]

    def _default_swarm_config(self, index: int):
        color_name = self._default_swarm_color_name(index)
        return {
            "name": f"Swarm {index + 1}",
            "x": round((index % 2) * 2.0 - 1.0, 2),
            "y": round((index // 2) * 2.0 - 1.0, 2),
            "z": 0.8,
            "color_name": color_name,
            "color": list(SWARM_COLOR_PRESETS[color_name]),
        }

    def _default_drone_offset(self, index: int):
        row = index // 3
        col = index % 3
        return {
            "local_x": round((col - 1) * 0.35, 2),
            "local_y": round((row - 1) * 0.35, 2),
            "local_z": 0.0,
        }

    def _init_swarm_spawn_configs(self):
        with self._config_lock:
            self._swarm_configs = [
                self._default_swarm_config(i) for i in range(self._num_swarms)
            ]
            self._expanded_swarms = set()
            self._drone_spawn_configs = []

            for swarm_idx in range(self._num_swarms):
                for drone_local_idx in range(self._drones_per_swarm):
                    offset_cfg = self._default_drone_offset(drone_local_idx)
                    self._drone_spawn_configs.append(
                        {
                            "name": f"Swarm {swarm_idx + 1} Drone {drone_local_idx + 1}",
                            "swarm_index": swarm_idx,
                            **offset_cfg,
                        }
                    )

            self._rebuild_drone_spawn_configs_locked()

    def _rebuild_drone_spawn_configs_locked(self):
        expected = self._num_swarms * self._drones_per_swarm

        if len(self._drone_spawn_configs) > expected:
            self._drone_spawn_configs = self._drone_spawn_configs[:expected]
        elif len(self._drone_spawn_configs) < expected:
            for idx in range(len(self._drone_spawn_configs), expected):
                swarm_idx = idx // self._drones_per_swarm
                drone_local_idx = idx % self._drones_per_swarm
                offset_cfg = self._default_drone_offset(drone_local_idx)
                self._drone_spawn_configs.append(
                    {
                        "name": f"Swarm {swarm_idx + 1} Drone {drone_local_idx + 1}",
                        "swarm_index": swarm_idx,
                        **offset_cfg,
                    }
                )

        for idx, cfg in enumerate(self._drone_spawn_configs):
            swarm_idx = int(cfg.get("swarm_index", idx // self._drones_per_swarm))
            swarm_idx = max(0, min(self._num_swarms - 1, swarm_idx))
            drone_local_idx = idx % self._drones_per_swarm
            swarm_cfg = self._swarm_configs[swarm_idx]

            local_x = float(cfg.get("local_x", 0.0))
            local_y = float(cfg.get("local_y", 0.0))
            local_z = float(cfg.get("local_z", 0.0))

            cfg["swarm_index"] = swarm_idx
            cfg["local_x"] = local_x
            cfg["local_y"] = local_y
            cfg["local_z"] = local_z
            cfg["name"] = f"Swarm {swarm_idx + 1} Drone {drone_local_idx + 1}"
            cfg["x"] = float(swarm_cfg.get("x", 0.0)) + local_x
            cfg["y"] = float(swarm_cfg.get("y", 0.0)) + local_y
            cfg["z"] = float(swarm_cfg.get("z", 0.8)) + local_z
            cfg["color"] = list(swarm_cfg.get("color", SWARM_COLOR_PRESETS["Green"]))

    def _toggle_swarm_expand(self, swarm_idx: int):
        if swarm_idx in self._expanded_swarms:
            self._expanded_swarms.remove(swarm_idx)
        else:
            self._expanded_swarms.add(swarm_idx)
        self._rebuild_object_controls()

    def _set_swarm_field(self, swarm_idx: int, key: str, raw_value):
        with self._config_lock:
            if not (0 <= swarm_idx < len(self._swarm_configs)):
                return
            swarm_cfg = self._swarm_configs[swarm_idx]

            if key in ("x", "y", "z"):
                value = self._coerce_float(raw_value, float(swarm_cfg.get(key, 0.0)))
                swarm_cfg[key] = value
            elif key == "color_name":
                color_name = str(raw_value).strip()
                if color_name not in SWARM_COLOR_PRESETS:
                    color_name = self._default_swarm_color_name(swarm_idx)
                swarm_cfg["color_name"] = color_name
                swarm_cfg["color"] = list(SWARM_COLOR_PRESETS[color_name])
            else:
                value = str(raw_value).strip()
                if value:
                    swarm_cfg[key] = value

            if 0 <= swarm_idx < len(self._target_spawn_configs):
                self._target_spawn_configs[swarm_idx]["color"] = list(
                    swarm_cfg.get("color", SWARM_COLOR_PRESETS["Green"])
                )

            self._rebuild_drone_spawn_configs_locked()

        self._request_restart(f"Restarting after updating Swarm {swarm_idx + 1}...")

    def _set_drone_offset_field(self, drone_index: int, key: str, raw_value):
        with self._config_lock:
            if not (0 <= drone_index < len(self._drone_spawn_configs)):
                return

            current = float(self._drone_spawn_configs[drone_index].get(key, 0.0))
            self._drone_spawn_configs[drone_index][key] = self._coerce_float(
                raw_value, current
            )
            self._rebuild_drone_spawn_configs_locked()

        self._request_restart(f"Restarting after updating Drone {drone_index + 1}...")

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
            "direction": 1,
        }

        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

        return config

    def _sync_object_config_count(self, count: int):
        if self._sidebar_mode == "object_spawns":
            current_count = len(self._object_configs)
            if count > current_count:
                for i in range(current_count, count):
                    self._object_configs.append(self._default_object_config(i))
            elif count < current_count:
                self._object_configs = self._object_configs[:count]
        elif self._sidebar_mode == "target_spawns":
            current_count = len(self._target_spawn_configs)
            if count > current_count:
                for i in range(current_count, count):
                    self._target_spawn_configs.append(
                        self._default_target_spawn_config(i)
                    )
            elif count < current_count:
                self._target_spawn_configs = self._target_spawn_configs[:count]

    def _set_object_field(self, index: int, key: str, raw_value):
        with self._config_lock:
            if self._sidebar_mode == "object_spawns":
                if not (0 <= index < len(self._object_configs)):
                    return
                old = self._object_configs[index].get(key)
                if isinstance(old, float):
                    value = self._coerce_float(raw_value, old)
                elif isinstance(old, int):
                    value = self._coerce_int(raw_value, old)
                else:
                    value = (
                        str(raw_value).strip() if str(raw_value).strip() else str(old)
                    )

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
            else:
                configs = (
                    self._drone_spawn_configs
                    if self._sidebar_mode == "drone_spawns"
                    else self._target_spawn_configs
                )
                if not (0 <= index < len(configs)):
                    return
                old = configs[index].get(key)
                if key in ("x", "y", "z"):
                    value = self._coerce_float(raw_value, old)
                else:
                    value = (
                        str(raw_value).strip() if str(raw_value).strip() else str(old)
                    )
                configs[index][key] = value

        self._request_restart(
            f"Restarting after updating {self._object_name(index)}..."
        )

    def _object_name(self, index: int) -> str:
        with self._config_lock:
            if self._sidebar_mode == "object_spawns":
                if 0 <= index < len(self._object_configs):
                    return str(
                        self._object_configs[index].get("name", f"Object {index + 1}")
                    )
            elif self._sidebar_mode == "drone_spawns":
                if 0 <= index < len(self._drone_spawn_configs):
                    return str(
                        self._drone_spawn_configs[index].get(
                            "name", f"Drone {index + 1}"
                        )
                    )
            else:
                if 0 <= index < len(self._target_spawn_configs):
                    return str(
                        self._target_spawn_configs[index].get(
                            "name", f"Target {index + 1}"
                        )
                    )
        return f"Object {index + 1}"

    def _get_drone_spawn_configs_snapshot(self):
        with self._config_lock:
            self._rebuild_drone_spawn_configs_locked()
            return [dict(cfg) for cfg in self._drone_spawn_configs]

    def _get_target_spawn_configs_snapshot(self):
        with self._config_lock:
            targets = [dict(cfg) for cfg in self._target_spawn_configs]
            for i, cfg in enumerate(targets):
                if "color" in cfg:
                    continue
                if 0 <= i < len(self._swarm_configs):
                    cfg["color"] = list(
                        self._swarm_configs[i].get(
                            "color", SWARM_COLOR_PRESETS["Green"]
                        )
                    )
            return targets

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
                normalized_configs = [
                    self._ensure_config_fields(dict(cfg))
                    for cfg in configs
                    if isinstance(cfg, dict)
                ]
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
                        obstacles=len(normalized_configs),
                    )
                    + f"Speed: {self._env_viewer.get_speed_multiplier():.2f}x\n"
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
        self.time_now_lbl.setText(
            f"{self._format_clock(now_secs)} / {self._format_clock(total_secs)}"
        )

    def _seek_to_frame(self, frame_index: int):
        if self._running:
            return

        if not self._env_viewer.seek_replay_frame(frame_index):
            return

        meta = self._env_viewer.get_replay_frame_meta(frame_index) or {}
        replay_state = dict(self._last_live_state)
        replay_state["episode"] = int(
            meta.get("episode", replay_state.get("episode", 0))
        )
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
