import queue
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from interface.types import LiveState
from interface.core.config_manager import ConfigManager
from interface.core.env_runner import EnvRunner
from interface.core.frame_buffer import FrameBuffer
from interface.widgets.sidebar_editor import SidebarEditor
from interface.widgets.terminal import DualTerminal
from interface.widgets.timeline_widget import TimelineWidget

LEFT_SIDEBAR_WIDTH = 550


class SimulationPage(QWidget):
    """
    Creates ConfigManager, FrameBuffer, EnvRunner, SidebarEditor, and TimelineWidget, then wires them together.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self._running = False
        self._console: DualTerminal | None = None
        self._console_visible = False
        self._state_queue: queue.Queue = queue.Queue(maxsize=1)
        self._last_live_state = LiveState(episode=0, step=0, drones=0, obstacles=0)

        num_swarms = max(1, int(self.config.get("num_swarms", 1)))
        drones_per_swarm = max(1, int(self.config.get("num_drones", 3)))
        world_scale = float(self.config.get("world_scale", 5))

        # ---- Config manager ----
        self._config_manager = ConfigManager(num_swarms, drones_per_swarm, world_scale)
        num_obstacles = int(self.config.get("num_obstacles", 0))
        if num_obstacles:
            self._config_manager.sync_object_count(num_obstacles)

        # ---- Env runner ----
        self._frame_buffer = FrameBuffer()
        self._env_runner = EnvRunner(
            frame_buffer=self._frame_buffer,
            object_configs_getter=self._config_manager.get_object_configs,
            drone_spawn_configs_getter=self._config_manager.get_drone_spawn_configs,
            target_spawn_configs_getter=self._config_manager.get_target_spawn_configs,
            on_state=self._enqueue_state,
            on_stopped=self._enqueue_stopped,
            on_error=self._enqueue_error,
            num_drones=drones_per_swarm,
            max_steps=int(self.config.get("max_steps", 5000)),
            lidar=bool(self.config.get("lidar", False)),
            render_mode=str(self.config.get("render_mode", "human")),
            model_path=str(self.config.get("model_path", "")),
            world_scale=float(self.config.get("world_scale", 5)),
            num_swarms=num_swarms,
            seed=self.config.get("seed", None),
        )

        # ---- Build layout ----
        self._build_layout(drones_per_swarm)

        # ---- Wire signals ----
        self.sidebar.restart_requested.connect(self._request_restart)
        self.sidebar.swarm_count_changed.connect(self._env_runner.set_num_swarms)
        self.sidebar.randomize_requested.connect(self._on_randomize)
        self.timeline.speed_changed.connect(self._on_speed_changed)
        self.timeline.frame_seeked.connect(self._on_frame_seeked)

        # ---- UI poll timer ----
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._consume_state_updates)
        self._ui_timer.start(50)

    # ---- Layout ----

    def _build_layout(self, drones_per_swarm: int):
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

        # ---- Top bar (controls) ----
        top_bar = QFrame()
        top_bar.setFrameShape(QFrame.Shape.Box)
        top_bar.setFixedHeight(52)
        top_bar.setStyleSheet("background-color: #4CAF50; color: black; font-size: 12px;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)

        title_lbl = QLabel("Simulation Controls")
        title_lbl.setStyleSheet("font-weight: 700;")

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

        self.return_btn = QPushButton("Return")
        self.return_btn.setFixedWidth(100)
        self.return_btn.clicked.connect(self.go_back)

        # ---- Top bar layout ----
        top_layout.addWidget(title_lbl)
        top_layout.addStretch(1)
        top_layout.addWidget(self.status_lbl)
        top_layout.addSpacing(10)
        self.change_model_btn.setStyleSheet("QPushButton:hover { background-color: #45a049; }")
        top_layout.addWidget(self.change_model_btn)
        top_layout.addSpacing(10)
        self.return_btn.setStyleSheet("QPushButton:hover { background-color: #45a049; }")
        top_layout.addWidget(self.return_btn)
        top_layout.addSpacing(10)
        self.save_config_btn.setStyleSheet("QPushButton:hover { background-color: #45a049; }")
        top_layout.addWidget(self.save_config_btn)
        self.load_config_btn.setStyleSheet("QPushButton:hover { background-color: #45a049; }")
        top_layout.addWidget(self.load_config_btn)
        self.console_btn.setStyleSheet("QPushButton:hover { background-color: #45a049; }")
        top_layout.addWidget(self.console_btn)
        self.stop_btn.setStyleSheet("QPushButton:hover { background-color: #45a049; }")
        top_layout.addWidget(self.stop_btn)
        self.start_btn.setStyleSheet("QPushButton:hover { background-color: #45a049; }")
        top_layout.addWidget(self.start_btn)

        # ---- Timeline bar ----
        self.timeline = TimelineWidget(self._env_runner)
        timeline_scroll = QScrollArea()
        timeline_scroll.setWidgetResizable(True)
        timeline_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        timeline_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        timeline_scroll.setWidget(self.timeline)

        # ---- Body: sidebar + info panel ----
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.body_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.body_splitter.setHandleWidth(8)

        self.sidebar = SidebarEditor(self._config_manager, drones_per_swarm)

        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Shape.Box)
        left_frame.setMinimumWidth(0)
        left_frame.setMaximumWidth(LEFT_SIDEBAR_WIDTH)
        left_inner = QVBoxLayout(left_frame)
        left_inner.setContentsMargins(0, 0, 0, 0)
        left_inner.addWidget(self.sidebar)

        self.info_panel = QFrame()
        self.info_panel.setFrameShape(QFrame.Shape.Box)
        info_layout = QVBoxLayout(self.info_panel)
        info_layout.setContentsMargins(14, 14, 14, 12)

        panel_title = QLabel("PyBullet Live View")
        panel_title.setStyleSheet("font-weight: 700;")
        panel_text = QLabel(
            "Any changes while running automatically restart the episode "
            "with the new object setup.\nStatistics:"
        )
        panel_text.setWordWrap(True)
        self.telemetry_lbl = QLabel(
            self._telemetry_text(0, 0, 0, self._config_manager.count("object_spawns"))
        )
        self.telemetry_lbl.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace;"
        )
        info_layout.addWidget(panel_title)
        info_layout.addWidget(panel_text)
        info_layout.addWidget(self.telemetry_lbl)
        info_layout.addStretch(1)

        self.body_splitter.addWidget(left_frame)
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

    # ---- State queue helpers ----

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

    # ---- UI update loop ----

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
            self.timeline.refresh()
            return

        if "error" in latest:
            self.status_lbl.setText(f"Error: {latest['error']}")
            self._stop()
            return

        if "warning" in latest:
            self.status_lbl.setText(f"Warning: {latest['warning']}")

        self._update_telemetry(latest)
        self.timeline.refresh()
        self.timeline.update_live_position(
            int(latest.get("replay_index", self._env_runner.get_replay_index()))
        )

    def _update_telemetry(self, state):
        self._last_live_state = LiveState(
            episode=int(state.get("episode", self._last_live_state["episode"])),
            step=int(state.get("step", self._last_live_state["step"])),
            drones=int(state.get("drones", self._last_live_state["drones"])),
            obstacles=int(state.get("obstacles", self._last_live_state["obstacles"])),
        )
        speed = float(
            state.get("speed_multiplier", self._env_runner.get_speed_multiplier())
        )
        self.telemetry_lbl.setText(
            self._telemetry_text(
                self._last_live_state["episode"],
                self._last_live_state["step"],
                self._last_live_state["drones"],
                self._last_live_state["obstacles"],
            )
            + f"Speed       : {speed:.2f}x\n"
        )

    def _on_frame_seeked(self):
        """Called by TimelineWidget after a successful seeks; updates telemetry."""
        replay_state = LiveState(
            episode=self._last_live_state["episode"],
            step=self._last_live_state["step"],
            drones=self._last_live_state["drones"],
            obstacles=self._last_live_state["obstacles"],
        )
        self._update_telemetry(replay_state)

    @staticmethod
    def _telemetry_text(episode: int, step: int, drones: int, obstacles: int) -> str:
        return (
            f"Episode     : {episode}\n"
            f"Step        : {step}\n"
            f"Live drones : {drones}\n"
            f"Obstacles   : {obstacles}\n"
        )

    # ---- Speed ----

    def _on_speed_changed(self, value: int):
        self._env_runner.set_speed_multiplier(value / 100.0)

    # ---- Console ----

    def _toggle_console(self):
        if self._console is None:
            self._console = DualTerminal(self._root)
            self._root_layout.addWidget(self._console)
        self._console_visible = not self._console_visible
        self._console.setVisible(self._console_visible)
        self.console_btn.setText(
            "Hide console" if self._console_visible else "Open console"
        )

    # ---- Model ----

    def _change_model(self):
        # parents[2] = davids_drones/
        models_dir = Path(__file__).resolve().parents[2] / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select policy model",
            str(models_dir),
            "Model Files (*.zip);;All Files (*)",
        )
        if not file_name:
            return
        ok, message = self._env_runner.set_model_path(file_name)
        if not ok:
            self.status_lbl.setText(f"Error: {message}")
            return
        self.config["model_path"] = file_name
        model_name = Path(file_name).name
        self.status_lbl.setText(f"Model changed to {model_name}")
        if self._running:
            self._request_restart(f"Restarting with model: {model_name}")

    # ---- Simulation control ----

    def _start(self):
        if self._running:
            return
        if not self._env_runner.has_model:
            self.status_lbl.setText("No model loaded. Use 'Change model' to load one.")
            return
        self.timeline.stop_replay()
        self._running = True
        self._on_speed_changed(int(self.timeline.speed_spinbox.value()))
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        mode = str(self.config.get("render_mode", "human"))
        mode_label = "Headless capture" if mode == "none" else "PyBullet GUI"
        self.status_lbl.setText(
            f"Running ({mode_label}) @ {self._env_runner.get_speed_multiplier():.2f}x"
        )
        self._env_runner.start()

    def _stop(self):
        if not self._running:
            return
        self._running = False
        self._env_runner.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("Stopped")
        self.timeline.refresh()

    def _on_randomize(self):
        self._config_manager.randomize()
        self.sidebar.refresh()
        self._request_restart("Restarting with randomized config...")

    def _request_restart(self, message: str):
        if self._running:
            self._env_runner.request_restart()
            self.status_lbl.setText(message)

    def closeEvent(self, event):
        self.timeline.stop_replay()
        self._env_runner.close()
        super().closeEvent(event)

    def go_back(self):
        main_window = self.window()
        if hasattr(main_window, "switch_to_landing"):
            main_window.switch_to_landing()

    # ---- Config persistence ----

    def _save_config(self):
        # parents[1] = interface/
        configs_dir = Path(__file__).resolve().parents[1] / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = configs_dir / f"obj_{timestamp}.json"
        self._config_manager.save_json(file_path)

    def _load_config(self):
        configs_dir = Path(__file__).resolve().parents[1] / "configs"
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

        configs = self._config_manager.load_json(file_path)
        if configs is None:
            self.status_lbl.setText("Load failed")
            return

        self._config_manager.set_object_configs(configs)
        self.sidebar.count_spinbox.blockSignals(True)
        self.sidebar.count_spinbox.setValue(len(configs))
        self.sidebar.count_spinbox.blockSignals(False)
        self.sidebar.refresh()

        self.telemetry_lbl.setText(
            self._telemetry_text(0, 0, 0, len(configs))
            + f"Speed       : {self._env_runner.get_speed_multiplier():.2f}x\n"
        )

        if self._running:
            self._request_restart(f"Restarting with: {file_path.name}")
        else:
            self.status_lbl.setText(f"Loaded config: {file_path.name}")
