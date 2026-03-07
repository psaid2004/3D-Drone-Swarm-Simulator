import queue
import random
import threading
import time
import importlib
from functools import partial
from pathlib import Path

from envs.createObjects import Dynamic, EnvObject
from wrappers.pz_parallel_3d import env as make_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load Qt bindings and assign classes to variables
_QT_CORE = importlib.import_module("PyQt6.QtCore")
_QT_WIDGETS = importlib.import_module("PyQt6.QtWidgets")
Qt = _QT_CORE.Qt
QTimer = _QT_CORE.QTimer
QApplication = _QT_WIDGETS.QApplication
QComboBox = _QT_WIDGETS.QComboBox
QFrame = _QT_WIDGETS.QFrame
QHBoxLayout = _QT_WIDGETS.QHBoxLayout
QLabel = _QT_WIDGETS.QLabel
QLineEdit = _QT_WIDGETS.QLineEdit
QMainWindow = _QT_WIDGETS.QMainWindow
QScrollArea = _QT_WIDGETS.QScrollArea
QSlider = _QT_WIDGETS.QSlider
QVBoxLayout = _QT_WIDGETS.QVBoxLayout
QWidget = _QT_WIDGETS.QWidget
QPushButton = _QT_WIDGETS.QPushButton
QGroupBox = _QT_WIDGETS.QGroupBox

# Constants for object configuration
SHAPES = ("box", "wall", "sphere", "cylinder")
KINDS = ("static", "dynamic")
AXIS_LABEL_TO_INT = {"x": 0, "y": 1, "z": 2}
AXIS_INT_TO_LABEL = {0: "x", 1: "y", 2: "z"}


class SimulationViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Simulation")
        self.resize(1680, 1024)
        self.setMinimumSize(980, 640)

        self._running = False
        self._sim_thread = None
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._state_queue = queue.Queue(maxsize=5)

        self._episode = 0
        self._step = 0

        self._config_lock = threading.Lock()
        self._object_configs = [self._default_object_config(i) for i in range(4)]

        self._build_layout()

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._consume_state_updates)
        self._ui_timer.start(50)

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

    def _build_layout(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
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

        top_layout.addWidget(title)
        top_layout.addStretch(1)
        top_layout.addWidget(self.status_lbl)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.stop_btn)
        top_layout.addWidget(self.start_btn)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.left_sidebar = QFrame()
        self.left_sidebar.setFrameShape(QFrame.Shape.Box)
        self.left_sidebar.setFixedWidth(390)
        left_layout = QVBoxLayout(self.left_sidebar)
        left_layout.setContentsMargins(12, 12, 12, 10)

        side_title = QLabel("Objects")
        side_title.setStyleSheet("font-weight: 700;")

        count_label = QLabel("Object count")
        self.object_count_value_lbl = QLabel(str(len(self._object_configs)))

        self.object_slider = QSlider(Qt.Orientation.Horizontal)
        self.object_slider.setRange(0, 20)
        self.object_slider.setValue(len(self._object_configs))
        self.object_slider.valueChanged.connect(self._on_object_slider)

        count_row = QHBoxLayout()
        count_row.addWidget(count_label)
        count_row.addStretch(1)
        count_row.addWidget(self.object_count_value_lbl)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_host = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_host)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(8)
        self.scroll_area.setWidget(self.scroll_host)

        left_layout.addWidget(side_title)
        left_layout.addLayout(count_row)
        left_layout.addWidget(self.object_slider)
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

        body_layout.addWidget(self.left_sidebar)
        body_layout.addWidget(self.info_panel, 1)

        root_layout.addWidget(top_bar)
        root_layout.addWidget(body, 1)

        self._rebuild_object_controls()

    def _telemetry_text(self, episode: int, step: int, drones: int, obstacles: int) -> str:
        return (
            f"Episode     : {episode}\n"
            f"Step        : {step}\n"
            f"Live drones : {drones}\n"
            f"Obstacles   : {obstacles}\n"
        )

    def _on_object_slider(self, value: int):
        self.object_count_value_lbl.setText(str(value))
        self._sync_object_config_count(int(value))
        self._rebuild_object_controls()
        self._request_restart("Restarting with updated object count...")

    def _sync_object_config_count(self, count: int):
        with self._config_lock:
            current = list(self._object_configs)
            if count > len(current):
                for i in range(len(current), count):
                    current.append(self._default_object_config(i))
            elif count < len(current):
                current = current[:count]
            self._object_configs = current

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

            if key == "shape" and value not in SHAPES:
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

    def _request_restart(self, message: str):
        if self._running:
            self._restart_event.set()
            self.status_lbl.setText(message)

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
        combo = QComboBox()
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

        for i, cfg in enumerate(configs):
            card = QGroupBox(f"Object {i + 1}")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(4)

            card_layout.addWidget(self._add_labeled_entry(card, "Name", cfg["name"], partial(self._set_object_field, i, "name")))
            card_layout.addWidget(self._add_option_menu(card, "Type", cfg["kind"], KINDS, partial(self._set_object_field, i, "kind")))
            card_layout.addWidget(self._add_option_menu(card, "Shape", cfg["shape"], SHAPES, partial(self._set_object_field, i, "shape")))

            card_layout.addWidget(self._add_labeled_entry(card, "X", cfg["x"], partial(self._set_object_field, i, "x")))
            card_layout.addWidget(self._add_labeled_entry(card, "Y", cfg["y"], partial(self._set_object_field, i, "y")))
            card_layout.addWidget(self._add_labeled_entry(card, "Z", cfg["z"], partial(self._set_object_field, i, "z")))
            card_layout.addWidget(self._add_labeled_entry(card, "Size", cfg["size"], partial(self._set_object_field, i, "size")))
            card_layout.addWidget(self._add_labeled_entry(card, "Yaw", cfg["yaw"], partial(self._set_object_field, i, "yaw")))

            axis_label = AXIS_INT_TO_LABEL.get(int(cfg["axis"]), "x")
            card_layout.addWidget(self._add_option_menu(card, "Axis", axis_label, ("x", "y", "z"), partial(self._set_object_field, i, "axis")))
            card_layout.addWidget(self._add_labeled_entry(card, "Min", cfg["min"], partial(self._set_object_field, i, "min")))
            card_layout.addWidget(self._add_labeled_entry(card, "Max", cfg["max"], partial(self._set_object_field, i, "max")))
            card_layout.addWidget(self._add_labeled_entry(card, "Speed", cfg["speed"], partial(self._set_object_field, i, "speed")))
            card_layout.addWidget(self._add_labeled_entry(card, "Dir", cfg["direction"], partial(self._set_object_field, i, "direction")))

            self.scroll_layout.addWidget(card)

        self.scroll_layout.addStretch(1)

    def _start(self):
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._restart_event.clear()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText("Running (PyBullet GUI)")

        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

    def _stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("Stopped")

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)

    def _simulation_loop(self):
        current_env = None
        try:
            while not self._stop_event.is_set():
                current_env = self._make_env()
                obs_dict, _ = current_env.reset(seed=random.randint(0, 2**31 - 1))
                self._apply_object_configs(current_env)
                self._episode += 1
                self._step = 0

                done = {agent: False for agent in current_env.possible_agents}
                trunc = {agent: False for agent in current_env.possible_agents}

                while not self._stop_event.is_set():
                    if self._restart_event.is_set():
                        self._restart_event.clear()
                        break

                    act_dict = {agent: current_env.action_space(agent).sample() for agent in obs_dict}
                    obs_dict, _, done, trunc, _ = current_env.step(act_dict)
                    self._step += 1

                    self._publish_state(current_env)
                    time.sleep(1 / 30)

                    if all(done[a] or trunc[a] for a in done):
                        break

                current_env.close()
                current_env = None

        except Exception as exc:
            self._state_queue.put({"error": str(exc)})
        finally:
            if current_env is not None:
                try:
                    current_env.close()
                except Exception:
                    pass
            self._state_queue.put({"stopped": True})

    def _make_env(self):
        with self._config_lock:
            configs = [dict(cfg) for cfg in self._object_configs]

        static_count = sum(1 for cfg in configs if cfg.get("kind") == "static")
        dynamic_count = sum(1 for cfg in configs if cfg.get("kind") == "dynamic")

        return make_env(
            num_drones=8,
            num_static_obstacles=static_count,
            num_dynamic_obstacles=dynamic_count,
            max_steps=500,
            lidar=False,
            render_mode="human",
        )

    def _apply_object_configs(self, env_instance):
        with self._config_lock:
            configs = [dict(cfg) for cfg in self._object_configs]

        p = getattr(env_instance, "_p", None)
        if p is None:
            return

        old_obstacles = list(getattr(env_instance, "_obstacles", []))
        for obj in old_obstacles:
            body_id = getattr(obj, "body_id", None)
            if body_id is None:
                continue
            try:
                env_instance._call_physics(p.removeBody, body_id)
            except Exception:
                continue

        new_obstacles = []
        for cfg in configs:
            pos = [float(cfg["x"]), float(cfg["y"]), float(cfg["z"])]
            ori = [0.0, 0.0, float(cfg["yaw"])]
            size = max(0.1, float(cfg["size"]))
            shape = str(cfg["shape"]) if str(cfg["shape"]) in SHAPES else "box"

            if cfg.get("kind") == "dynamic":
                axis = int(cfg.get("axis", 0))
                min_pos = float(cfg.get("min", -1.0))
                max_pos = float(cfg.get("max", 1.0))
                if min_pos > max_pos:
                    min_pos, max_pos = max_pos, min_pos
                speed = max(0.0, float(cfg.get("speed", 0.2)))
                direction = -1 if int(cfg.get("direction", 1)) < 0 else 1
                obj = Dynamic(axis, min_pos, max_pos, speed, direction, pos, size, ori)
            else:
                obj = EnvObject(pos, size, ori)

            try:
                obj.create(shape, physics_call=env_instance._call_physics)
                new_obstacles.append(obj)
            except Exception:
                continue

        env_instance._obstacles = new_obstacles
        env_instance.num_static_obstacles = sum(1 for cfg in configs if cfg.get("kind") == "static")
        env_instance.num_dynamic_obstacles = sum(1 for cfg in configs if cfg.get("kind") == "dynamic")

    def _publish_state(self, env_instance):
        payload = {
            "episode": self._episode,
            "step": self._step,
            "drones": len(getattr(env_instance, "agents", [])),
            "obstacles": len(getattr(env_instance, "_obstacles", [])),
        }

        while not self._state_queue.empty():
            try:
                self._state_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self._state_queue.put_nowait(payload)
        except queue.Full:
            pass

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
            self._stop()
            return

        if "error" in latest:
            self.status_lbl.setText(f"Error: {latest['error']}")
            self._stop()
            return

        self._update_telemetry(latest)

    def _update_telemetry(self, state):
        self.telemetry_lbl.setText(
            self._telemetry_text(
                episode=int(state["episode"]),
                step=int(state["step"]),
                drones=int(state["drones"]),
                obstacles=int(state["obstacles"]),
            )
        )


def main():
    app = QApplication(sys.argv)
    viewer = SimulationViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
