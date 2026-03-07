import queue
import threading
import importlib
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interface.environment_viewer import EnvironmentViewer


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
QSpinBox = _QT_WIDGETS.QSpinBox

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
        self._state_queue = queue.Queue(maxsize=5)

        self._config_lock = threading.Lock()
        self._object_configs = [self._default_object_config(i) for i in range(4)]

        self._env_viewer = EnvironmentViewer(
            object_configs_getter=self._get_object_configs_snapshot,
            on_state=self._enqueue_state,
            on_stopped=self._enqueue_stopped,
            on_error=self._enqueue_error,
        )

        self._build_layout()

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._consume_state_updates)
        self._ui_timer.start(50)

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
        self.object_spinbox = QSpinBox()
        self.object_spinbox.setRange(0, 100)  # You can set max to whatever you want
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
    def _on_object_spinbox(self, value: int):
        """Handle spin box value changes"""
        # Remove this line: self.object_count_value_lbl.setText(str(value))
        self._sync_object_config_count(int(value))
        self._rebuild_object_controls()
        self._request_restart(f"Restarting with updated object count: {value}")
        
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
            self._env_viewer.request_restart()
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
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText("Running (PyBullet GUI)")
        self._env_viewer.start()

    def _stop(self):
        if not self._running:
            return

        self._running = False
        self._env_viewer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("Stopped")

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)

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
            return

        if "error" in latest:
            self.status_lbl.setText(f"Error: {latest['error']}")
            self._stop()
            return

        self._update_telemetry(latest)

    def _update_telemetry(self, state):
        self.telemetry_lbl.setText(
            self._telemetry_text(
                episode=int(state.get("episode", 0)),
                step=int(state.get("step", 0)),
                drones=int(state.get("drones", 0)),
                obstacles=int(state.get("obstacles", 0)),
            )
        )


def main():
    app = QApplication(sys.argv)
    viewer = SimulationViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
