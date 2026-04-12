from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,

    QSlider,
    QSpinBox,
    QWidget,
)

from interface.core.env_runner import EnvRunner


class TimelineWidget(QWidget):
    """
    Replay timeline: slider, time labels, replay play/pause, and speed control.
    Delegates frame seek/count operations to EnvRunner.
    """

    speed_changed = pyqtSignal(int)
    frame_seeked = pyqtSignal(dict)

    def __init__(self, env_runner: EnvRunner, parent=None):
        super().__init__(parent)
        self._env_runner = env_runner   
        self._is_scrubbing = False      # True when user is dragging the slider, to avoid conflicts with live updates
        self._is_replay_playing = False
        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._advance_replay)
        self._build_ui()

    # ---- UI construction ----

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)

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

        speed_label = QLabel("Speed")
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(10, 200)
        self.speed_spinbox.setSingleStep(10)
        self.speed_spinbox.setSuffix("%")
        self.speed_spinbox.setValue(100)
        self.speed_spinbox.valueChanged.connect(self._on_speed_changed)

        layout.addWidget(self.time_start_lbl)
        layout.addWidget(self.timeline_slider, 1)
        layout.addWidget(self.time_now_lbl)
        layout.addSpacing(10)
        layout.addWidget(self.replay_btn)
        layout.addSpacing(10)
        layout.addWidget(speed_label)
        layout.addWidget(self.speed_spinbox)

    # ---- Public ----

    def refresh(self):
        """Update slider range and time labels from current EnvRunner state."""
        frame_count = self._env_runner.get_replay_frame_count()
        max_index = max(0, frame_count - 1)

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMaximum(max_index)
        self.timeline_slider.setEnabled(frame_count > 0)
        self.timeline_slider.blockSignals(False)

        is_running = self._env_runner.running   # replay controls are only enabled when not running
        self.replay_btn.setEnabled((not is_running) and frame_count > 1) # more than 1 frame required

        current_index = min(self.timeline_slider.value(), max_index)    # if current index is out of bounds, move it to the end 
        if current_index != self.timeline_slider.value(): 
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(current_index)
            self.timeline_slider.blockSignals(False)

        total_secs = max_index / 30.0   # play in 30 fps
        now_secs = current_index / 30.0
        self.time_start_lbl.setText("00:00")
        self.time_now_lbl.setText(
            f"{self._format_clock(now_secs)} / {self._format_clock(total_secs)}"
        )

    def update_live_position(self, replay_index: int):
        """Called by SimulationPage during live simulation to advance the slider."""
        if not self._is_scrubbing:
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(replay_index)
            self.timeline_slider.blockSignals(False)

    def stop_replay(self):
        """Called by SimulaltionPage to stop replay."""
        self._replay_timer.stop()
        self._is_replay_playing = False
        self.replay_btn.setText("Play Replay")

    # ---- Internal ----

    # Format: mm:ss (ignoring hours and fractional seconds)
    def _format_clock(self, seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"

    def _seek_to_frame(self, frame_index: int):
        """ 
        Seek to the given frame index and emit frame_seeked with the new episode/step.
        Does nothing if EnvRunner is currently running."""
        if self._env_runner.running:
            return
        if not self._env_runner.seek_replay_frame(frame_index):
            return
        meta = self._env_runner.get_replay_frame_meta(frame_index) or {}
        self.frame_seeked.emit({
            "episode": int(meta.get("episode", 0)),
            "step": int(meta.get("step", 0)),
            "speed_multiplier": self._env_runner.get_speed_multiplier(),
        })
        self.refresh()

    def _on_timeline_pressed(self):
        self._is_scrubbing = True

    def _on_timeline_released(self):
        self._is_scrubbing = False
        self._seek_to_frame(self.timeline_slider.value())

    def _on_timeline_value_changed(self, value: int):
        if self._env_runner.running:
            return
        self._seek_to_frame(int(value))

    def _on_speed_changed(self, value: int):
        self.speed_changed.emit(int(value))

    def _toggle_replay(self):
        """
        Start/pause replay playback.
        Does nothing if EnvRunner is currently running.
        """
        if self._env_runner.running:
            return
        frame_count = self._env_runner.get_replay_frame_count()
        if frame_count <= 1:
            return

        self._is_replay_playing = not self._is_replay_playing
        if self._is_replay_playing:
            self.replay_btn.setText("Pause Replay")
            if self._replay_timer.interval() <= 0:
                self._replay_timer.setInterval(33)
            self._replay_timer.start()
        else:
            self._replay_timer.stop()
            self.replay_btn.setText("Play Replay")

    def _advance_replay(self):
        """
        Advance replay by one frame, loops to start if at the end.
        """
        if self._env_runner.running:
            self._replay_timer.stop()
            return
        max_index = max(0, self._env_runner.get_replay_frame_count() - 1)
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
