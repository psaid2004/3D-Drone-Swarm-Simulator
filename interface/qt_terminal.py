import os
import sys
import importlib

_QT_CORE = importlib.import_module("PyQt6.QtCore")
_QT_WIDGETS = importlib.import_module("PyQt6.QtWidgets")
QProcess = _QT_CORE.QProcess
QTimer = _QT_CORE.QTimer
QFrame = _QT_WIDGETS.QFrame
QHBoxLayout = _QT_WIDGETS.QHBoxLayout
QLabel = _QT_WIDGETS.QLabel
QLineEdit = _QT_WIDGETS.QLineEdit
QPushButton = _QT_WIDGETS.QPushButton
QTextEdit = _QT_WIDGETS.QTextEdit
QVBoxLayout = _QT_WIDGETS.QVBoxLayout
QWidget = _QT_WIDGETS.QWidget
QObject = _QT_CORE.QObject
pyqtSignal = _QT_CORE.pyqtSignal


class CommandTerminal(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self._start_dir = os.getcwd()

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output_ready)
        self._process.finished.connect(self._on_process_finished)

        self._build_ui()
        self._start_shell()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        title = QLabel("Terminal")
        title.setStyleSheet("font-weight: 700;")

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(170)
        self.output.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")

        cmd_row = QWidget(self)
        cmd_layout = QHBoxLayout(cmd_row)
        cmd_layout.setContentsMargins(0, 0, 0, 0)
        cmd_layout.setSpacing(6)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Enter a command and press Enter")
        self.input.returnPressed.connect(self.run_command)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_command)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_command)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.output.clear)

        cmd_layout.addWidget(self.input, 1)
        cmd_layout.addWidget(self.run_btn)
        cmd_layout.addWidget(self.stop_btn)
        cmd_layout.addWidget(self.clear_btn)

        layout.addWidget(title)
        layout.addWidget(self.output, 1)
        layout.addWidget(cmd_row)

    def _shell_program_and_args(self, command: str):
        if sys.platform.startswith("win"):
            return "powershell.exe", ["-NoProfile", "-NoExit"]
        shell = os.environ.get("SHELL", "/bin/bash")
        return shell, ["-i"]

    def _start_shell(self):
        if self._process.state() != QProcess.ProcessState.NotRunning:
            return
        program, args = self._shell_program_and_args("")
        self._process.setWorkingDirectory(self._start_dir)
        self._process.start(program, args)
        if not self._process.waitForStarted(2000):
            self._append_line("[Terminal] Failed to start shell.")
            self.input.setEnabled(False)
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            return
        self.input.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.input.setFocus()

    def _append_line(self, text: str):
        if not text:
            return
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)
        self.output.insertPlainText(text)
        if not text.endswith("\n"):
            self.output.insertPlainText("\n")
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)

    def run_command(self):
        if self._process.state() == QProcess.ProcessState.NotRunning:
            self._start_shell()
            if self._process.state() == QProcess.ProcessState.NotRunning:
                return

        command = self.input.text()
        self._append_line(f"> {command}")
        self.input.clear()
        self._process.write((command + "\n").encode("utf-8"))
        self._process.waitForBytesWritten(1000)

    def stop_command(self):
        if self._process.state() == QProcess.ProcessState.NotRunning:
            return
        self._process.terminate()
        if not self._process.waitForFinished(1500):
            self._process.kill()

    def _on_output_ready(self):
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_line(chunk)

    def _on_process_finished(self, exit_code: int, exit_status):
        self._append_line(f"[Terminal] Shell exited (code={exit_code}). Restarting...")
        self.input.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        QTimer.singleShot(0, self._start_shell)

class StreamRedirect(QObject):
    store = []
    text_written = pyqtSignal(str)
    
    def write(self, text):
        if text:
            cleaned_text = str(text).strip()
            
            if not cleaned_text:
                return
            
            self.text_written.emit(text)
            self.store.append(cleaned_text)
            
    def flush(self):
        pass

class AppTerminal(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        title = QLabel("Application Output")
        title.setStyleSheet("font-weight: 700;")

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.output.clear)

        layout.addWidget(title)
        layout.addWidget(self.output, 1)
        layout.addWidget(clear_btn)

        # redirect stdout/stderr
        self.redirect = StreamRedirect()
        self.redirect.text_written.connect(self._append)

        sys.stdout = self.redirect
        sys.stderr = self.redirect

    def _append(self, text):
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)

class DualTerminal(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)

        self.cmd_terminal = CommandTerminal()
        self.app_terminal = AppTerminal()

        layout.addWidget(self.cmd_terminal, 1)
        layout.addWidget(self.app_terminal, 1)