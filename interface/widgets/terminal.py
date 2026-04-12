import os
import sys

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class CommandTerminal(QFrame):
    """Terminal widget that allows running commands in shell."""
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

    # ---- UI ----

    def _build_ui(self):
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(170)
        self.output.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")

        self.input = QLineEdit()
        self.input.setPlaceholderText("Enter a command")
        self.input.returnPressed.connect(self.run_command)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_command)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_command)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.output.clear)

        title = QLabel("Terminal")
        title.setStyleSheet("font-weight: 700;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        cmd_row = QWidget(self)
        cmd_layout = QHBoxLayout(cmd_row)
        cmd_layout.setContentsMargins(0, 0, 0, 0)
        cmd_layout.setSpacing(6)

        cmd_layout.addWidget(self.input, 1)
        cmd_layout.addWidget(self.run_btn)
        cmd_layout.addWidget(self.stop_btn)
        cmd_layout.addWidget(self.clear_btn)

        layout.addWidget(title)
        layout.addWidget(self.output, 1)
        layout.addWidget(cmd_row)

    # ---- Terminal logic ----

    def _start_shell(self):
        """Starts the Shel process."""
        if self._process.state() != QProcess.ProcessState.NotRunning:
            return
        # Try Windows Powershell 
        if sys.platform.startswith("win"):
            program, args = "powershell.exe", ["-NoProfile", "-NoExit"]
        # Try Linux Shell
        else:
            shell = os.environ.get("SHELL", "/bin/bash")
            program, args = shell, ["-i"]

        self._process.setWorkingDirectory(self._start_dir)
        self._process.start(program, args)
        if not self._process.waitForStarted(1500):
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
        """Append a line of text to the terminal output."""
        if not text:
            return
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)
        self.output.insertPlainText(text)
        if not text.endswith("\n"):
            self.output.insertPlainText("\n")
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)

    def run_command(self):
        """Run the command input from the user. Start shell and skip first try if shell not open."""
        if self._process.state() == QProcess.ProcessState.NotRunning:
            self._start_shell()
            return
        
        command = self.input.text()
        self.input.clear()
        self._process.write((command + "\n").encode("utf-8"))

    def stop_command(self):
        """Terminate the shell process."""
        if self._process.state() == QProcess.ProcessState.NotRunning:
            return
        self._process.terminate()
        if not self._process.waitForFinished(1500):
            self._process.kill()

    def _on_output_ready(self):
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_line(chunk)

    def _on_process_finished(self, exit_code: int):
        """Handle the shell process finishing and restart.""" 
        self._append_line(f"[Terminal] Shell stopped (code={exit_code}). Restarting...")
        self.input.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        QTimer.singleShot(0, self._start_shell)


class StreamRedirect(QObject):
    """Redirects text written to it as a Qt signal."""
    text_written = pyqtSignal(str)

    def write(self, text):
        if text:
            self.text_written.emit(str(text))

    def flush(self):
        pass


class AppTerminal(QFrame):
    """Terminal widget that captures and displays stdout and stderr from the application."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.output.clear)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        title = QLabel("Application Output")
        title.setStyleSheet("font-weight: 700;")

        layout.addWidget(title)
        layout.addWidget(self.output, 1)
        layout.addWidget(clear_btn)

        self.redirect = StreamRedirect()
        self.redirect.text_written.connect(self._append)
        sys.stdout = self.redirect
        sys.stderr = self.redirect

    def _append(self, text: str):
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)


class DualTerminal(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.cmd_terminal = CommandTerminal()
        self.app_terminal = AppTerminal()
        layout.addWidget(self.cmd_terminal, 1)
        layout.addWidget(self.app_terminal, 1)
