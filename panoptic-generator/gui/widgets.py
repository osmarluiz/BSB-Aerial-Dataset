"""Reusable custom widgets: DropLineEdit, FileField, DirField."""

from __future__ import annotations

from pathlib import Path

from gui.qt_compat import (
    QDragEnterEvent,
    QDropEvent,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    Qt,
    Signal,
    QWidget,
)


class DropLineEdit(QLineEdit):
    """QLineEdit that accepts file drops."""

    file_dropped = Signal(str)

    def __init__(
        self,
        accepted_extensions: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._accepted = (
            [e.lower() for e in accepted_extensions] if accepted_extensions else None
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("border-color: #3b82f6;")

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent) -> None:
        self.setStyleSheet("")
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if self._accepted and path.suffix.lower() not in self._accepted:
            return
        self.setText(str(path))
        self.file_dropped.emit(str(path))


class FileField(QWidget):
    """File input: line edit + browse button. Supports drag-and-drop."""

    file_selected = Signal(str)

    def __init__(
        self,
        label: str = "",
        extensions: list[str] | None = None,
        filter_str: str = "All Files (*)",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._filter = filter_str

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        if label:
            lbl = QLabel(label)
            lbl.setFixedWidth(40)
            lbl.setStyleSheet("color: #71717a; font-size: 12px;")
            layout.addWidget(lbl)

        self.line_edit = DropLineEdit(accepted_extensions=extensions, parent=self)
        self.line_edit.setPlaceholderText("Drop file or click Browse")
        self.line_edit.file_dropped.connect(self._on_file)
        layout.addWidget(self.line_edit)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self.browse_btn)

    def text(self) -> str:
        return self.line_edit.text()

    def set_text(self, text: str) -> None:
        self.line_edit.setText(text)

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self._filter)
        if path:
            self.line_edit.setText(path)
            self._on_file(path)

    def _on_file(self, path: str) -> None:
        self.file_selected.emit(path)


class DirField(QWidget):
    """Directory input: line edit + browse button."""

    dir_selected = Signal(str)

    def __init__(self, label: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        if label:
            lbl = QLabel(label)
            lbl.setFixedWidth(40)
            lbl.setStyleSheet("color: #71717a; font-size: 12px;")
            layout.addWidget(lbl)

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Select directory")
        layout.addWidget(self.line_edit)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self.browse_btn)

    def text(self) -> str:
        return self.line_edit.text()

    def set_text(self, text: str) -> None:
        self.line_edit.setText(text)

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            self.line_edit.setText(path)
            self.dir_selected.emit(path)
