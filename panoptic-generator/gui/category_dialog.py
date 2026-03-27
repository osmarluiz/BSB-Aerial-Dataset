"""Dialog for editing auto-discovered categories."""

from __future__ import annotations

from pathlib import Path

from gui.qt_compat import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    Qt,
)

from core.annotation_source import DiscoveredCategory
from core.models import CategoryInfo


class CategoryEditorDialog(QDialog):
    """Table dialog to edit auto-discovered categories."""

    def __init__(
        self,
        discovered: list[DiscoveredCategory],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Categories")
        self.setMinimumSize(700, 400)
        self._discovered = discovered
        self._result: list[CategoryInfo] | None = None

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Review and edit the discovered categories:"))

        # Table
        self._table = QTableWidget(len(discovered), 5)
        self._table.setHorizontalHeaderLabels(
            ["Original", "ID", "Name", "Thing?", "Supercategory"]
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        for row, cat in enumerate(discovered):
            # Original value (read-only)
            item = QTableWidgetItem(f"{cat.original_value} ({cat.count})")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, item)

            # ID (spin box)
            id_spin = QSpinBox()
            id_spin.setMinimum(1)
            id_spin.setMaximum(999)
            id_spin.setValue(cat.suggested_id)
            self._table.setCellWidget(row, 1, id_spin)

            # Name
            self._table.setItem(row, 2, QTableWidgetItem(cat.suggested_name))

            # Is thing (checkbox)
            thing_check = QCheckBox()
            thing_check.setChecked(True)
            container = _center_widget(thing_check)
            self._table.setCellWidget(row, 3, container)

            # Supercategory
            self._table.setItem(row, 4, QTableWidgetItem(""))

        layout.addWidget(self._table)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        categories = []
        for row in range(self._table.rowCount()):
            id_spin = self._table.cellWidget(row, 1)
            name_item = self._table.item(row, 2)
            thing_container = self._table.cellWidget(row, 3)
            thing_check = thing_container.findChild(QCheckBox)
            supercat_item = self._table.item(row, 4)

            categories.append(
                CategoryInfo(
                    id=id_spin.value(),
                    name=name_item.text(),
                    is_thing=thing_check.isChecked(),
                    supercategory=supercat_item.text() if supercat_item else "",
                )
            )

        # Validate unique IDs
        ids = [c.id for c in categories]
        if len(ids) != len(set(ids)):
            from gui.qt_compat import QMessageBox
            QMessageBox.warning(self, "Error", "Category IDs must be unique.")
            return

        self._result = categories
        self.accept()

    def get_categories(self) -> list[CategoryInfo] | None:
        return self._result


def _center_widget(widget):
    """Wrap a widget in a container that centers it."""
    from gui.qt_compat import QWidget, QHBoxLayout

    container = QWidget()
    layout = QHBoxLayout(container)
    layout.addWidget(widget)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.setContentsMargins(0, 0, 0, 0)
    return container
