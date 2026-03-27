"""Main application window — modern minimal dark UI."""

from __future__ import annotations

import sys
from pathlib import Path

from gui.qt_compat import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QThread,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)

from gui.widgets import DirField, FileField


# ── Style ────────────────────────────────────────────────────────────────────

BG = "#18181b"
CARD = "#27272a"
BORDER = "#3f3f46"
TEXT = "#fafafa"
TEXT2 = "#a1a1aa"
ACCENT = "#3b82f6"
ACCENT_HOVER = "#60a5fa"
INPUT_BG = "#18181b"
HOVER = "#3f3f46"

STYLE = f"""
* {{
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
    color: {TEXT};
}}

QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {BG};
}}

QFrame[class="card"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

QLabel {{
    background: transparent;
    border: none;
}}

QLabel[class="section"] {{
    font-size: 15px;
    font-weight: 700;
    color: {TEXT};
    padding: 0;
}}

QLabel[class="hint"] {{
    font-size: 11px;
    color: {TEXT2};
}}

QLabel[class="field-label"] {{
    font-size: 12px;
    color: {TEXT2};
    padding-bottom: 2px;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {INPUT_BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px;
    min-height: 16px;
    selection-background-color: #1e3a5f;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}

QListWidget {{
    background: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
    font-size: 13px;
    color: {TEXT};
}}

QListWidget::item {{
    padding: 3px 8px;
    border-radius: 4px;
}}

QListWidget::item:hover {{
    background: {HOVER};
}}

QPushButton {{
    background: {BORDER};
    color: {TEXT};
    border: none;
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 600;
}}

QPushButton:hover {{
    background: {HOVER};
}}

QPushButton:disabled {{
    color: #52525b;
    background: #27272a;
}}

QPushButton[class="primary"] {{
    background: {ACCENT};
    color: white;
}}

QPushButton[class="primary"]:hover {{
    background: {ACCENT_HOVER};
}}

QPushButton[class="small"] {{
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
}}

QRadioButton, QCheckBox {{
    spacing: 8px;
    font-size: 13px;
    color: {TEXT};
}}

QRadioButton::indicator, QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}

QProgressBar {{
    text-align: center;
    border: none;
    border-radius: 8px;
    background: {INPUT_BG};
    color: {TEXT2};
    font-size: 12px;
    min-height: 28px;
    border: 1px solid {BORDER};
}}

QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 7px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 6px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 3px;
    min-height: 30px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 10px;
}}

QComboBox QAbstractItemView {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: #1e3a5f;
}}
"""


def _card() -> QFrame:
    """Create a card container (rounded, subtle background)."""
    frame = QFrame()
    frame.setProperty("class", "card")
    return frame


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("class", "section")
    return lbl


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("class", "field-label")
    return lbl


def _hint_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("class", "hint")
    return lbl


# ── Pipeline Worker Thread ───────────────────────────────────────────────────


class PipelineWorker(QThread):
    stage_started = Signal(str)
    stage_ended = Signal(str)
    progress = Signal(int, int, str)
    warning = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config

    def run(self):
        from core.models import PipelineListener
        from core.pipeline import Pipeline

        worker = self

        class QtListener(PipelineListener):
            def on_stage_start(self, name):
                worker.stage_started.emit(name)

            def on_stage_end(self, name):
                worker.stage_ended.emit(name)

            def on_progress(self, current, total, message):
                worker.progress.emit(current, total, message)

            def on_warning(self, message):
                worker.warning.emit(message)

        pipeline = Pipeline(self._config, listener=QtListener())
        try:
            stats = pipeline.run()
            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


# ── Main Window ──────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panoptic Generator")
        self.setMinimumSize(560, 700)
        self.resize(600, 860)

        self._categories = None
        self._worker = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)

        root = QWidget()
        scroll.setWidget(root)
        m = QVBoxLayout(root)
        m.setSpacing(12)
        m.setContentsMargins(24, 20, 24, 24)

        # Header
        title = QLabel("Panoptic Generator")
        title.setStyleSheet("font-size: 24px; font-weight: 800; letter-spacing: -0.5px;")
        m.addWidget(title)
        m.addWidget(_hint_label("Generate segmentation datasets from remote sensing imagery"))
        m.addSpacing(4)

        # ── Image card ───────────────────────────────────────────────
        card = _card()
        cl = QVBoxLayout(card)
        cl.setSpacing(10)
        cl.setContentsMargins(16, 16, 16, 16)
        cl.addWidget(_section_label("Image"))

        cl.addWidget(_field_label("Source raster"))
        self.image_field = FileField(
            extensions=[".tif", ".tiff", ".img", ".hdr"],
            filter_str="Raster (*.tif *.tiff *.img *.hdr);;All (*)",
        )
        self.image_field.file_selected.connect(self._on_image_selected)
        cl.addWidget(self.image_field)

        self.bands_label = _field_label("Bands")
        self.bands_label.hide()
        cl.addWidget(self.bands_label)
        self.bands_list = QListWidget()
        self.bands_list.setMaximumHeight(80)
        self.bands_list.hide()
        cl.addWidget(self.bands_list)

        self.bands_btns = QWidget()
        hb = QHBoxLayout(self.bands_btns)
        hb.setContentsMargins(0, 0, 0, 0)
        btn_all = QPushButton("Select All")
        btn_all.setProperty("class", "small")
        btn_all.clicked.connect(lambda: self._toggle_bands(True))
        btn_none = QPushButton("Deselect All")
        btn_none.setProperty("class", "small")
        btn_none.clicked.connect(lambda: self._toggle_bands(False))
        hb.addWidget(btn_all)
        hb.addWidget(btn_none)
        hb.addStretch()
        self.bands_btns.hide()
        cl.addWidget(self.bands_btns)

        m.addWidget(card)

        # ── Annotation card ──────────────────────────────────────────
        card = _card()
        cl = QVBoxLayout(card)
        cl.setSpacing(10)
        cl.setContentsMargins(16, 16, 16, 16)
        cl.addWidget(_section_label("Annotation"))

        hr = QHBoxLayout()
        self.rb_ann_shp = QRadioButton("Shapefile")
        self.rb_ann_raster = QRadioButton("Raster (legacy)")
        self.rb_ann_shp.setChecked(True)
        self.ann_mode_group = QButtonGroup()
        self.ann_mode_group.addButton(self.rb_ann_shp, 0)
        self.ann_mode_group.addButton(self.rb_ann_raster, 1)
        hr.addWidget(self.rb_ann_shp)
        hr.addWidget(self.rb_ann_raster)
        hr.addStretch()
        cl.addLayout(hr)

        # Shapefile sub-section
        self.ann_shp_widget = QWidget()
        ls = QVBoxLayout(self.ann_shp_widget)
        ls.setContentsMargins(0, 0, 0, 0)
        ls.setSpacing(8)
        ls.addWidget(_field_label("Annotation shapefile"))
        self.ann_shp_field = FileField(extensions=[".shp"], filter_str="Shapefile (*.shp)")
        self.ann_shp_field.file_selected.connect(self._on_annotation_shp_selected)
        ls.addWidget(self.ann_shp_field)

        ls.addWidget(_field_label("Class column"))
        self.class_column_combo = QComboBox()
        ls.addWidget(self.class_column_combo)
        cl.addWidget(self.ann_shp_widget)

        # Raster sub-section
        self.ann_raster_widget = QWidget()
        lr = QVBoxLayout(self.ann_raster_widget)
        lr.setContentsMargins(0, 0, 0, 0)
        lr.setSpacing(8)
        lr.addWidget(_field_label("Semantic mask"))
        self.semantic_field = FileField(extensions=[".tif", ".tiff"], filter_str="Raster (*.tif *.tiff)")
        lr.addWidget(self.semantic_field)
        lr.addWidget(_field_label("Panoptic mask (optional)"))
        self.panoptic_field = FileField(extensions=[".tif", ".tiff"], filter_str="Raster (*.tif *.tiff)")
        lr.addWidget(self.panoptic_field)
        self.ann_raster_widget.hide()
        cl.addWidget(self.ann_raster_widget)

        # Categories
        cl.addWidget(_field_label("Categories"))
        hc = QHBoxLayout()
        self.cat_field = FileField(extensions=[".yaml", ".yml", ".json"], filter_str="Config (*.yaml *.yml *.json)")
        hc.addWidget(self.cat_field)
        self.btn_auto_detect = QPushButton("Auto-detect")
        self.btn_auto_detect.setProperty("class", "primary")
        self.btn_auto_detect.clicked.connect(self._on_auto_detect)
        hc.addWidget(self.btn_auto_detect)
        cl.addLayout(hc)

        self.ann_mode_group.idToggled.connect(self._on_ann_mode_changed)
        m.addWidget(card)

        # ── Sampling card ────────────────────────────────────────────
        card = _card()
        cl = QVBoxLayout(card)
        cl.setSpacing(10)
        cl.setContentsMargins(16, 16, 16, 16)
        cl.addWidget(_section_label("Sampling"))

        # Tile size row
        ht = QHBoxLayout()
        ht.addWidget(_field_label("Tile"))
        self.tile_h_spin = QSpinBox()
        self.tile_h_spin.setRange(16, 9999)
        self.tile_h_spin.setValue(512)
        self.tile_h_spin.setPrefix("H ")
        ht.addWidget(self.tile_h_spin)
        self.tile_w_spin = QSpinBox()
        self.tile_w_spin.setRange(16, 9999)
        self.tile_w_spin.setValue(512)
        self.tile_w_spin.setPrefix("W ")
        ht.addWidget(self.tile_w_spin)
        ht.addStretch()
        cl.addLayout(ht)

        # Mode
        hr2 = QHBoxLayout()
        self.rb_points = QRadioButton("Point Shapefiles")
        self.rb_sliding = QRadioButton("Sliding Window")
        self.rb_points.setChecked(True)
        self.samp_mode_group = QButtonGroup()
        self.samp_mode_group.addButton(self.rb_points, 0)
        self.samp_mode_group.addButton(self.rb_sliding, 1)
        hr2.addWidget(self.rb_points)
        hr2.addWidget(self.rb_sliding)
        hr2.addStretch()
        cl.addLayout(hr2)

        # Point fields
        self.points_widget = QWidget()
        lp = QVBoxLayout(self.points_widget)
        lp.setContentsMargins(0, 0, 0, 0)
        lp.setSpacing(6)
        self.train_shp_field = FileField("Train", [".shp"], "Shapefile (*.shp)")
        self.val_shp_field = FileField("Val", [".shp"], "Shapefile (*.shp)")
        self.test_shp_field = FileField("Test", [".shp"], "Shapefile (*.shp)")
        lp.addWidget(self.train_shp_field)
        lp.addWidget(self.val_shp_field)
        lp.addWidget(self.test_shp_field)
        cl.addWidget(self.points_widget)

        # Sliding window fields
        self.sliding_widget = QWidget()
        lsl = QVBoxLayout(self.sliding_widget)
        lsl.setContentsMargins(0, 0, 0, 0)
        lsl.setSpacing(8)
        hs = QHBoxLayout()
        hs.addWidget(_field_label("Stride"))
        self.stride_y_spin = QSpinBox()
        self.stride_y_spin.setRange(1, 9999)
        self.stride_y_spin.setValue(512)
        self.stride_y_spin.setPrefix("Y ")
        hs.addWidget(self.stride_y_spin)
        self.stride_x_spin = QSpinBox()
        self.stride_x_spin.setRange(1, 9999)
        self.stride_x_spin.setValue(512)
        self.stride_x_spin.setPrefix("X ")
        hs.addWidget(self.stride_x_spin)
        hs.addStretch()
        lsl.addLayout(hs)

        hsp = QHBoxLayout()
        hsp.addWidget(_field_label("Split %"))
        self.split_train = QSpinBox()
        self.split_train.setRange(0, 100)
        self.split_train.setValue(70)
        self.split_train.setSuffix(" train")
        hsp.addWidget(self.split_train)
        self.split_val = QSpinBox()
        self.split_val.setRange(0, 100)
        self.split_val.setValue(15)
        self.split_val.setSuffix(" val")
        hsp.addWidget(self.split_val)
        self.split_test = QSpinBox()
        self.split_test.setRange(0, 100)
        self.split_test.setValue(15)
        self.split_test.setSuffix(" test")
        hsp.addWidget(self.split_test)
        lsl.addLayout(hsp)
        self.sliding_widget.hide()
        cl.addWidget(self.sliding_widget)

        self.samp_mode_group.idToggled.connect(self._on_samp_mode_changed)
        m.addWidget(card)

        # ── Output card ──────────────────────────────────────────────
        card = _card()
        cl = QVBoxLayout(card)
        cl.setSpacing(10)
        cl.setContentsMargins(16, 16, 16, 16)
        cl.addWidget(_section_label("Output"))

        cl.addWidget(_field_label("Output directory"))
        self.output_dir_field = DirField()
        self.output_dir_field.set_text("./output")
        cl.addWidget(self.output_dir_field)

        # Image format
        hf = QHBoxLayout()
        hf.addWidget(_field_label("Image format"))
        self.rb_tiff = QRadioButton("TIFF")
        self.rb_png = QRadioButton("PNG")
        self.rb_tiff.setChecked(True)
        hf.addWidget(self.rb_tiff)
        hf.addWidget(self.rb_png)
        hf.addStretch()
        cl.addLayout(hf)

        # Mask modes
        cl.addWidget(_field_label("Generate masks"))
        hm = QHBoxLayout()
        self.chk_semantic = QCheckBox("Semantic")
        self.chk_semantic.setChecked(True)
        self.chk_instance = QCheckBox("Instance")
        self.chk_instance.setChecked(True)
        self.chk_panoptic = QCheckBox("Panoptic")
        self.chk_panoptic.setChecked(True)
        self.chk_stuff = QCheckBox("Stuff")
        self.chk_stuff.setChecked(True)
        hm.addWidget(self.chk_semantic)
        hm.addWidget(self.chk_instance)
        hm.addWidget(self.chk_panoptic)
        hm.addWidget(self.chk_stuff)
        hm.addStretch()
        cl.addLayout(hm)

        # Annotation formats
        cl.addWidget(_field_label("Annotation formats"))
        haf = QHBoxLayout()
        self.chk_coco_pan = QCheckBox("COCO Pan.")
        self.chk_coco_pan.setChecked(True)
        self.chk_coco_inst = QCheckBox("COCO Inst.")
        self.chk_coco_inst.setChecked(True)
        self.chk_yolo_det = QCheckBox("YOLO Det.")
        self.chk_yolo_seg = QCheckBox("YOLO Seg.")
        self.chk_voc = QCheckBox("VOC")
        haf.addWidget(self.chk_coco_pan)
        haf.addWidget(self.chk_coco_inst)
        haf.addWidget(self.chk_yolo_det)
        haf.addWidget(self.chk_yolo_seg)
        haf.addWidget(self.chk_voc)
        haf.addStretch()
        cl.addLayout(haf)

        # Filters
        hfil = QHBoxLayout()
        hfil.addWidget(_field_label("Min area"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 99999)
        self.min_area_spin.setValue(10)
        self.min_area_spin.setSuffix(" px")
        hfil.addWidget(self.min_area_spin)
        hfil.addWidget(_field_label("Min ratio"))
        self.min_ratio_spin = QDoubleSpinBox()
        self.min_ratio_spin.setRange(0.0, 1.0)
        self.min_ratio_spin.setSingleStep(0.01)
        self.min_ratio_spin.setValue(0.0)
        hfil.addWidget(self.min_ratio_spin)
        hfil.addStretch()
        cl.addLayout(hfil)

        self.chk_stats = QCheckBox("Generate statistics report")
        self.chk_stats.setChecked(True)
        cl.addWidget(self.chk_stats)

        m.addWidget(card)

        # ── Progress ─────────────────────────────────────────────────
        self.stage_label = QLabel("Ready")
        self.stage_label.setProperty("class", "hint")
        m.addWidget(self.stage_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(28)
        m.addWidget(self.progress_bar)

        # ── Actions ──────────────────────────────────────────────────
        ha = QHBoxLayout()
        ha.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        ha.addWidget(self.btn_cancel)

        self.btn_generate = QPushButton("Generate Dataset")
        self.btn_generate.setProperty("class", "primary")
        self.btn_generate.setStyleSheet(
            f"background: {ACCENT}; padding: 11px 32px; font-size: 14px; font-weight: 700;"
        )
        self.btn_generate.clicked.connect(self._on_generate)
        ha.addWidget(self.btn_generate)
        m.addLayout(ha)

        m.addStretch()

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_image_selected(self, path: str) -> None:
        try:
            from core.image_reader import ImageReader

            with ImageReader(Path(path)) as img:
                self.bands_list.clear()
                for name in img.band_names:
                    item = QListWidgetItem(name)
                    item.setCheckState(Qt.CheckState.Checked)
                    self.bands_list.addItem(item)
                self.bands_label.show()
                self.bands_list.show()
                self.bands_btns.show()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read image:\n{e}")

    def _toggle_bands(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.bands_list.count()):
            self.bands_list.item(i).setCheckState(state)

    def _on_annotation_shp_selected(self, path: str) -> None:
        try:
            import geopandas as gpd

            gdf = gpd.read_file(path, rows=0)
            self.class_column_combo.clear()
            for col in gdf.columns:
                if col != "geometry":
                    self.class_column_combo.addItem(col)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read shapefile:\n{e}")

    def _on_ann_mode_changed(self, id_: int, checked: bool) -> None:
        if not checked:
            return
        self.ann_shp_widget.setVisible(id_ == 0)
        self.ann_raster_widget.setVisible(id_ == 1)

    def _on_samp_mode_changed(self, id_: int, checked: bool) -> None:
        if not checked:
            return
        self.points_widget.setVisible(id_ == 0)
        self.sliding_widget.setVisible(id_ == 1)

    def _on_auto_detect(self) -> None:
        shp_path = self.ann_shp_field.text()
        if not shp_path:
            QMessageBox.warning(self, "Error", "Select an annotation shapefile first.")
            return

        class_col = self.class_column_combo.currentText()
        if not class_col:
            QMessageBox.warning(self, "Error", "Select a class column first.")
            return

        try:
            from core.annotation_source import discover_categories
            from gui.category_dialog import CategoryEditorDialog

            discovered = discover_categories(Path(shp_path), class_col)
            dialog = CategoryEditorDialog(discovered, parent=self)
            if dialog.exec():
                self._categories = dialog.get_categories()
                if self._categories:
                    QMessageBox.information(
                        self, "Done", f"{len(self._categories)} categories configured."
                    )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Auto-detect failed:\n{e}")

    def _on_generate(self) -> None:
        try:
            config = self._build_config()
        except Exception as e:
            QMessageBox.warning(self, "Validation Error", str(e))
            return

        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self._worker = PipelineWorker(config, parent=self)
        self._worker.stage_started.connect(lambda s: self.stage_label.setText(s))
        self._worker.stage_ended.connect(lambda s: self.stage_label.setText(s))
        self._worker.progress.connect(self._on_progress)
        self._worker.warning.connect(lambda s: self.stage_label.setText(f"Warning: {s}"))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self.stage_label.setText("Cancelled")
            self.btn_generate.setEnabled(True)
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_finished(self, stats) -> None:
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        msg = "Dataset generated successfully!"
        if stats:
            from core.stats import stats_to_text

            msg += f"\n\n{stats_to_text(stats)}"
        self.stage_label.setText("Done!")
        QMessageBox.information(self, "Complete", msg)

    def _on_error(self, message: str) -> None:
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.stage_label.setText("Error!")
        QMessageBox.critical(self, "Error", message)

    # ── Config Builder ───────────────────────────────────────────────

    def _build_config(self):
        from core.config import (
            ExtractionConfig,
            OutputConfig,
            PipelineConfig,
            PointSamplingConfig,
            RasterAnnotationConfig,
            ShapefileAnnotationConfig,
            SlidingWindowConfig,
        )
        from core.models import AnnotationFormat, ImageFormat, OutputMode

        image_path = self.image_field.text()
        if not image_path:
            raise ValueError("Select an image file")

        if self.rb_ann_shp.isChecked():
            shp = self.ann_shp_field.text()
            if not shp:
                raise ValueError("Select an annotation shapefile")
            annotation = ShapefileAnnotationConfig(
                shapefile_path=Path(shp),
                class_column=self.class_column_combo.currentText() or "class_id",
            )
        else:
            sem = self.semantic_field.text()
            if not sem:
                raise ValueError("Select a semantic raster")
            annotation = RasterAnnotationConfig(
                semantic_path=Path(sem),
                panoptic_path=Path(self.panoptic_field.text()) if self.panoptic_field.text() else None,
            )

        categories = self._categories
        cat_path = self.cat_field.text()
        category_config_path = Path(cat_path) if cat_path else None

        if self.rb_points.isChecked():
            shapefiles: dict[str, Path] = {}
            if self.train_shp_field.text():
                shapefiles["train"] = Path(self.train_shp_field.text())
            if self.val_shp_field.text():
                shapefiles["val"] = Path(self.val_shp_field.text())
            if self.test_shp_field.text():
                shapefiles["test"] = Path(self.test_shp_field.text())
            if not shapefiles:
                raise ValueError("Select at least a train shapefile")
            sampling = PointSamplingConfig(shapefiles=shapefiles)
        else:
            t = self.split_train.value() / 100
            v = self.split_val.value() / 100
            te = self.split_test.value() / 100
            sampling = SlidingWindowConfig(
                stride_y=self.stride_y_spin.value(),
                stride_x=self.stride_x_spin.value(),
                split_ratios={"train": t, "val": v, "test": te},
            )

        bands = []
        for i in range(self.bands_list.count()):
            if self.bands_list.item(i).checkState() == Qt.CheckState.Checked:
                bands.append(i + 1)
        bands = bands or None

        mode = OutputMode(0)
        if self.chk_semantic.isChecked():
            mode |= OutputMode.SEMANTIC
        if self.chk_instance.isChecked():
            mode |= OutputMode.INSTANCE
        if self.chk_panoptic.isChecked():
            mode |= OutputMode.PANOPTIC
        if not mode:
            mode = OutputMode.ALL

        formats = []
        if self.chk_coco_pan.isChecked():
            formats.append(AnnotationFormat.COCO_PANOPTIC)
        if self.chk_coco_inst.isChecked():
            formats.append(AnnotationFormat.COCO_INSTANCE)
        if self.chk_yolo_det.isChecked():
            formats.append(AnnotationFormat.YOLO_DETECT)
        if self.chk_yolo_seg.isChecked():
            formats.append(AnnotationFormat.YOLO_SEG)
        if self.chk_voc.isChecked():
            formats.append(AnnotationFormat.PASCAL_VOC)
        if not formats:
            formats = [AnnotationFormat.COCO_PANOPTIC]

        return PipelineConfig(
            image_path=Path(image_path),
            annotation=annotation,
            sampling=sampling,
            tile_height=self.tile_h_spin.value(),
            tile_width=self.tile_w_spin.value(),
            categories=categories,
            category_config_path=category_config_path,
            extraction=ExtractionConfig(
                bands=bands,
                output_mode=mode,
                min_segment_area=self.min_area_spin.value(),
                min_annotated_ratio=self.min_ratio_spin.value(),
            ),
            output=OutputConfig(
                output_dir=Path(self.output_dir_field.text()),
                image_format=ImageFormat.TIFF if self.rb_tiff.isChecked() else ImageFormat.PNG,
                annotation_formats=formats,
                create_stuff_masks=self.chk_stuff.isChecked(),
                generate_stats=self.chk_stats.isChecked(),
            ),
        )


def run_gui():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())
