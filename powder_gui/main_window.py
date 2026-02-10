"""
Main window implementation for the powder detection GUI.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPixmap, QBrush, QPen, QPainter, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QGraphicsDropShadowEffect,
)

from .history_manager import HistoryManager
from .inference_worker import InferenceWorker, WorkerConfig


class NeonFrame(QGroupBox):
    """Group box with animated electric border (lightweight version)."""

    # 共享定时器，减少系统开销
    _shared_timer: QTimer | None = None
    _instances: list = []

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._phase = 0.0
        NeonFrame._instances.append(self)
        if NeonFrame._shared_timer is None:
            NeonFrame._shared_timer = QTimer()
            NeonFrame._shared_timer.timeout.connect(NeonFrame._global_advance)
            NeonFrame._shared_timer.start(50)  # 20fps，足够流畅且省资源

    @staticmethod
    def _global_advance() -> None:
        for inst in NeonFrame._instances:
            inst._phase = (inst._phase + 0.015) % 1.0
            inst.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        rect = self.rect().adjusted(8, 18, -8, -8)
        w, h = rect.width(), rect.height()
        if w <= 0 or h <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 静态底边
        painter.setPen(QPen(QColor(16, 52, 76, 200), 1.0))
        painter.drawRoundedRect(rect, 18, 18)

        # 计算电流位置
        perimeter = 2.0 * (w + h)
        head_pos = self._phase * perimeter
        tail_len = perimeter * 0.12  # 电流总长度

        # 绘制渐变尾巴（3段足够，减少绘制次数）
        for i in range(3):
            ratio = 1.0 - i / 3.0
            alpha = int(220 * ratio)
            thickness = 1.0 + 1.8 * ratio
            color = QColor(88, 213, 255, alpha)
            pen = QPen(color, thickness)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)

            seg_start = head_pos - i * (tail_len / 3)
            seg_len = tail_len / 3
            self._draw_segment(painter, rect, seg_start, seg_len, perimeter)

        painter.end()

    def _draw_segment(self, painter: QPainter, rect, start: float, length: float, perimeter: float) -> None:
        if length <= 0:
            return
        dist = start % perimeter
        remaining = length
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        # 四条边：上右下左
        edges = [
            (w, x, y, 1, 0),      # 上边
            (h, x + w, y, 0, 1),  # 右边
            (w, x + w, y + h, -1, 0),  # 下边
            (h, x, y + h, 0, -1),      # 左边
        ]

        while remaining > 0:
            edge_idx, offset = self._find_edge(dist, edges)
            edge_len, ox, oy, dx, dy = edges[edge_idx]
            span = min(remaining, edge_len - offset)

            p1 = QPointF(ox + dx * offset, oy + dy * offset)
            p2 = QPointF(ox + dx * (offset + span), oy + dy * (offset + span))
            painter.drawLine(p1, p2)

            remaining -= span
            dist = (dist + span) % perimeter

    @staticmethod
    def _find_edge(dist: float, edges) -> tuple:
        acc = 0.0
        for i, (length, *_) in enumerate(edges):
            if dist < acc + length:
                return i, dist - acc
            acc += length
        return 0, 0.0


class MainWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self, model_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model_path = model_path
        self.selected_images: List[str] = []
        self.current_output_path: str | None = None
        self.worker: InferenceWorker | None = None
        self.current_pixmap: QPixmap | None = None

        history_path = Path("powder_gui") / "history" / "history.json"
        self.history_manager = HistoryManager(history_path)

        self.setWindowTitle("粉末检测助手 - YOLOv8")
        self.resize(1300, 820)
        self._build_ui()
        self._load_history_table()

    # ------------------------------------------------------------------ UI ----

    def _build_ui(self) -> None:
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        main_layout.addWidget(self._build_banner())

        body_widget = QWidget()
        body_widget.setObjectName("bodyWrapper")
        body_layout = QHBoxLayout(body_widget)
        body_layout.setSpacing(18)
        self._apply_body_background(body_widget)

        body_layout.addWidget(self._build_controls_panel(), 1)
        body_layout.addWidget(self._build_preview_panel(), 2)
        body_layout.addWidget(self._build_metrics_panel(), 1)

        main_layout.addWidget(body_widget)

        self.setCentralWidget(container)

    def _build_controls_panel(self) -> QWidget:
        panel = NeonFrame("任务设置")
        panel.setObjectName("controlPanel")
        self._apply_neon_shadow(panel)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        # Image selection
        select_btn = QPushButton("选择图片")
        select_btn.clicked.connect(self._select_images)
        layout.addWidget(select_btn)

        self.image_list = QListWidget()
        self.image_list.setMinimumHeight(160)
        layout.addWidget(self.image_list)

        remove_btn = QPushButton("移除选中图片")
        remove_btn.clicked.connect(self._remove_selected_images)
        layout.addWidget(remove_btn)

        clear_btn = QPushButton("清空列表")
        clear_btn.clicked.connect(self._clear_image_list)
        layout.addWidget(clear_btn)

        # Output directory
        output_box = QGroupBox("输出目录")
        output_layout = QHBoxLayout(output_box)
        self.output_edit = QLineEdit()
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self._choose_output_dir)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(browse_btn)
        layout.addWidget(output_box)

        # Naming strategy
        naming_box = QGroupBox("文件命名")
        naming_layout = QVBoxLayout(naming_box)
        self.naming_group = QButtonGroup(self)
        suffix_radio = QRadioButton("原名 + _test")
        suffix_radio.setChecked(True)
        custom_radio = QRadioButton("自定义前缀")
        self.naming_group.addButton(suffix_radio, 0)
        self.naming_group.addButton(custom_radio, 1)
        naming_layout.addWidget(suffix_radio)
        naming_layout.addWidget(custom_radio)
        prefix_row = QHBoxLayout()
        prefix_label = QLabel("前缀：")
        prefix_label.setObjectName("prefixLabel")
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.prefix_edit.setMinimumHeight(32)
        self.prefix_edit.setPlaceholderText("例如：powder_")
        self.prefix_edit.setEnabled(False)
        custom_radio.toggled.connect(self.prefix_edit.setEnabled)
        prefix_row.addWidget(prefix_label)
        prefix_row.addWidget(self.prefix_edit)
        naming_layout.addLayout(prefix_row)
        layout.addWidget(naming_box)

        # Start button
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setStyleSheet("QPushButton { font-size: 16px; padding: 10px; }")
        self.start_btn.clicked.connect(self._start_detection)
        layout.addWidget(self.start_btn)
        layout.addStretch(1)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = NeonFrame("检测进度 / 结果")
        panel.setObjectName("previewPanel")
        self._apply_neon_shadow(panel)
        layout = QVBoxLayout(panel)

        self.preview_stack = QStackedWidget()

        # Progress widget
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label = QLabel("等待开始...")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 18px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(28)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addSpacing(12)
        progress_layout.addWidget(self.progress_bar)

        # Image preview widget
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        self.preview_image = QLabel("检测结果将显示在此")
        self.preview_image.setObjectName("previewCanvas")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_image.setScaledContents(False)
        preview_layout.addWidget(self.preview_image)
        self.success_label = QLabel("")
        self.success_label.setObjectName("success_label")
        self.success_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.success_label)

        self.preview_stack.addWidget(progress_widget)
        self.preview_stack.addWidget(preview_widget)
        layout.addWidget(self.preview_stack)
        return panel

    def _build_metrics_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(self._build_metrics_page())
        self.right_stack.addWidget(self._build_history_page())
        layout.addWidget(self.right_stack)
        return panel

    def _build_metrics_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        metrics_box = NeonFrame("实时指标")
        metrics_box.setObjectName("metricsPanel")
        self._apply_neon_shadow(metrics_box)
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setSpacing(10)
        grid = QGridLayout()
        labels = [
            ("颗粒总数", "count"),
            ("平均置信度", "avg_conf"),
            ("最高置信度", "max_conf"),
            ("最低置信度", "min_conf"),
            ("小颗粒", "small"),
            ("中颗粒", "medium"),
            ("大颗粒", "large"),
        ]
        self.metric_labels: Dict[str, QLabel] = {}
        columns = 2
        for idx, (title, key) in enumerate(labels):
            card = QGroupBox()
            card.setObjectName("metricCard")
            card_layout = QVBoxLayout(card)
            caption = QLabel(title)
            caption.setObjectName("metricTitle")
            value = QLabel("--")
            value.setObjectName("metricValue")
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            row = idx // columns
            col = idx % columns
            grid.addWidget(card, row, col)
            self.metric_labels[key] = value
        metrics_layout.addLayout(grid)
        layout.addWidget(metrics_box)
        history_btn = QPushButton("查看历史记录")
        history_btn.clicked.connect(lambda: self.right_stack.setCurrentIndex(1))
        layout.addWidget(history_btn)
        return page

    def _build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        history_box = QGroupBox("历史记录")
        history_box.setObjectName("historyPanel")
        self._apply_neon_shadow(history_box)
        history_layout = QVBoxLayout(history_box)
        self.history_table = QTableWidget(0, 4)
        self.history_table.setObjectName("historyTable")
        self.history_table.setHorizontalHeaderLabels(["时间", "文件", "颗粒数", "结果路径"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.doubleClicked.connect(self._load_history_preview)
        history_layout.addWidget(self.history_table)
        btn_row = QHBoxLayout()
        back_btn = QPushButton("返回仪表盘")
        back_btn.clicked.connect(lambda: self.right_stack.setCurrentIndex(0))
        clear_history_btn = QPushButton("清空历史")
        clear_history_btn.clicked.connect(self._clear_history)
        btn_row.addWidget(back_btn)
        btn_row.addWidget(clear_history_btn)
        history_layout.addLayout(btn_row)
        layout.addWidget(history_box)
        return page

    def _build_banner(self) -> QWidget:
        banner = QGroupBox()
        banner.setObjectName("bannerFrame")
        layout = QHBoxLayout(banner)

        text_layout = QVBoxLayout()
        title = QLabel("POWDER LAB // 霓虹粉末检测控制台")
        title.setObjectName("bannerTitle")
        subtitle = QLabel("实时监控 · 精准识别 · 实验室级可视化")
        subtitle.setObjectName("bannerSubtitle")
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)
        layout.addLayout(text_layout)

        layout.addStretch()

        self.theme_btn = QPushButton("切换主题")
        self.theme_btn.setObjectName("themeButton")
        self.theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(self.theme_btn)

        return banner

    def _toggle_theme(self) -> None:
        from pathlib import Path
        if not hasattr(self, '_current_theme'):
            self._current_theme = "dark"

        if self._current_theme == "dark":
            self._current_theme = "light"
            style_path = Path(__file__).resolve().parent / "assets" / "style_light.qss"
        else:
            self._current_theme = "dark"
            style_path = Path(__file__).resolve().parent / "assets" / "style.qss"

        if style_path.exists():
            with style_path.open("r", encoding="utf-8") as f:
                QApplication.instance().setStyleSheet(f.read())

    def _apply_neon_shadow(self, widget: QWidget) -> None:
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(40)
        effect.setColor(QColor(14, 165, 233, 120))
        effect.setOffset(0, 0)
        widget.setGraphicsEffect(effect)

    def _apply_body_background(self, widget: QWidget) -> None:
        size = 240
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor("#050608"))
        painter = QPainter(pixmap)
        grid_pen = QPen(QColor(14, 165, 233, 25))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        step = 24
        for pos in range(0, size, step):
            painter.drawLine(pos, 0, pos, size)
            painter.drawLine(0, pos, size, pos)
        painter.end()
        palette = widget.palette()
        palette.setBrush(QPalette.ColorRole.Window, QBrush(pixmap))
        widget.setAutoFillBackground(True)
        widget.setPalette(palette)

    # ----------------------------------------------------------- UI actions ---

    def _select_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要检测的图片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not files:
            return
        for path in files:
            if path not in self.selected_images:
                self.selected_images.append(path)
                self.image_list.addItem(path)

    def _remove_selected_images(self) -> None:
        for item in self.image_list.selectedItems():
            row = self.image_list.row(item)
            path = self.selected_images[row]
            self.selected_images.pop(row)
            self.image_list.takeItem(row)

    def _clear_image_list(self) -> None:
        self.selected_images.clear()
        self.image_list.clear()

    def _choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.output_edit.setText(directory)

    def _start_detection(self) -> None:
        if not self.selected_images:
            QMessageBox.warning(self, "提示", "请先选择至少一张图片。")
            return
        output_dir = self.output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录。")
            return
        naming_mode = "custom" if self.naming_group.checkedId() == 1 else "suffix"
        prefix = self.prefix_edit.text().strip() if naming_mode == "custom" else None
        if naming_mode == "custom" and not prefix:
            QMessageBox.warning(self, "提示", "请输入自定义前缀。")
            return

        self._toggle_controls(False)
        self.success_label.setText("")
        self.preview_stack.setCurrentIndex(0)
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在初始化模型...")

        config = WorkerConfig(
            model_path=self.model_path,
            image_paths=list(self.selected_images),
            output_dir=Path(output_dir),
            naming_mode=naming_mode,
            custom_prefix=prefix,
        )
        self.worker = InferenceWorker(config)
        self.worker.progress_changed.connect(self._update_progress)
        self.worker.status_changed.connect(self._update_status)
        self.worker.image_finished.connect(self._handle_image_finished)
        self.worker.run_failed.connect(self._handle_run_failed)
        self.worker.run_completed.connect(self._handle_run_completed)
        self.worker.start()

    def _toggle_controls(self, enabled: bool) -> None:
        for widget in [
            self.image_list,
            self.output_edit,
            self.start_btn,
        ]:
            widget.setEnabled(enabled)

    def _update_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"检测进度：{value}%")

    def _update_status(self, message: str) -> None:
        self.progress_label.setText(message)

    def _handle_image_finished(self, info: Dict) -> None:
        self.current_output_path = info["output"]
        self._update_metrics(info["metrics"])
        self._add_history_entry(info)
        if Path(self.current_output_path).exists():
            pix = QPixmap(self.current_output_path)
            self._set_preview_pixmap(pix)
            self.preview_stack.setCurrentIndex(1)

    def _handle_run_failed(self, message: str) -> None:
        QMessageBox.critical(self, "检测失败", message)
        self._toggle_controls(True)

    def _handle_run_completed(self) -> None:
        self._toggle_controls(True)
        self.success_label.setText("✅ 检测成功")
        if self.current_output_path and Path(self.current_output_path).exists():
            pixmap = QPixmap(self.current_output_path)
            self._set_preview_pixmap(pixmap)
            self.preview_stack.setCurrentIndex(1)
        else:
            self.preview_image.setText("未找到结果图。")
        self.progress_label.setText("全部检测完成")

    def _update_metrics(self, metrics: Dict) -> None:
        self.metric_labels["count"].setText(str(metrics["count"]))
        self.metric_labels["avg_conf"].setText(f'{metrics["avg_conf"] * 100:.1f}%')
        self.metric_labels["max_conf"].setText(f'{metrics["max_conf"] * 100:.1f}%')
        self.metric_labels["min_conf"].setText(f'{metrics["min_conf"] * 100:.1f}%')
        size_bins = metrics["size_bins"]
        self.metric_labels["small"].setText(str(size_bins["small"]))
        self.metric_labels["medium"].setText(str(size_bins["medium"]))
        self.metric_labels["large"].setText(str(size_bins["large"]))

    # ----------------------------------------------------------- History ------

    def _add_history_entry(self, info: Dict) -> None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics = info["metrics"]
        row_data = [timestamp, Path(info["output"]).name, str(metrics["count"]), info["output"]]
        row_position = self.history_table.rowCount()
        self.history_table.insertRow(row_position)
        for col, value in enumerate(row_data):
            item = QTableWidgetItem(value)
            if col == 2:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.history_table.setItem(row_position, col, item)
        entry = {
            "timestamp": timestamp,
            "file": info["output"],
            "count": metrics["count"],
            "avg_conf": metrics["avg_conf"],
        }
        self.history_manager.add_entry(entry)

    def _load_history_table(self) -> None:
        for entry in self.history_manager.entries:
            row_position = self.history_table.rowCount()
            self.history_table.insertRow(row_position)
            self.history_table.setItem(row_position, 0, QTableWidgetItem(entry.get("timestamp", "")))
            self.history_table.setItem(
                row_position, 1, QTableWidgetItem(Path(entry.get("file", "")).name)
            )
            self.history_table.setItem(
                row_position,
                2,
                QTableWidgetItem(str(entry.get("count", 0))),
            )
            self.history_table.setItem(row_position, 3, QTableWidgetItem(entry.get("file", "")))

    def _load_history_preview(self) -> None:
        row = self.history_table.currentRow()
        if row < 0:
            return
        path_item = self.history_table.item(row, 3)
        if not path_item:
            return
        output_path = path_item.text()
        if Path(output_path).exists():
            pixmap = QPixmap(output_path)
            self._set_preview_pixmap(pixmap)
            self.preview_stack.setCurrentIndex(1)
            self.success_label.setText(f"历史记录：{Path(output_path).name}")
        else:
            QMessageBox.information(self, "提示", "该历史记录的文件不存在。")

    def _clear_history(self) -> None:
        if QMessageBox.question(self, "确认", "确定要清空所有历史记录吗？") == QMessageBox.StandardButton.Yes:
            self.history_manager.clear()
            self.history_table.setRowCount(0)

    # ----------------------------------------------------------- Preview helpers

    def _set_preview_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            return
        self.current_pixmap = pixmap
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self) -> None:
        if not self.current_pixmap or self.preview_image.width() <= 0 or self.preview_image.height() <= 0:
            return
        target_width = max(100, self.preview_image.width() - 40)
        target_height = max(100, self.preview_image.height() - 40)
        scaled = self.current_pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_image.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_preview_pixmap()


