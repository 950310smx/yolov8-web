"""
Background worker that runs YOLO inference without blocking the GUI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO


@dataclass
class WorkerConfig:
    model_path: str
    image_paths: List[str]
    output_dir: Path
    device: str = "0"
    naming_mode: str = "suffix"  # suffix|custom
    custom_prefix: Optional[str] = None
    conf: float = 0.25
    imgsz: int = 640


class InferenceWorker(QThread):
    """Runs YOLO predictions on a separate thread."""

    progress_changed = Signal(int)
    status_changed = Signal(str)
    image_finished = Signal(dict)
    run_failed = Signal(str)
    run_completed = Signal()

    def __init__(self, config: WorkerConfig) -> None:
        super().__init__()
        self.config = config
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            model = YOLO(self.config.model_path)
        except Exception as exc:  # pragma: no cover - GUI feedback only
            self.run_failed.emit(f"模型加载失败: {exc}")
            return

        total = len(self.config.image_paths)
        if not total:
            self.run_failed.emit("未选择任何图片。")
            return

        min_duration_per_image = 2.0  # 每张图片至少显示 2 秒进度

        for index, image_path in enumerate(self.config.image_paths):
            if self._abort:
                break

            start = time.perf_counter()
            self.status_changed.emit(f"正在检测: {Path(image_path).name}")

            # 模拟进度：先快速到 30%，再慢慢到 90%，最后完成
            base_progress = int((index / total) * 100)
            target_progress = int(((index + 1) / total) * 100)

            # 阶段1：快速到 30% 位置
            stage1_target = base_progress + int((target_progress - base_progress) * 0.3)
            for p in range(base_progress, stage1_target + 1, 2):
                self.progress_changed.emit(p)
                time.sleep(0.03)

            try:
                results = model.predict(
                    source=image_path,
                    conf=self.config.conf,
                    imgsz=self.config.imgsz,
                    device=self.config.device,
                    save=False,
                    verbose=False,
                )
            except Exception as exc:  # pragma: no cover
                self.run_failed.emit(f"检测失败 {image_path}: {exc}")
                return

            if not results:
                self.run_failed.emit(f"未获得检测结果: {image_path}")
                return

            # 阶段2：推理完成，进度到 90%
            stage2_target = base_progress + int((target_progress - base_progress) * 0.9)
            for p in range(stage1_target, stage2_target + 1, 3):
                self.progress_changed.emit(p)
                time.sleep(0.02)

            result = results[0]
            annotated = result.plot()
            output_path = self._build_output_path(Path(image_path), index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)

            metrics = self._collect_metrics(result, annotated.shape[:2])
            duration = time.perf_counter() - start

            # 阶段3：完成到 100%，并确保总时长至少 2 秒
            remaining_time = max(0, min_duration_per_image - duration)
            steps = max(1, target_progress - stage2_target)
            step_delay = remaining_time / steps
            for p in range(stage2_target, target_progress + 1):
                self.progress_changed.emit(p)
                time.sleep(step_delay)

            self.image_finished.emit(
                {
                    "source": image_path,
                    "output": str(output_path),
                    "metrics": metrics,
                    "duration": time.perf_counter() - start,
                }
            )

        if self._abort:
            self.run_failed.emit("检测被用户中断。")
            return

        self.status_changed.emit("全部检测完成")
        self.run_completed.emit()

    # Helpers -----------------------------------------------------------------

    def _build_output_path(self, image_path: Path, index: int) -> Path:
        stem = image_path.stem
        suffix = image_path.suffix or ".jpg"
        if self.config.naming_mode == "custom" and self.config.custom_prefix:
            filename = f"{self.config.custom_prefix}{index + 1:03d}{suffix}"
        else:
            filename = f"{stem}_test{suffix}"
        return self.config.output_dir / filename

    @staticmethod
    def _collect_metrics(result, shape) -> dict:
        boxes = result.boxes
        if boxes is None or boxes.data.shape[0] == 0:
            return {
                "count": 0,
                "avg_conf": 0.0,
                "max_conf": 0.0,
                "min_conf": 0.0,
                "size_bins": {"small": 0, "medium": 0, "large": 0},
            }

        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        h, w = shape
        image_area = float(h * w) if h and w else 1.0
        areas = np.maximum((xyxy[:, 2] - xyxy[:, 0]), 0) * np.maximum(
            (xyxy[:, 3] - xyxy[:, 1]), 0
        )
        ratios = areas / image_area

        size_bins = {"small": 0, "medium": 0, "large": 0}
        for ratio in ratios:
            if ratio < 0.005:
                size_bins["small"] += 1
            elif ratio < 0.02:
                size_bins["medium"] += 1
            else:
                size_bins["large"] += 1

        return {
            "count": int(len(confs)),
            "avg_conf": float(confs.mean()),
            "max_conf": float(confs.max()),
            "min_conf": float(confs.min()),
            "size_bins": size_bins,
        }


