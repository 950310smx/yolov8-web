"""
Entry point for the powder detection GUI.
"""
from __future__ import annotations

import site
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication


def ensure_site_packages_priority() -> None:
    """Ensure pip-installed Ultralytics takes precedence over local source tree."""
    project_root = str(Path(__file__).resolve().parents[1])
    # Remove current project path so it doesn't overshadow site-packages
    sys.path = [p for p in sys.path if p != project_root]
    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except AttributeError:
        pass
    site_paths.append(site.getusersitepackages())
    for path in reversed([p for p in site_paths if p and Path(p).exists()]):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_stylesheet(app: QApplication) -> None:
    style_path = Path(__file__).resolve().parent / "assets" / "style.qss"
    if style_path.exists():
        with style_path.open("r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())


def main() -> None:
    ensure_site_packages_priority()
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    load_stylesheet(app)
    root_dir = Path(__file__).resolve().parents[1]
    model_path = str(root_dir / "1031.onnx")
    window = MainWindow(model_path=model_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


