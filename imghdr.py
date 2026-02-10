"""
Compatibility shim for Python 3.13 where the standard-library ``imghdr`` module
has been removed.

Streamlit < 1.30 仍然会执行 ``import imghdr``，只需要提供一个
兼容的 ``what()`` 函数即可。这里用 Pillow 来做简单的图片类型检测，
失败时返回 ``None``，行为与旧版 ``imghdr.what`` 基本一致。
"""

from __future__ import annotations

from typing import IO, Optional, Union

from PIL import Image


FileArg = Union[str, bytes, "os.PathLike[str]", IO[bytes]]


def what(file: FileArg, h: bytes | None = None) -> Optional[str]:
    """
    Rough replacement for :func:`imghdr.what`.

    - ``file`` 可以是路径字符串，也可以是二进制文件对象。
    - 返回值是小写格式名，例如 ``'png'``、``'jpeg'``，无法识别时返回 ``None``。
    """
    try:
        if hasattr(file, "read"):
            # file-like object
            pos = file.tell()
            try:
                img = Image.open(file)
                fmt = img.format
            finally:
                file.seek(pos)
        else:
            # path-like
            with open(file, "rb") as f:
                img = Image.open(f)
                fmt = img.format

        return fmt.lower() if fmt else None
    except Exception:
        return None


__all__ = ["what"]

