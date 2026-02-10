"""
极简版 ``pkg_resources`` 兼容层，用于在 Python 3.13 + 精简 setuptools
环境中满足 Ultralytics YOLO 的依赖检查代码。

注意：这里只实现了 Ultralytics 运行时会用到的极少数接口：

- ``parse_version``
- ``parse_requirements``
- ``require``
- ``VersionConflict`` / ``DistributionNotFound``

这些实现都被刻意写得“宽松”，不会真的阻止程序运行，只是让
`ultralytics.yolo.utils.checks` 里的版本检查代码不再抛异常。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List

from packaging.version import parse as _parse_version


def parse_version(v: str):
    """Proxy to ``packaging.version.parse``."""
    return _parse_version(v)


@dataclass
class _Requirement:
    name: str
    specifier: str = ""


def parse_requirements(stream) -> Iterator[_Requirement]:
    """
    简化版 requirements 解析器，只抽取包名，忽略版本等其它信息。
    支持传入一个按行可迭代的文件对象。
    """
    for line in stream:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 去掉环境标记、注释等，只保留最前面的“包名[版本]”部分
        token = line.split(";", 1)[0].split("#", 1)[0].strip()
        if not token:
            continue
        # 非严格解析：认为第一个非空 token 就是包名
        name = token.split()[0]
        yield _Requirement(name=name)


class VersionConflict(Exception):
    pass


class DistributionNotFound(Exception):
    pass


def require(requirements: Iterable[str] | str) -> List[None]:
    """
    极简实现：直接假定所有依赖都已满足，不做任何真实检查。
    这足以让 Ultralytics 的 `check_requirements` 流程继续执行。
    """
    return []


__all__ = [
    "parse_version",
    "parse_requirements",
    "require",
    "VersionConflict",
    "DistributionNotFound",
]

