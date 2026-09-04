"""Minimal ``loguru`` stub for machines without loguru installed.

No-op logger matching the methods the servers call (info/warning/error/debug/
exception).  The servers use loguru's ``{}``-style formatting; a no-op simply
ignores the arguments.
"""


class _Logger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def debug(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


logger = _Logger()
