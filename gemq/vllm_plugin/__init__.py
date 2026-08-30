"""RobustGEMQ 与 vLLM 的集成组件。"""

from .checkpoint_schema import FORMAT_NAME, SCHEMA_VERSION, validate_manifest


def register() -> None:
    """由 vLLM 通用插件入口调用；导入即完成量化方法注册。"""

    from . import quantization as _quantization  # noqa: F401


__all__ = ["FORMAT_NAME", "SCHEMA_VERSION", "register", "validate_manifest"]
