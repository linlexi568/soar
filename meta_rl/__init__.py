"""Meta-RL hot-plug controller for Soar."""

try:  # pragma: no cover - optional dependency guard
    from .config import MetaRLConfig, MetaRLOutput, TelemetrySample  # type: ignore
    from .rnn_meta_policy import MetaRNNPolicy  # type: ignore
    from .controller import MetaRLController  # type: ignore
    from .hot_swap import maybe_apply_overrides, write_override_file  # type: ignore

    __all__ = [
        "MetaRLConfig",
        "MetaRLOutput",
        "TelemetrySample",
        "MetaRNNPolicy",
        "MetaRLController",
        "maybe_apply_overrides",
        "write_override_file",
    ]
except Exception:  # pragma: no cover
    __all__ = []
