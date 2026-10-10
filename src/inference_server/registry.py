"""Thread-safe local model registry used by future rollout code."""

from dataclasses import dataclass
from threading import RLock

from inference_server.model import CpuEmbeddingModel


@dataclass(frozen=True)
class ModelHandle:
    """An active model and its stable response version."""

    version: str
    model: CpuEmbeddingModel


class ModelRegistry:
    """Atomically swap complete model handles without exposing partial state."""

    def __init__(self, initial: ModelHandle) -> None:
        self._active = initial
        self._lock = RLock()

    def active(self) -> ModelHandle:
        """Return the currently active immutable handle."""
        with self._lock:
            return self._active

    def replace(self, replacement: ModelHandle) -> None:
        """Install a valid new handle atomically."""
        if not replacement.version:
            raise ValueError("model version must not be empty")
        with self._lock:
            self._active = replacement
