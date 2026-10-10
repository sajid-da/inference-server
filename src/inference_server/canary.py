"""Canary routing strategy and rollback for model rollouts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from threading import RLock

from inference_server.registry import ModelHandle, ModelRegistry


@dataclass(frozen=True)
class CanaryConfig:
    """Canary routing parameters."""

    canary_weight: float = 0.0  # Fraction of traffic (0.0 to 1.0) routed to canary

    def __post_init__(self) -> None:
        if not (0.0 <= self.canary_weight <= 1.0):
            raise ValueError("canary_weight must be between 0.0 and 1.0")


class CanaryRouter:
    """Thread-safe canary router supporting traffic splitting and instant rollback."""

    def __init__(
        self,
        registry: ModelRegistry,
        canary: ModelHandle | None = None,
        config: CanaryConfig | None = None,
    ) -> None:
        self._registry = registry
        self._canary = canary
        self._config = config or CanaryConfig(canary_weight=0.0)
        self._lock = RLock()

    @property
    def canary_handle(self) -> ModelHandle | None:
        """Return current canary model handle if configured."""
        with self._lock:
            return self._canary

    @property
    def config(self) -> CanaryConfig:
        """Return active canary configuration."""
        with self._lock:
            return self._config

    def set_canary(self, canary: ModelHandle, weight: float) -> None:
        """Configure or update canary model handle and traffic weight."""
        if not canary.version:
            raise ValueError("canary model version must not be empty")
        config = CanaryConfig(canary_weight=weight)
        with self._lock:
            self._canary = canary
            self._config = config

    def rollback(self) -> None:
        """Instantly divert 100% traffic back to primary active model."""
        with self._lock:
            self._canary = None
            self._config = CanaryConfig(canary_weight=0.0)

    def promote_canary(self) -> None:
        """Promote current canary to become the active primary model in registry."""
        with self._lock:
            if self._canary is None:
                raise RuntimeError("no active canary to promote")
            self._registry.replace(self._canary)
            self._canary = None
            self._config = CanaryConfig(canary_weight=0.0)

    def select_handle(self, seed: float | None = None) -> ModelHandle:
        """Select model handle for request based on canary traffic split."""
        with self._lock:
            primary = self._registry.active()
            canary = self._canary
            weight = self._config.canary_weight

        if canary is None or weight <= 0.0:
            return primary

        val = seed if seed is not None else random.random()
        if val < weight:
            return canary
        return primary
