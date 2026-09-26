"""Claim, expert, and third-party plugin registries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from types import MappingProxyType
from typing import Any

from .protocols import ECGExpert
from .schema import ClaimSpec


class FrameworkRegistry:
    """Validated clinical ontology and expert routing table."""

    def __init__(self) -> None:
        self._claims: dict[str, ClaimSpec] = {}
        self._experts: dict[str, ECGExpert] = {}

    @property
    def claims(self) -> Mapping[str, ClaimSpec]:
        return MappingProxyType(self._claims)

    @property
    def experts(self) -> Mapping[str, ECGExpert]:
        return MappingProxyType(self._experts)

    def register_claim(self, spec: ClaimSpec, *, replace: bool = False) -> None:
        if spec.claim_id in self._claims and not replace:
            raise KeyError(f"claim already registered: {spec.claim_id}")
        self._claims[spec.claim_id] = spec

    def register_expert(
        self, name: str, expert: ECGExpert, *, replace: bool = False
    ) -> None:
        if name in self._experts and not replace:
            raise KeyError(f"expert already registered: {name}")
        self._experts[name] = expert

    def validate(self) -> None:
        if not self._claims:
            raise ValueError("framework has no registered claims")
        missing = sorted(
            {spec.expert for spec in self._claims.values()} - set(self._experts)
        )
        if missing:
            raise ValueError(f"claims reference missing experts: {', '.join(missing)}")
        for spec in self._claims.values():
            if not 0.0 <= spec.priority <= 1.0:
                raise ValueError(f"claim priority must be in [0, 1]: {spec.claim_id}")


Factory = Callable[..., Any]


class ComponentRegistry:
    """Named component factories with optional Python entry-point discovery."""

    ENTRY_POINT_GROUP = "ecg_framework.plugins"

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], Factory] = {}

    def register(
        self, kind: str, name: str, factory: Factory, *, replace: bool = False
    ) -> None:
        key = (kind, name)
        if key in self._factories and not replace:
            raise KeyError(f"component already registered: {kind}/{name}")
        self._factories[key] = factory

    def create(self, kind: str, name: str, **kwargs: Any) -> Any:
        try:
            factory = self._factories[(kind, name)]
        except KeyError as error:
            available = ", ".join(
                f"{item_kind}/{item_name}"
                for item_kind, item_name in sorted(self._factories)
            )
            raise KeyError(
                f"unknown component {kind}/{name}; available: {available or 'none'}"
            ) from error
        return factory(**kwargs)

    def available(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._factories))

    def discover(self) -> tuple[str, ...]:
        """Load plugins whose entry point is ``register(registry)``."""

        loaded = []
        selected = entry_points().select(group=self.ENTRY_POINT_GROUP)
        for item in selected:
            register = item.load()
            if not callable(register):
                raise TypeError(f"plugin entry point is not callable: {item.name}")
            register(self)
            loaded.append(item.name)
        return tuple(sorted(loaded))


components = ComponentRegistry()
