"""Registered, versioned executable options bound to their frozen specs.

An :class:`OptionRegistry` is an immutable mapping from a content-addressed
``option_id`` to the Python state-machine class that can execute it.  Only
specs registered here may be frozen into a :class:`~evolve.types.BranchSpec`;
this keeps "an option is a registered executable state machine, not a prompt
label" true at the boundary where allocation arms are turned into branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Tuple, Type

from evolve.types import Role

from .base import ExecutableOption, OptionError, validate_option_spec_identity
from evolve.types import OptionSpec


class OptionRegistryError(OptionError):
    """A registry cannot resolve or admit a requested executable option."""


@dataclass(frozen=True)
class OptionRegistry:
    """Immutable option_id -> (spec, implementation) resolution table."""

    specs: Mapping[str, OptionSpec] = field(default_factory=dict)
    implementations: Mapping[str, Type[ExecutableOption]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", dict(self.specs))
        object.__setattr__(self, "implementations", dict(self.implementations))
        if set(self.specs) != set(self.implementations):
            raise OptionRegistryError("specs and implementations must share option_ids")
        for option_id, spec in self.specs.items():
            if spec.option_id != option_id:
                raise OptionRegistryError(f"option_id key mismatch for {option_id}")
            validate_option_spec_identity(spec)
            impl = self.implementations[option_id]
            if not (isinstance(impl, type) and issubclass(impl, ExecutableOption)):
                raise OptionRegistryError(f"{option_id} implementation must subclass ExecutableOption")
            if impl.STATE_MACHINE != spec.state_machine or impl.BEHAVIOR_VERSION != spec.version:
                raise OptionRegistryError(
                    f"{option_id} implementation does not match its spec identity"
                )

    def register(
        self, spec: OptionSpec, implementation: Type[ExecutableOption]
    ) -> "OptionRegistry":
        validate_option_spec_identity(spec)
        existing = self.specs.get(spec.option_id)
        if existing is not None:
            if existing.to_dict() != spec.to_dict():
                raise OptionRegistryError(f"option_id collision for {spec.option_id}")
            return self
        specs = dict(self.specs)
        implementations = dict(self.implementations)
        specs[spec.option_id] = spec
        implementations[spec.option_id] = implementation
        return replace(self, specs=specs, implementations=implementations)

    def spec(self, option_id: str) -> OptionSpec:
        try:
            return self.specs[option_id]
        except KeyError as exc:
            raise OptionRegistryError(f"unknown option_id {option_id!r}") from exc

    def create(self, option_id: str) -> ExecutableOption:
        """Instantiate a fresh, stateless executable bound to its frozen spec."""

        spec = self.spec(option_id)
        implementation = self.implementations[option_id]
        return implementation(spec)

    def option_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self.specs))

    def eligible_for(self, *, role: Role, harness_id: str) -> Tuple[str, ...]:
        """Return every registered option a role may initiate under one harness."""

        owner = role if isinstance(role, Role) else Role(role)
        return tuple(
            option_id
            for option_id in self.option_ids()
            if owner in self.specs[option_id].allowed_roles
            and harness_id in self.specs[option_id].harness_eligibility
        )


__all__ = ["OptionRegistry", "OptionRegistryError"]
