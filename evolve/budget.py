"""Pure, idempotent multi-resource budget accounting."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

from evolve.ids import content_id
from evolve.types import (
    BudgetLedger,
    BudgetTransaction,
    BudgetTransactionKind,
    FrozenDict,
    InvariantViolation,
)


class BudgetError(ValueError):
    """Base class for budget transaction failures."""


class BudgetOverrun(BudgetError):
    """A debit would consume more of one resource than was reserved."""


class BudgetTransactionConflict(BudgetError):
    """An idempotency key was reused for a different operation."""


class InvalidRefund(BudgetError):
    """A refund is not backed by an existing unrefunded debit."""


def _amount(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BudgetError("amount must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise BudgetError("amount must be finite and positive")
    return value


def _key(value: Any, field_name: str = "transaction_key") -> str:
    if not isinstance(value, str) or not value.strip():
        raise BudgetError(f"{field_name} must be a non-empty string")
    return value


def _resource(ledger: BudgetLedger, resource: Any) -> str:
    if not isinstance(resource, str) or not resource.strip():
        raise BudgetError("resource must be a non-empty string")
    if resource not in ledger.limits:
        raise BudgetError(f"unknown budget resource {resource!r}")
    return resource


class BudgetService:
    """Functional transaction service over immutable :class:`BudgetLedger`.

    Retrying the exact same operation with the same key returns the identical
    ledger object.  Reusing a key with a changed resource, amount, kind, or debit
    reference is rejected, so recovery cannot double count or silently mutate a
    durable assignment.
    """

    @staticmethod
    def create(
        limits: Mapping[str, float],
        *,
        identity: Sequence[Any] = (),
        ledger_id: Optional[str] = None,
    ) -> BudgetLedger:
        frozen_limits = FrozenDict(dict(limits))
        if ledger_id is None:
            ledger_id = content_id(
                "budget",
                {"identity": list(identity), "limits": dict(frozen_limits)},
            )
        return BudgetLedger(ledger_id=ledger_id, limits=frozen_limits)

    @staticmethod
    def _by_key(ledger: BudgetLedger, transaction_key: str) -> Optional[BudgetTransaction]:
        return next(
            (
                transaction
                for transaction in ledger.transactions
                if transaction.transaction_key == transaction_key
            ),
            None,
        )

    @staticmethod
    def _append(
        ledger: BudgetLedger, transaction: BudgetTransaction
    ) -> BudgetLedger:
        updated = replace(
            ledger, transactions=ledger.transactions + (transaction,)
        )
        # Schema metadata uses init=False for Python 3.8/3.9-compatible frozen
        # dataclass inheritance.  Preserve migrated unknown fields across a pure
        # ledger update instead of dropping them at the first debit.
        object.__setattr__(updated, "schema_version", ledger.schema_version)
        object.__setattr__(updated, "extensions", ledger.extensions)
        return updated

    @staticmethod
    def debit(
        ledger: BudgetLedger,
        *,
        resource: str,
        amount: float,
        transaction_key: str,
    ) -> BudgetLedger:
        resource = _resource(ledger, resource)
        amount = _amount(amount)
        transaction_key = _key(transaction_key)
        existing = BudgetService._by_key(ledger, transaction_key)
        if existing is not None:
            if (
                existing.kind == BudgetTransactionKind.DEBIT
                and existing.resource == resource
                and float(existing.amount) == amount
                and existing.debit_transaction_key is None
            ):
                return ledger
            raise BudgetTransactionConflict(
                f"transaction key {transaction_key!r} was already used for a different operation"
            )

        remaining = ledger.remaining(resource)
        if amount > remaining + 1e-12:
            raise BudgetOverrun(
                f"debit {amount:g} {resource} exceeds remaining budget {remaining:g}"
            )
        payload = {
            "ledger_id": ledger.ledger_id,
            "transaction_key": transaction_key,
            "resource": resource,
            "kind": BudgetTransactionKind.DEBIT.value,
            "amount": amount,
        }
        transaction = BudgetTransaction(
            transaction_id=content_id("budget_transaction", payload),
            ledger_id=ledger.ledger_id,
            transaction_key=transaction_key,
            resource=resource,
            kind=BudgetTransactionKind.DEBIT,
            amount=amount,
        )
        try:
            return BudgetService._append(ledger, transaction)
        except InvariantViolation as exc:
            raise BudgetOverrun(str(exc)) from exc

    @staticmethod
    def refund(
        ledger: BudgetLedger,
        *,
        resource: str,
        amount: float,
        transaction_key: str,
        debit_transaction_key: str,
    ) -> BudgetLedger:
        resource = _resource(ledger, resource)
        amount = _amount(amount)
        transaction_key = _key(transaction_key)
        debit_transaction_key = _key(debit_transaction_key, "debit_transaction_key")
        existing = BudgetService._by_key(ledger, transaction_key)
        if existing is not None:
            if (
                existing.kind == BudgetTransactionKind.REFUND
                and existing.resource == resource
                and float(existing.amount) == amount
                and existing.debit_transaction_key == debit_transaction_key
            ):
                return ledger
            raise BudgetTransactionConflict(
                f"transaction key {transaction_key!r} was already used for a different operation"
            )

        debit = BudgetService._by_key(ledger, debit_transaction_key)
        if debit is None or debit.kind != BudgetTransactionKind.DEBIT:
            raise InvalidRefund(
                f"refund references missing debit key {debit_transaction_key!r}"
            )
        if debit.resource != resource:
            raise InvalidRefund("refund resource does not match its debit")
        already_refunded = sum(
            float(transaction.amount)
            for transaction in ledger.transactions
            if (
                transaction.kind == BudgetTransactionKind.REFUND
                and transaction.debit_transaction_key == debit_transaction_key
            )
        )
        refundable = float(debit.amount) - already_refunded
        if amount > refundable + 1e-12:
            raise InvalidRefund(
                f"refund {amount:g} exceeds unrefunded debit amount {refundable:g}"
            )
        payload = {
            "ledger_id": ledger.ledger_id,
            "transaction_key": transaction_key,
            "resource": resource,
            "kind": BudgetTransactionKind.REFUND.value,
            "amount": amount,
            "debit_transaction_key": debit_transaction_key,
        }
        transaction = BudgetTransaction(
            transaction_id=content_id("budget_transaction", payload),
            ledger_id=ledger.ledger_id,
            transaction_key=transaction_key,
            resource=resource,
            kind=BudgetTransactionKind.REFUND,
            amount=amount,
            debit_transaction_key=debit_transaction_key,
        )
        try:
            return BudgetService._append(ledger, transaction)
        except InvariantViolation as exc:
            raise InvalidRefund(str(exc)) from exc


__all__ = [
    "BudgetError",
    "BudgetOverrun",
    "BudgetTransactionConflict",
    "InvalidRefund",
    "BudgetService",
]
