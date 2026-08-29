import pytest

from evolve.budget import (
    BudgetOverrun,
    BudgetService,
    BudgetTransactionConflict,
    InvalidRefund,
)
from evolve.types import BudgetLedger, InvariantViolation


def test_debit_retry_is_idempotent_but_changed_operation_conflicts():
    ledger = BudgetService.create(
        {"tokens": 100, "verifier_calls": 4}, identity=("run", "epoch0")
    )
    debited = BudgetService.debit(
        ledger, resource="tokens", amount=30, transaction_key="branch-1:tokens"
    )
    retried = BudgetService.debit(
        debited, resource="tokens", amount=30, transaction_key="branch-1:tokens"
    )
    assert retried is debited
    assert debited.consumed("tokens") == 30
    assert debited.remaining("tokens") == 70
    with pytest.raises(BudgetTransactionConflict):
        BudgetService.debit(
            debited, resource="tokens", amount=31,
            transaction_key="branch-1:tokens",
        )
    with pytest.raises(BudgetTransactionConflict):
        BudgetService.debit(
            debited, resource="verifier_calls", amount=1,
            transaction_key="branch-1:tokens",
        )


def test_each_resource_has_an_independent_hard_limit():
    ledger = BudgetService.create({"tokens": 100, "verifier_calls": 2})
    ledger = BudgetService.debit(
        ledger, resource="tokens", amount=100, transaction_key="tokens-all"
    )
    ledger = BudgetService.debit(
        ledger, resource="verifier_calls", amount=1, transaction_key="verify-one"
    )
    assert ledger.remaining("tokens") == 0
    assert ledger.remaining("verifier_calls") == 1
    with pytest.raises(BudgetOverrun):
        BudgetService.debit(
            ledger, resource="tokens", amount=1, transaction_key="tokens-overrun"
        )


def test_refund_is_idempotent_bounded_and_debit_specific():
    ledger = BudgetService.create({"tokens": 100, "wall_time_s": 20})
    ledger = BudgetService.debit(
        ledger, resource="tokens", amount=60, transaction_key="reservation"
    )
    refunded = BudgetService.refund(
        ledger,
        resource="tokens",
        amount=15,
        transaction_key="unused-1",
        debit_transaction_key="reservation",
    )
    retried = BudgetService.refund(
        refunded,
        resource="tokens",
        amount=15,
        transaction_key="unused-1",
        debit_transaction_key="reservation",
    )
    assert retried is refunded
    assert refunded.consumed("tokens") == 45
    with pytest.raises(InvalidRefund):
        BudgetService.refund(
            refunded,
            resource="tokens",
            amount=46,
            transaction_key="unused-too-much",
            debit_transaction_key="reservation",
        )
    with pytest.raises(InvalidRefund):
        BudgetService.refund(
            refunded,
            resource="wall_time_s",
            amount=1,
            transaction_key="wrong-resource",
            debit_transaction_key="reservation",
        )


def test_budget_ledger_round_trip_retains_transactions_and_balances():
    ledger = BudgetService.create({"tokens": 10})
    ledger = BudgetService.debit(
        ledger, resource="tokens", amount=8, transaction_key="debit"
    )
    ledger = BudgetService.refund(
        ledger,
        resource="tokens",
        amount=3,
        transaction_key="refund",
        debit_transaction_key="debit",
    )
    restored = BudgetLedger.from_json(ledger.to_json())
    assert restored == ledger
    assert restored.consumed("tokens") == 5
    assert restored.remaining("tokens") == 5


def test_transaction_identity_is_bound_to_all_persisted_content():
    ledger = BudgetService.create({"tokens": 10})
    ledger = BudgetService.debit(
        ledger, resource="tokens", amount=4, transaction_key="debit"
    )
    payload = ledger.to_dict()
    payload["transactions"][0]["amount"] = 3
    with pytest.raises(InvariantViolation, match="transaction_id"):
        BudgetLedger.from_dict(payload)

    payload = ledger.to_dict()
    payload["transactions"][0]["ledger_id"] = BudgetService.create(
        {"tokens": 10}, identity=("other",)
    ).ledger_id
    with pytest.raises(InvariantViolation, match="transaction_id|containing ledger"):
        BudgetLedger.from_dict(payload)
