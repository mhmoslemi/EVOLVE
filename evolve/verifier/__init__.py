"""Independent, payload-only scientific verification for EVOLVE."""

from .adapters import (
    LegacyProblemFallbackAdapter,
    ProblemScientificAdapter,
    ScientificProblemAdapter,
)
from .evidence import (
    ScientificVerificationResult,
    bounded_diagnostics,
    build_descriptor,
    descriptor_id,
    scientific_state_id,
    validate_descriptor_identity,
    validate_evidence_identity,
    validate_state_identity,
)
from .models import (
    ExecutionCapture,
    FailureClassification,
    PersistedAnswerPayload,
    VerificationDecision,
    VerificationPolicy,
    VerificationValidationError,
    classify_failure,
)
from .service import (
    VerificationServiceError,
    confirm_persisted_answer,
    verify_persisted_answer,
)

__all__ = [
    "ExecutionCapture",
    "FailureClassification",
    "LegacyProblemFallbackAdapter",
    "PersistedAnswerPayload",
    "ProblemScientificAdapter",
    "ScientificProblemAdapter",
    "ScientificVerificationResult",
    "VerificationDecision",
    "VerificationPolicy",
    "VerificationServiceError",
    "VerificationValidationError",
    "bounded_diagnostics",
    "build_descriptor",
    "classify_failure",
    "confirm_persisted_answer",
    "descriptor_id",
    "scientific_state_id",
    "validate_descriptor_identity",
    "validate_evidence_identity",
    "validate_state_identity",
    "verify_persisted_answer",
]
