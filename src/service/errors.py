"""Service-layer error contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ServiceError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def serialize_service_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ServiceError):
        return exc.to_dict()
    return {
        "code": "internal_error",
        "message": str(exc) or exc.__class__.__name__,
        "details": {"error_type": exc.__class__.__name__},
    }
