from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class ManifestIssue:
    code: str
    message: str
    subject: str = ""


class ManifestValidationError(ValueError):
    def __init__(self, issues: Iterable[ManifestIssue]):
        self.issues = tuple(sorted(issues))
        super().__init__(
            "\n".join(
                f"[{issue.code}] {issue.subject}: {issue.message}"
                for issue in self.issues
            )
        )
