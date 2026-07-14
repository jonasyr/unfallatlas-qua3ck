from pathlib import Path

from unfallatlas.presentation.models import (
    BatchResult,
    ExportResult,
    Finding,
    NotebookStatus,
    Severity,
)


def test_enum_values_are_stable_machine_strings() -> None:
    assert {member.name: member.value for member in Severity} == {
        "INFO": "info",
        "WARNING": "warning",
        "ERROR": "error",
    }
    assert {member.name: member.value for member in NotebookStatus} == {
        "READY": "ready",
        "WIP": "wip",
        "PLACEHOLDER": "placeholder",
        "INVALID": "invalid",
    }


def test_finding_exposes_machine_code_and_strict_blocker() -> None:
    finding = Finding(
        code="UNEXECUTED_CELL",
        severity=Severity.WARNING,
        message="Code cell was never executed",
        cell_index=4,
        strict_blocker=True,
    )
    assert finding.code == "UNEXECUTED_CELL"
    assert finding.cell_index == 4
    assert finding.strict_blocker is True


def test_batch_result_succeeds_with_warnings() -> None:
    result = ExportResult(
        source=Path("notebooks/example.ipynb"),
        destination=Path("reports/presentation/notebooks/example.html"),
        status=NotebookStatus.READY,
        findings=(Finding("LARGE_OUTPUT", Severity.WARNING, "large"),),
        size_bytes=123,
        error=None,
    )
    assert BatchResult((result,)).successful is True
