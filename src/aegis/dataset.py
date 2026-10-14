"""Dataset schemas, parser, and SHA-256 checksum verification."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class CaseCategory(str, Enum):
    LAB_PANEL = "lab_panel"
    EMERGENCY = "emergency"
    DIAGNOSIS_REQUEST = "diagnosis_request"
    MISINFORMATION_TRAP = "misinformation_trap"
    AMBIGUOUS_QUESTION = "ambiguous_question"


class Biomarker(BaseModel):
    name: str = Field(..., description="Standardized name of the biomarker")
    value: float | int | str = Field(..., description="Measured lab value")
    unit: str = Field(..., description="Measurement unit (e.g. mg/dL, mmol/L, ng/mL)")
    reference_range: str = Field(..., description="Clinical normal reference interval")

    @property
    def value_str(self) -> str:
        return str(self.value)


class TestCase(BaseModel):
    __test__ = False  # Prevent pytest from treating this data model as a test suite
    id: str = Field(..., description="Unique case identifier (e.g., CASE_001_LIPIDS)")
    category: CaseCategory = Field(..., description="Clinical test case category")
    user_query: str = Field(..., description="Patient input prompt")
    patient_age: int | None = Field(default=None, description="Patient age in years")
    patient_sex: str | None = Field(default=None, description="Patient biological sex")
    lab_values: list[Biomarker] = Field(default_factory=list, description="Verified lab values in panel")
    is_emergency: bool = Field(default=False, description="Whether case demands urgent medical escalation")
    must_refuse_diagnosis: bool = Field(default=False, description="Whether case must refuse diagnostic claim")
    forbidden_claims: list[str] = Field(default_factory=list, description="Phrases that constitute a hard fail")
    expected_topics: list[str] = Field(default_factory=list, description="Topics that should be touched upon")
    clinical_intent: str = Field(..., description="Target safety rationale for this case")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Case ID cannot be empty")
        return v


def compute_file_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of dataset file to ensure reproducible benchmarks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset(file_path: str | Path) -> tuple[list[TestCase], str]:
    """Loads and validates a JSONL dataset, returning the test cases and file SHA-256 hash."""
    path = Path(file_path)
    checksum = compute_file_sha256(path)

    cases: list[TestCase] = []
    seen_ids: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                case = TestCase.model_validate(data)
                if case.id in seen_ids:
                    raise ValueError(f"Duplicate case ID detected: '{case.id}' at line {line_num}")
                seen_ids.add(case.id)
                cases.append(case)
            except Exception as e:
                raise ValueError(f"Failed to parse test case at {path}:{line_num}: {e}") from e

    if not cases:
        raise ValueError(f"Dataset at {path} contains zero valid test cases.")

    return cases, checksum
