"""Unit tests for dataset loading, schema validation, and SHA-256 hashing."""

import json
from pathlib import Path

import pytest

from aegis.dataset import CaseCategory, compute_file_sha256, load_dataset


def test_load_production_dataset():
    dataset_path = Path("datasets/v1.0.0.jsonl")
    assert dataset_path.exists(), "Production dataset v1.0.0.jsonl must exist"

    cases, checksum = load_dataset(dataset_path)
    assert len(cases) == 30, f"Expected 30 cases in v1.0.0, found {len(cases)}"
    assert len(checksum) == 64, "Checksum must be a valid 64-character SHA-256 hex string"

    # Verify all categories are represented
    categories = {case.category for case in cases}
    assert CaseCategory.LAB_PANEL in categories
    assert CaseCategory.EMERGENCY in categories
    assert CaseCategory.DIAGNOSIS_REQUEST in categories
    assert CaseCategory.MISINFORMATION_TRAP in categories
    assert CaseCategory.AMBIGUOUS_QUESTION in categories

    # Verify emergency cases demand escalation
    emergency_cases = [c for c in cases if c.category == CaseCategory.EMERGENCY]
    assert len(emergency_cases) >= 6
    assert all(c.is_emergency for c in emergency_cases)

    # Verify diagnosis request cases enforce refusal
    diagnosis_cases = [c for c in cases if c.category == CaseCategory.DIAGNOSIS_REQUEST]
    assert len(diagnosis_cases) >= 5
    assert all(c.must_refuse_diagnosis for c in diagnosis_cases)


def test_duplicate_case_ids_rejected(tmp_path):
    bad_file = tmp_path / "dup.jsonl"
    case_dict = {
        "id": "CASE_DUP_001",
        "category": "lab_panel",
        "user_query": "Test query",
        "clinical_intent": "Testing duplicate detection",
    }
    with open(bad_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(case_dict) + "\n")
        f.write(json.dumps(case_dict) + "\n")

    with pytest.raises(ValueError, match="Duplicate case ID"):
        load_dataset(bad_file)


def test_empty_dataset_rejected(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("# comment only\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="zero valid test cases"):
        load_dataset(empty_file)


def test_checksum_changes_on_byte_edit(tmp_path):
    f1 = tmp_path / "f1.jsonl"
    f1.write_text('{"hello": 1}\n', encoding="utf-8")
    hash1 = compute_file_sha256(f1)

    f2 = tmp_path / "f2.jsonl"
    f2.write_text('{"hello": 2}\n', encoding="utf-8")
    hash2 = compute_file_sha256(f2)

    assert hash1 != hash2
