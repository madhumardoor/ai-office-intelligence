from pathlib import Path

import pytest

from app.ingestion.loaders import IngestionFileError, load_file
from app.ingestion.schemas import CompanyRow


def _write(tmp_path: Path, content: str, name: str = "f.csv") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_headers_are_aliased(tmp_path: Path) -> None:
    p = _write(tmp_path, "Company,Website,Bogus\nAcme,acme.example,1\n")
    loaded = load_file(p)
    assert loaded.rows[0][1]["company_name"] == "Acme"
    assert loaded.unknown_columns == ["Bogus"]


def test_missing_required_column_rejected(tmp_path: Path) -> None:
    with pytest.raises(IngestionFileError, match="must contain"):
        load_file(_write(tmp_path, "Website,City\nx.example,Pune\n"))


def test_duplicate_canonical_columns_rejected(tmp_path: Path) -> None:
    with pytest.raises(IngestionFileError, match="duplicate column"):
        load_file(_write(tmp_path, "Company,Name\nA,B\n"))


def test_wrong_extension_rejected(tmp_path: Path) -> None:
    with pytest.raises(IngestionFileError, match="unsupported"):
        load_file(_write(tmp_path, "x", "f.txt"))


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(IngestionFileError, match="not found"):
        load_file(tmp_path / "nope.csv")


def test_row_normalization_and_validation() -> None:
    row = CompanyRow.from_raw(1, {
        "company_name": "Nova Labs Pvt Ltd",
        "website": "www.Nova.example/",
        "city": "Bangalore",
        "employee_count": "1,200",
        "tenant_phone": "bad",
    })
    assert (row.normalized_name, row.domain, row.city, row.employee_count) == (
        "nova labs", "nova.example", "Bengaluru", 1200,
    )
    assert row.tenant_phone is None


def test_row_rejects_bad_coordinates() -> None:
    with pytest.raises(ValueError):
        CompanyRow.from_raw(1, {"company_name": "X", "latitude": "95", "longitude": "77"})
    with pytest.raises(ValueError):
        CompanyRow.from_raw(1, {"company_name": "X", "latitude": "12.9"})


def test_row_requires_a_name() -> None:
    with pytest.raises(ValueError, match="missing"):
        CompanyRow.from_raw(1, {"website": "x.example"})