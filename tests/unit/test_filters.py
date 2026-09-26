import uuid
from datetime import date

import pytest
from pydantic import ValidationError

from app.retrieval.filters import RetrievalFilters


def test_empty_filters_produce_no_sql() -> None:
    assert RetrievalFilters().to_sql() == ("", {})


def test_filters_build_parameterised_sql_only() -> None:
    cid = uuid.uuid4()
    f = RetrievalFilters(company_ids=(cid,), document_types=("news",), published_after=date(2026, 1, 1))
    sql, params = f.to_sql("c")
    assert "c.company_id = ANY(:f_company_ids)" in sql
    assert "'news'" not in sql and "2026" not in sql
    assert params["f_doc_types"] == ["news"]


def test_unknown_document_type_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilters(document_types=("news'; DROP TABLE x;--",))


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilters(evil="x")  # type: ignore[call-arg]