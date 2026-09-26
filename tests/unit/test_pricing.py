from app.llm.pricing import PriceTable


def test_estimate_from_table() -> None:
    table = PriceTable.from_json(
        '{"m": {"input_per_million": 1.0, "output_per_million": 4.0}}'
    )

    assert table.estimate(
        "m",
        1_000_000,
        500_000,
    ) == 3.0


def test_unknown_model_or_missing_tokens_is_none_not_guessed() -> None:
    table = PriceTable.from_json(
        '{"m": {"input_per_million": 1.0, "output_per_million": 4.0}}'
    )

    assert table.estimate(
        "other",
        10,
        10,
    ) is None

    assert table.estimate(
        "m",
        None,
        10,
    ) is None


def test_empty_or_invalid_json_disables_costing_without_raising() -> None:
    assert PriceTable.from_json(
        ""
    ).estimate("m", 1, 1) is None

    assert PriceTable.from_json(
        "{not json"
    ).estimate("m", 1, 1) is None

    assert PriceTable.from_json(
        '{"m": {"input_per_million": "x"}}'
    ).estimate("m", 1, 1) is None