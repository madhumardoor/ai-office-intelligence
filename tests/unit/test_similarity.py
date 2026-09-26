from app.entity_resolution.similarity import Candidate, name_similarity, score_pair


def test_same_domain_is_hard_match_even_with_different_names() -> None:
    a = Candidate("nova labs", "nova.example", None, "Bengaluru")
    b = Candidate("nova", "nova.example", None, "Bengaluru")
    s = score_pair(a, b)
    assert s.hard_match and not s.hard_block
    assert s.confidence >= 0.98


def test_different_domain_and_linkedin_hard_blocks_despite_identical_name() -> None:
    a = Candidate("acme", "acme-one.example", "https://www.linkedin.com/company/acme-one")
    b = Candidate("acme", "acme-two.example", "https://www.linkedin.com/company/acme-two")
    s = score_pair(a, b)
    assert s.hard_block and not s.hard_match


def test_identical_name_different_domain_lands_in_review_band_not_auto_merge() -> None:
    a = Candidate("zephyr", "zephyr.example", None, "Bengaluru")
    b = Candidate("zephyr", "zephyr-systems.example", None, "Bengaluru")
    s = score_pair(a, b)
    assert 0.75 <= s.confidence <= 0.85
    assert not s.hard_match and not s.hard_block


def test_different_name_and_different_domain_stays_low() -> None:
    a = Candidate("nova labs", "nova.example", None, "Bengaluru")
    b = Candidate("helix cloud", "helix.example", None, "Bengaluru")
    assert score_pair(a, b).confidence < 0.75


def test_missing_features_are_excluded_not_penalised() -> None:
    a = Candidate("nova labs", None, None, None)
    b = Candidate("nova labs", None, None, None)
    assert score_pair(a, b).confidence == 1.0


def test_unrelated_names_score_low() -> None:
    a = Candidate("nova labs")
    b = Candidate("helix cloud")
    assert score_pair(a, b).confidence < 0.6


def test_short_names_are_capped() -> None:
    assert name_similarity("abc", "abd") <= 0.7


def test_features_are_recorded_for_explainability() -> None:
    s = score_pair(Candidate("a b c", "x.example"), Candidate("a b c", "x.example"))
    assert {"domain", "linkedin", "name", "city"} <= set(s.features)