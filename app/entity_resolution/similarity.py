"""Pure, deterministic similarity features. No I/O."""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


@dataclass(frozen=True, slots=True)
class Candidate:
    """Identity + optional physical/operator context for one incoming/company record."""

    normalized_name: str
    domain: str | None = None
    linkedin_url: str | None = None
    city: str | None = None
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    # Set only when the row explicitly identifies a coworking operator.
    # This prevents operator-branch logic from being applied to ordinary tenants.
    operator_name: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureWeights:
    domain: float = 0.45
    linkedin: float = 0.25
    name: float = 0.25
    city: float = 0.05


@dataclass(slots=True)
class MatchScore:
    confidence: float
    features: dict[str, float | bool | None] = field(default_factory=dict)
    hard_match: bool = False
    hard_block: bool = False


def name_similarity(a: str, b: str) -> float:
    """Blend Jaro-Winkler and token-set similarity."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    jw = JaroWinkler.similarity(a, b)
    ts = fuzz.token_set_ratio(a, b) / 100.0
    score = 0.5 * jw + 0.5 * ts

    if min(len(a), len(b)) <= 3:
        score = min(score, 0.7)

    return round(score, 4)


def score_pair(
    a: Candidate,
    b: Candidate,
    w: FeatureWeights = FeatureWeights(),
) -> MatchScore:
    """Score two candidates using available identity features."""
    features: dict[str, float | bool | None] = {}
    parts: list[tuple[float, float]] = []
    hard_match = False
    hard_block = False

    if a.domain and b.domain:
        same = a.domain == b.domain
        features["domain"] = 1.0 if same else 0.0
        # A domain mismatch is negative evidence, but do not let it
        # mathematically drag a name-only candidate out of the review band.
        # The mismatch is still recorded above and enforced separately by
        # the hard-block rule when BOTH strong identifiers conflict.
        if same:
            parts.append((w.domain, 1.0))
            hard_match = True
    else:
        features["domain"] = None

    if a.linkedin_url and b.linkedin_url:
        same = a.linkedin_url == b.linkedin_url
        features["linkedin"] = 1.0 if same else 0.0
        parts.append((w.linkedin, features["linkedin"]))  # type: ignore[arg-type]
        hard_match |= same
    else:
        features["linkedin"] = None

    # Both strong identifiers present and both differ => must not merge.
    if features["domain"] == 0.0 and features["linkedin"] == 0.0:
        hard_block = True

    ns = name_similarity(a.normalized_name, b.normalized_name)
    features["name"] = ns
    parts.append((w.name, ns))

    if a.city and b.city:
        features["city"] = 1.0 if a.city.lower() == b.city.lower() else 0.0
        parts.append((w.city, features["city"]))  # type: ignore[arg-type]
    else:
        features["city"] = None

    total_w = sum(pw for pw, _ in parts)
    confidence = (
        sum(pw * s for pw, s in parts) / total_w
        if total_w
        else 0.0
    )

    if hard_match and not hard_block:
        confidence = max(confidence, 0.98)

    # Different domains without an exact strong match are never allowed
    # to become an automatic fuzzy match.
    if features["domain"] == 0.0 and not hard_match:
        confidence = min(confidence, 0.85)

    return MatchScore(
        round(confidence, 4),
        features,
        hard_match and not hard_block,
        hard_block,
    )
