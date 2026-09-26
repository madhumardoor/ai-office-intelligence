import uuid

from app.retrieval.fusion import rrf_fuse
from app.retrieval.vector import Hit

A, B, C = (uuid.UUID(int=i) for i in (1, 2, 3))


def test_document_found_by_both_methods_outranks_single_method_top_hit() -> None:
    fused = rrf_fuse({"vector": [Hit(A, 1), Hit(B, 2)], "keyword": [Hit(B, 1), Hit(C, 2)]})
    assert fused[0][0] == B
    assert fused[0][2] == {"vector": 2, "keyword": 1}


def test_single_method_preserves_order() -> None:
    fused = rrf_fuse({"vector": [Hit(A, 1), Hit(B, 2), Hit(C, 3)]})
    assert [f[0] for f in fused] == [A, B, C]


def test_weights_shift_ranking() -> None:
    lists = {"vector": [Hit(A, 1)], "keyword": [Hit(B, 1)]}
    assert rrf_fuse(lists, weights={"vector": 2.0})[0][0] == A
    assert rrf_fuse(lists, weights={"keyword": 2.0})[0][0] == B


def test_deterministic_tiebreak_and_empty() -> None:
    lists = {"vector": [Hit(B, 1)], "keyword": [Hit(A, 1)]}
    assert [f[0] for f in rrf_fuse(lists)] == [A, B]
    assert rrf_fuse({}) == []