"""Tests for conversation ID encode/decode."""

from agent_runtime.ids import decode, encode


def test_encode_decode_roundtrip():
    for i in [1, 42, 100, 99999]:
        assert decode(encode(i)) == i


def test_encode_returns_string():
    result = encode(1)
    assert isinstance(result, str)
    assert len(result) >= 8  # min_length=8


def test_encode_different_ids_different_results():
    assert encode(1) != encode(2)


def test_decode_invalid_raises():
    import pytest

    with pytest.raises(ValueError, match="Invalid conversation ID"):
        decode("not_a_valid_id!!!")
