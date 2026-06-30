from __future__ import annotations

import pytest

from oxyz import _remote


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("s3://bucket/train.xyz", True),
        ("gs://bucket/train.xyz", True),
        ("az://acct/container/train.xyz", True),
        ("/local/train.xyz", False),
        ("train.xyz", False),
        ("file:///local/train.xyz", False),  # local file URL is not "remote" here
    ],
)
def test_is_remote(path, expected):
    assert _remote.is_remote(path) is expected


def test_missing_obstore_raises_helpful_error(monkeypatch):
    monkeypatch.setattr(_remote, "_import_obstore", _remote._raise_missing)
    with pytest.raises(ImportError, match=r"oxyz\[s3\]"):
        _remote.open_source(
            "s3://bucket/train.xyz",
            compression="infer",
            member=None,
            storage_options=None,
        )
