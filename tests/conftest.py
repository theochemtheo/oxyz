from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, NamedTuple

import pytest


class S3Fixture(NamedTuple):
    """A per-test bucket on the moto server.

    ``put(key, data)`` uploads bytes, ``url(key)`` is the ``s3://`` URL for that
    key, and ``options`` is the storage_options dict for oxyz read functions.
    """

    put: Callable[[str, bytes], None]
    options: dict[str, Any]
    url: Callable[[str], str]


@pytest.fixture(scope="session")
def s3_server():
    """A local S3 endpoint via moto's threaded server.

    Session-scoped: one server for the whole test run, created only when a
    test that actually needs it is collected.
    """
    moto_server = pytest.importorskip("moto.server")
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    yield endpoint
    server.stop()


@pytest.fixture
def s3_store(s3_server: str) -> S3Fixture:
    """A fresh, uniquely-named bucket on the moto server for one test.

    Each test gets its own bucket, so objects never leak between tests sharing
    the session-scoped server.
    """
    boto3 = pytest.importorskip("boto3")
    pytest.importorskip("obstore")

    options: dict[str, Any] = {
        # S3Config keys
        "endpoint": s3_server,
        "region": "us-east-1",
        "access_key_id": "test",
        "secret_access_key": "test",
        "virtual_hosted_style_request": False,
        # ClientConfig key — must be separated by _build_store before passing
        # to obstore's from_url; moto speaks plain HTTP.
        "allow_http": True,
    }

    client = boto3.client(
        "s3",
        endpoint_url=s3_server,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    bucket = f"test-{uuid.uuid4().hex}"
    client.create_bucket(Bucket=bucket)

    def put(key: str, data: bytes) -> None:
        client.put_object(Bucket=bucket, Key=key, Body=data)

    return S3Fixture(put=put, options=options, url=lambda key: f"s3://{bucket}/{key}")
