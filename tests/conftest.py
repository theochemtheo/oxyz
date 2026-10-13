from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


class S3Fixture(NamedTuple):
    """A per-test bucket: `put(key, data)` uploads, `url(key)` names it, `options`
    is the storage_options dict.
    """

    put: Callable[[str, bytes], None]
    options: dict[str, Any]
    url: Callable[[str], str]


@pytest.fixture(scope="session")
def s3_server():
    """A session-scoped local S3 endpoint, started only when a test needs it."""
    moto_server = pytest.importorskip("moto.server")
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    yield endpoint
    server.stop()


@pytest.fixture
def s3_store(s3_server: str) -> S3Fixture:
    """A fresh bucket for one test, so objects never leak between tests."""
    boto3 = pytest.importorskip("boto3")
    pytest.importorskip("obstore")

    options: dict[str, Any] = {
        # S3Config keys
        "endpoint": s3_server,
        "region": "us-east-1",
        "access_key_id": "test",
        "secret_access_key": "test",
        "virtual_hosted_style_request": False,
        # Moto speaks plain HTTP.
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
