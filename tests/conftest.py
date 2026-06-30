from __future__ import annotations

from typing import Any

import pytest


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
def s3_store(s3_server: str):
    """Fresh S3 bucket on the moto server, plus a put() helper and options dict.

    Returns ``(put, options)`` where:
    - ``put(key, data)`` uploads bytes to bucket ``test`` via boto3.
    - ``options`` is the storage_options dict to pass to oxyz read functions.

    A new bucket named ``test`` is created for each test (boto3 idempotent).
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
    client.create_bucket(Bucket="test")

    def put(key: str, data: bytes) -> None:
        client.put_object(Bucket="test", Key=key, Body=data)

    return put, options
