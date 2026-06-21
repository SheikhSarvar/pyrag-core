"""
Unit tests for MinIOClient — Minio SDK is mocked, no real server needed.
"""
import io
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import StorageError
from app.services.storage.minio_client import MinIOClient


@pytest.fixture
def mock_minio() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(mock_minio: MagicMock) -> MinIOClient:
    return MinIOClient(mock_minio)


def test_upload_bytes_calls_put_object(client: MinIOClient, mock_minio: MagicMock) -> None:
    client.upload_bytes("raw", "ds/doc/file.pdf", b"hello pdf", "application/pdf")
    mock_minio.put_object.assert_called_once()
    call_kwargs = mock_minio.put_object.call_args
    assert call_kwargs.kwargs["bucket_name"] == "raw"
    assert call_kwargs.kwargs["object_name"] == "ds/doc/file.pdf"


def test_upload_raises_storage_error_on_s3_error(client: MinIOClient, mock_minio: MagicMock) -> None:
    from minio.error import S3Error
    mock_minio.put_object.side_effect = S3Error(
        code="NoSuchBucket", message="bucket missing",
        resource="raw", request_id="1", host_id="h", response=MagicMock()
    )
    with pytest.raises(StorageError):
        client.upload_bytes("raw", "obj", b"data")


def test_download_bytes_returns_content(client: MinIOClient, mock_minio: MagicMock) -> None:
    fake_response = MagicMock()
    fake_response.read.return_value = b"file content"
    mock_minio.get_object.return_value = fake_response
    result = client.download_bytes("raw", "ds/doc/file.pdf")
    assert result == b"file content"


def test_download_stream_is_seekable(client: MinIOClient, mock_minio: MagicMock) -> None:
    fake_response = MagicMock()
    fake_response.read.return_value = b"stream data"
    mock_minio.get_object.return_value = fake_response
    stream = client.download_stream("raw", "obj")
    assert isinstance(stream, io.BytesIO)
    assert stream.read() == b"stream data"


def test_delete_object_calls_remove(client: MinIOClient, mock_minio: MagicMock) -> None:
    client.delete_object("raw", "path/to/obj")
    mock_minio.remove_object.assert_called_once_with("raw", "path/to/obj")


def test_object_exists_true(client: MinIOClient, mock_minio: MagicMock) -> None:
    mock_minio.stat_object.return_value = MagicMock()
    assert client.object_exists("raw", "exists.pdf") is True


def test_object_exists_false_on_s3_error(client: MinIOClient, mock_minio: MagicMock) -> None:
    from minio.error import S3Error
    mock_minio.stat_object.side_effect = S3Error(
        code="NoSuchKey", message="not found",
        resource="raw", request_id="1", host_id="h", response=MagicMock()
    )
    assert client.object_exists("raw", "missing.pdf") is False


def test_raw_path_helper() -> None:
    path = MinIOClient.raw_path("ds1", "doc1", "report.pdf")
    assert path == "ds1/doc1/raw/report.pdf"


def test_processed_path_helper() -> None:
    path = MinIOClient.processed_path("ds1", "doc1", "report.txt")
    assert path == "ds1/doc1/processed/report.txt"


def test_bucket_exists(client: MinIOClient, mock_minio: MagicMock) -> None:
    mock_minio.bucket_exists.return_value = True
    assert client.bucket_exists("raw") is True
