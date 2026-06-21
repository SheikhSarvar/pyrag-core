import io
from functools import lru_cache
from typing import BinaryIO

from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error

from app.core.config import get_settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger

logger = get_logger(__name__)


class MinIOClient:
    def __init__(self, client: Minio) -> None:
        self._client = client
        self._settings = get_settings()

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload_file(
        self,
        bucket: str,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a file-like object. Returns the object_name."""
        try:
            self._client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=data,
                length=length,
                content_type=content_type,
                metadata=metadata or {},
            )
            logger.info("Uploaded object", bucket=bucket, object=object_name, size=length)
            return object_name
        except S3Error as exc:
            raise StorageError(f"Upload failed: {exc}") from exc

    def upload_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload raw bytes."""
        return self.upload_file(
            bucket=bucket,
            object_name=object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
            metadata=metadata,
        )

    # ── Download ──────────────────────────────────────────────────────────────

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        """Download an object and return its content as bytes."""
        try:
            response = self._client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as exc:
            raise StorageError(f"Download failed: {exc}") from exc

    def download_stream(self, bucket: str, object_name: str) -> io.BytesIO:
        """Download and return a seekable BytesIO stream."""
        return io.BytesIO(self.download_bytes(bucket, object_name))

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_object(self, bucket: str, object_name: str) -> None:
        try:
            self._client.remove_object(bucket, object_name)
            logger.info("Deleted object", bucket=bucket, object=object_name)
        except S3Error as exc:
            raise StorageError(f"Delete failed: {exc}") from exc

    def delete_objects(self, bucket: str, object_names: list[str]) -> None:
        """Bulk delete."""
        try:
            errors = list(
                self._client.remove_objects(
                    bucket,
                    [DeleteObject(name) for name in object_names],
                )
            )
            if errors:
                raise StorageError(f"Bulk delete had {len(errors)} errors: {errors[0]}")
            logger.info("Bulk deleted objects", bucket=bucket, count=len(object_names))
        except S3Error as exc:
            raise StorageError(f"Bulk delete failed: {exc}") from exc

    # ── Presigned URLs ────────────────────────────────────────────────────────

    def presigned_get_url(
        self, bucket: str, object_name: str, expires_seconds: int = 3600
    ) -> str:
        from datetime import timedelta
        try:
            return self._client.presigned_get_object(
                bucket, object_name, expires=timedelta(seconds=expires_seconds)
            )
        except S3Error as exc:
            raise StorageError(f"Presigned URL failed: {exc}") from exc

    def presigned_put_url(
        self, bucket: str, object_name: str, expires_seconds: int = 900
    ) -> str:
        from datetime import timedelta
        try:
            return self._client.presigned_put_object(
                bucket, object_name, expires=timedelta(seconds=expires_seconds)
            )
        except S3Error as exc:
            raise StorageError(f"Presigned PUT URL failed: {exc}") from exc

    # ── Bucket management ─────────────────────────────────────────────────────

    def bucket_exists(self, bucket: str) -> bool:
        return self._client.bucket_exists(bucket)  # type: ignore[return-value]

    def make_bucket(self, bucket: str) -> None:
        try:
            self._client.make_bucket(bucket)
            logger.info("Created bucket", bucket=bucket)
        except S3Error as exc:
            raise StorageError(f"Bucket creation failed: {exc}") from exc

    def list_objects(
        self, bucket: str, prefix: str = "", recursive: bool = True
    ) -> list[str]:
        try:
            return [
                obj.object_name
                for obj in self._client.list_objects(bucket, prefix=prefix, recursive=recursive)
            ]
        except S3Error as exc:
            raise StorageError(f"List objects failed: {exc}") from exc

    def object_exists(self, bucket: str, object_name: str) -> bool:
        try:
            self._client.stat_object(bucket, object_name)
            return True
        except S3Error:
            return False

    # ── Path helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def raw_path(dataset_id: str, document_id: str, filename: str) -> str:
        return f"{dataset_id}/{document_id}/raw/{filename}"

    @staticmethod
    def processed_path(dataset_id: str, document_id: str, filename: str) -> str:
        return f"{dataset_id}/{document_id}/processed/{filename}"


@lru_cache
def get_minio_client() -> MinIOClient:
    settings = get_settings()
    client = Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_use_ssl,
    )
    return MinIOClient(client)
