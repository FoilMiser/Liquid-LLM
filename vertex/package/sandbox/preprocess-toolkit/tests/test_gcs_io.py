import io
import os

import pytest

from preprocess_toolkit import io as gcs_io


class FakeFile(io.BytesIO):
    def __init__(self, storage, path, mode, initial=b""):
        self.storage = storage
        self.path = path
        self.mode = mode
        super().__init__(initial)

    def close(self):
        if "w" in self.mode:
            self.storage[self.path] = self.getvalue()
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class FakeFS:
    def __init__(self):
        self.storage = {}

    def open(self, path, mode):
        if "r" in mode:
            data = self.storage.get(path, b"")
            return FakeFile(self.storage, path, mode, data)
        return FakeFile(self.storage, path, mode)

    def glob(self, pattern):
        # simple glob that returns all stored paths
        return list(self.storage.keys())


@pytest.fixture(autouse=True)
def patch_gcloud(monkeypatch):
    monkeypatch.setattr(gcs_io, "_run_gcloud", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    fake_fs = FakeFS()
    fake_fs.storage["bucket/object.txt"] = b"hello"
    monkeypatch.setattr(gcs_io, "_ensure_gcsfs", lambda: fake_fs)
    return fake_fs


def test_gcs_download_and_upload(tmp_path, patch_gcloud):
    local_file = tmp_path / "download.txt"
    gcs_io.gcs_to_local("gs://bucket/object.txt", str(local_file))
    assert local_file.read_text() == "hello"

    upload_file = tmp_path / "upload.txt"
    upload_file.write_text("goodbye")
    gcs_io.local_to_gcs(str(upload_file), "gs://bucket/new_object.txt")
    assert patch_gcloud.storage["bucket/new_object.txt"] == b"goodbye"


def test_list_gcs(tmp_path, patch_gcloud):
    patch_gcloud.storage["bucket/another.txt"] = b"x"
    file_path = tmp_path / "extra.txt"
    file_path.write_text("data")
    gcs_io.local_to_gcs(str(file_path), "gs://bucket/new_object.txt")
    items = gcs_io.list_gcs("gs://bucket/*")
    assert "gs://bucket/object.txt" in items
    assert "gs://bucket/new_object.txt" in items
    assert "gs://bucket/another.txt" in items
