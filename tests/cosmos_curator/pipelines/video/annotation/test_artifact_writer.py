# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for temporal annotation artifact publication."""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.pipelines.video.annotation import artifact_writer
from cosmos_curator.pipelines.video.annotation.artifact_writer import (
    TemporalAnnotationWriter,
)
from tests.cosmos_curator.core.utils.storage.conftest import FakeStorageClient


class RecordingStorageClient(FakeStorageClient):
    """In-memory storage client that records completed file uploads."""

    def __init__(self) -> None:
        """Initialize an empty client and upload log."""
        super().__init__()
        self.uploads: list[str] = []

    def upload_file(
        self,
        local_path: str,
        remote_path: StoragePrefix,
        _chunk_size: int = 100,
    ) -> None:
        """Record and store one completed file upload."""
        self.uploads.append(str(remote_path))
        super().upload_file(local_path, remote_path, _chunk_size)


def _write_chunk(path: Path, frame_start: int, frame_stop: int) -> None:
    np.savez(
        path,
        timestamps_ns=np.arange(frame_start, frame_stop, dtype=np.int64),
        normal=np.zeros((frame_stop - frame_start, 2, 3, 3), dtype=np.float16),
    )


def _metadata(*, frame_count: int = 65) -> dict[str, Any]:
    return {
        "arrays": {
            "normal": {
                "axes": "THWC",
                "dtype": "float16",
                "shape": [frame_count, 2, 3, 3],
            }
        },
        "alignment": {"fps_num": 15, "fps_den": 1, "source_frame_step": 2},
        "producer": {"id": "normalcrafter", "release": "normalcrafter-v1"},
    }


def _write_chunk_then_fail(
    writer: TemporalAnnotationWriter,
    clip_uuid: str,
) -> None:
    with writer.open_chunk(clip_uuid, 0, 2) as staging_path:
        _write_chunk(staging_path, 0, 2)
        message = "inference failed"
        raise RuntimeError(message)


def test_local_chunks_are_atomically_replaced_and_metadata_is_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local chunks use atomic replace and the metadata destination is last."""
    output_path = tmp_path / "annotations" / "normals" / "normalcrafter-v1"
    clip_uuid = str(uuid4())
    destinations: list[Path] = []
    real_replace = Path.replace

    def _record_replace(source: Path, destination: str | Path) -> Path:
        destinations.append(Path(destination))
        return real_replace(source, destination)

    monkeypatch.setattr(artifact_writer.Path, "replace", _record_replace)
    writer = TemporalAnnotationWriter(output_path)

    for start, stop in ((0, 32), (32, 64), (64, 65)):
        with writer.open_chunk(clip_uuid, start, stop) as staging_path:
            assert staging_path.parent == output_path / "chunks" / "v1" / clip_uuid
            _write_chunk(staging_path, start, stop)
        assert not staging_path.exists()

    metadata_location = writer.complete_clip(
        clip_uuid,
        frame_count=65,
        chunk_frames=32,
        metadata=_metadata(),
    )

    expected_destinations = [
        output_path / "chunks" / "v1" / clip_uuid / "frames-000000000-000000032.npz",
        output_path / "chunks" / "v1" / clip_uuid / "frames-000000032-000000064.npz",
        output_path / "chunks" / "v1" / clip_uuid / "frames-000000064-000000065.npz",
        output_path / "metas" / "v1" / f"{clip_uuid}.json",
    ]
    assert destinations == expected_destinations
    assert metadata_location == str(expected_destinations[-1])

    with np.load(expected_destinations[0]) as chunk:
        assert chunk["normal"].shape == (32, 2, 3, 3)
        assert chunk["timestamps_ns"].tolist() == list(range(32))

    document = json.loads(expected_destinations[-1].read_text(encoding="utf-8"))
    assert document["schema"] == "cosmos-curator.temporal-annotation/v1"
    assert document["clip_uuid"] == clip_uuid
    assert document["format"] == "npz"
    assert document["frame_count"] == 65
    assert document["chunk_frames"] == 32
    assert document["chunk_path_template"] == (
        f"chunks/v1/{clip_uuid}/frames-{{frame_start:09d}}-{{frame_stop:09d}}.npz"
    )
    assert document["metadata"] == _metadata()


def test_chunk_exception_publishes_neither_chunk_nor_metadata(tmp_path: Path) -> None:
    """An exception in the producer leaves no visible chunk or metadata."""
    output_path = tmp_path / "annotations" / "normals" / "normalcrafter-v1"
    clip_uuid = str(uuid4())
    writer = TemporalAnnotationWriter(output_path)

    with pytest.raises(RuntimeError, match="inference failed"):
        _write_chunk_then_fail(writer, clip_uuid)

    assert not (output_path / "chunks" / "v1" / clip_uuid / "frames-000000000-000000002.npz").exists()
    with pytest.raises(ValueError, match="no published chunks"):
        writer.complete_clip(
            clip_uuid,
            frame_count=2,
            chunk_frames=2,
            metadata=_metadata(),
        )
    assert not (output_path / "metas" / "v1" / f"{clip_uuid}.json").exists()


def test_empty_chunk_is_rejected(tmp_path: Path) -> None:
    """A producer must replace the initially empty staging file."""
    writer = TemporalAnnotationWriter(tmp_path / "annotations")
    clip_uuid = str(uuid4())

    with (
        pytest.raises(ValueError, match="must be non-empty"),
        writer.open_chunk(clip_uuid, 0, 1),
    ):
        pass

    assert not list((tmp_path / "annotations").rglob("*.npz"))


def test_complete_clip_rejects_non_contiguous_or_wrong_sized_ranges(tmp_path: Path) -> None:
    """Completion requires the exact ranges implied by frame and chunk counts."""
    output_path = tmp_path / "annotations"
    clip_uuid = str(uuid4())
    writer = TemporalAnnotationWriter(output_path)

    for start, stop in ((0, 32), (33, 65)):
        with writer.open_chunk(clip_uuid, start, stop) as staging_path:
            _write_chunk(staging_path, start, stop)

    with pytest.raises(ValueError, match="coverage mismatch"):
        writer.complete_clip(
            clip_uuid,
            frame_count=65,
            chunk_frames=32,
            metadata=_metadata(),
        )
    assert not (output_path / "metas" / "v1" / f"{clip_uuid}.json").exists()


def test_complete_clip_rejects_non_json_metadata_without_publishing(tmp_path: Path) -> None:
    """Non-finite or otherwise non-JSON metadata cannot become visible."""
    output_path = tmp_path / "annotations"
    clip_uuid = str(uuid4())
    writer = TemporalAnnotationWriter(output_path)
    with writer.open_chunk(clip_uuid, 0, 1) as staging_path:
        _write_chunk(staging_path, 0, 1)

    with pytest.raises(ValueError, match="finite JSON"):
        writer.complete_clip(
            clip_uuid,
            frame_count=1,
            chunk_frames=1,
            metadata={"invalid": float("nan")},
        )
    assert not (output_path / "metas" / "v1" / f"{clip_uuid}.json").exists()


@pytest.mark.parametrize(
    "output_path",
    [
        "s3://test-bucket/annotations/normals/normalcrafter-v1",
        "az://test-container/annotations/normals/normalcrafter-v1",
    ],
)
def test_remote_chunks_use_file_upload_and_metadata_is_uploaded_last(
    output_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3 and Azure publish staged files and upload the metadata last."""
    client = RecordingStorageClient()
    monkeypatch.setattr(storage_utils, "get_storage_client", lambda *_args, **_kwargs: client)
    clip_uuid = str(uuid4())
    writer = TemporalAnnotationWriter(output_path, tmp_dir=tmp_path)

    for start, stop in ((0, 2), (2, 3)):
        with writer.open_chunk(clip_uuid, start, stop) as staging_path:
            assert staging_path.parent == tmp_path
            _write_chunk(staging_path, start, stop)

    metadata_location = writer.complete_clip(
        clip_uuid,
        frame_count=3,
        chunk_frames=2,
        metadata=_metadata(frame_count=3),
    )

    expected_uploads = [
        f"{output_path}/chunks/v1/{clip_uuid}/frames-000000000-000000002.npz",
        f"{output_path}/chunks/v1/{clip_uuid}/frames-000000002-000000003.npz",
        f"{output_path}/metas/v1/{clip_uuid}.json",
    ]
    assert client.uploads == expected_uploads
    assert metadata_location == expected_uploads[-1]
    assert set(client.objects) == set(expected_uploads)
    assert json.loads(client.objects[expected_uploads[-1]])["frame_count"] == 3
    assert not list(tmp_path.iterdir())


def test_invalid_clip_uuid_is_rejected_before_creating_files(tmp_path: Path) -> None:
    """Clip identifiers cannot introduce unsafe path components."""
    writer = TemporalAnnotationWriter(tmp_path / "annotations")

    with (
        pytest.raises(ValueError, match="valid UUID"),
        writer.open_chunk("../not-a-uuid", 0, 1),
    ):
        pass
    assert not (tmp_path / "annotations").exists()
