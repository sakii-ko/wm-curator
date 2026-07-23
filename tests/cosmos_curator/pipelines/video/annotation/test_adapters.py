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
"""Tests for thin annotation dataset adapters."""

import json
import uuid
from pathlib import Path

import pytest

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.pipelines.video.annotation import adapters
from cosmos_curator.pipelines.video.annotation.adapters import (
    AnnotationDatasetAdapter,
    FilesystemDatasetAdapter,
    JsonlDatasetAdapter,
    SourceSpanDatasetAdapter,
)
from cosmos_curator.pipelines.video.annotation.data_model import AnnotationTask
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from tests.cosmos_curator.core.utils.storage.conftest import FakeStorageClient


def test_filesystem_adapter_recurses_filters_and_sorts(tmp_path: Path) -> None:
    """Filesystem discovery should emit stable, existing task types."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.MKV").touch()
    (tmp_path / "a.mp4").touch()
    (tmp_path / "ignore.txt").touch()

    adapter = FilesystemDatasetAdapter(
        tmp_path,
        stream_index=2,
        rotation_degrees_clockwise=-90,
        dataset_metadata={"dataset": "demo"},
    )
    tasks = adapter.discover()

    assert isinstance(adapter, AnnotationDatasetAdapter)
    assert [task.session_id for task in tasks] == ["a.mp4", "nested/b.MKV"]
    assert all(isinstance(task, AnnotationTask) for task in tasks)
    assert all(isinstance(task, SplitPipeTask) for task in tasks)
    assert all(isinstance(task.video, Video) for task in tasks)
    assert [task.video.relative_path for task in tasks] == ["a.mp4", "nested/b.MKV"]
    assert [task.video.input_video for task in tasks] == [
        (tmp_path / "a.mp4").resolve(),
        (tmp_path / "nested" / "b.MKV").resolve(),
    ]
    assert all(task.stream_index == 2 for task in tasks)
    assert all(task.rotation_degrees_clockwise == 270 for task in tasks)
    assert all(task.dataset_metadata == {"dataset": "demo"} for task in tasks)
    assert all(not task.video.clips for task in tasks)


def test_filesystem_adapter_normalizes_custom_extensions(tmp_path: Path) -> None:
    """Extension configuration should be case-insensitive and accept a missing dot."""
    (tmp_path / "video.custom").touch()
    (tmp_path / "video.mp4").touch()

    tasks = FilesystemDatasetAdapter(tmp_path, extensions={"CUSTOM"}).discover()

    assert [task.session_id for task in tasks] == ["video.custom"]


def test_jsonl_adapter_supports_source_hints_and_metadata(tmp_path: Path) -> None:
    """JSONL rows should map directly onto Video, Clip, and annotation hints."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    first = video_dir / "first.mp4"
    second = video_dir / "second.mov"
    first.touch()
    second.touch()
    input_list = tmp_path / "inputs.jsonl"
    rows = [
        {
            "path": "first.mp4",
            "id": "sample-1",
            "stream_index": 1,
            "rotation_degrees_clockwise": -90,
            "span": [1.25, 4.5],
            "metadata": {"scene": "indoor"},
        },
        {
            "path": second.as_posix(),
            "relative_path": "custom/second.mov",
            "metadata": {"scene": "outdoor"},
        },
    ]
    input_list.write_text("\n" + "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    tasks = JsonlDatasetAdapter(
        input_list,
        source_root=video_dir,
        dataset_metadata={"dataset": "demo", "scene": "default"},
    ).discover()

    assert [task.session_id for task in tasks] == ["sample-1", "custom/second.mov"]
    assert tasks[0].video.input_video == first.resolve()
    assert tasks[0].video.relative_path == "first.mp4"
    assert tasks[0].stream_index == 1
    assert tasks[0].rotation_degrees_clockwise == 270
    assert tasks[0].input_span == (1.25, 4.5)
    assert len(tasks[0].video.clips) == 1
    assert isinstance(tasks[0].video.clips[0], Clip)
    assert tasks[0].video.clips[0].source_video == first.resolve().as_posix()
    assert tasks[0].video.num_total_clips == 1
    assert tasks[0].dataset_metadata == {
        "dataset": "demo",
        "scene": "indoor",
    }

    assert tasks[1].video.input_video == second.resolve()
    assert tasks[1].video.relative_path == "custom/second.mov"
    assert tasks[1].input_span is None
    assert tasks[1].dataset_metadata["scene"] == "outdoor"


def test_jsonl_adapter_accepts_explicit_remote_sources(tmp_path: Path) -> None:
    """An explicit supported cloud URI should retain the existing StoragePrefix model."""
    input_list = tmp_path / "remote.jsonl"
    input_list.write_text(
        json.dumps({"path": "s3://example-bucket/videos/a.mp4", "id": "remote-a"}),
        encoding="utf-8",
    )

    task = JsonlDatasetAdapter(input_list).discover()[0]

    assert isinstance(task.video.input_video, S3Prefix)
    assert task.video.input_path == "s3://example-bucket/videos/a.mp4"
    assert task.video.relative_path == "a.mp4"


@pytest.mark.parametrize("uuid_field", ["clip_uuid", "span_uuid"])
def test_jsonl_adapter_preserves_explicit_span_uuid(tmp_path: Path, uuid_field: str) -> None:
    """Both accepted field names should retain an upstream clip identity."""
    source = tmp_path / "video.mp4"
    source.touch()
    clip_uuid = uuid.uuid4()
    input_list = tmp_path / "explicit-uuid.jsonl"
    input_list.write_text(
        json.dumps({"path": source.name, "span": [1.0, 2.0], uuid_field: str(clip_uuid)}),
        encoding="utf-8",
    )

    task = JsonlDatasetAdapter(input_list).discover()[0]

    assert task.video.clips[0].uuid == clip_uuid


def test_jsonl_adapter_rejects_conflicting_clip_uuid_aliases(tmp_path: Path) -> None:
    """clip_uuid and span_uuid cannot name two different source spans."""
    source = tmp_path / "video.mp4"
    source.touch()
    input_list = tmp_path / "conflicting-uuid.jsonl"
    input_list.write_text(
        json.dumps(
            {
                "path": source.name,
                "span": [1.0, 2.0],
                "clip_uuid": str(uuid.uuid4()),
                "span_uuid": str(uuid.uuid4()),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"conflicting-uuid\.jsonl:1:.*must match"):
        JsonlDatasetAdapter(input_list).discover()


def test_source_span_adapter_reads_cosmos_summary_and_preserves_uuid(tmp_path: Path) -> None:
    """No-transcode split metadata should feed annotation tasks without a manual JSONL export."""
    source = tmp_path / "raw" / "nested" / "video.mkv"
    source.parent.mkdir(parents=True)
    source.touch()
    output_root = tmp_path / "split-output"
    metadata_root = output_root / "metas" / "v0"
    metadata_root.mkdir(parents=True)
    valid_uuid = uuid.uuid4()
    invalid_uuid = uuid.uuid4()
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "num_videos": 1,
                "nested/video.mkv": {
                    "source_video": source.as_posix(),
                    "clips": [str(valid_uuid), str(invalid_uuid)],
                },
            }
        ),
        encoding="utf-8",
    )
    for clip_uuid, span, valid in (
        (valid_uuid, [1.25, 4.5], True),
        (invalid_uuid, [5.0, 6.0], False),
    ):
        (metadata_root / f"{clip_uuid}.json").write_text(
            json.dumps(
                {
                    "span_uuid": str(clip_uuid),
                    "source_video": source.as_posix(),
                    "duration_span": span,
                    "clip_format": "source_span",
                    "valid": valid,
                }
            ),
            encoding="utf-8",
        )

    tasks = SourceSpanDatasetAdapter(
        output_root,
        stream_index=2,
        rotation_degrees_clockwise=-90,
        dataset_metadata={"dataset": "demo"},
    ).discover()

    assert len(tasks) == 1
    task = tasks[0]
    assert task.video.input_video == source.resolve()
    assert task.video.relative_path == "nested/video.mkv"
    assert task.video.clips[0].uuid == valid_uuid
    assert task.video.clips[0].span == (1.25, 4.5)
    assert task.stream_index == 2
    assert task.rotation_degrees_clockwise == 270
    assert task.dataset_metadata == {"dataset": "demo"}


def test_source_span_adapter_reads_remote_output_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter should use Cosmos storage clients without materializing remote media."""
    output_root = "s3://test-bucket/split-output"
    source = "s3://media-bucket/raw/video.mkv"
    clip_uuid = uuid.uuid4()
    client = FakeStorageClient(
        {
            f"{output_root}/summary.json": json.dumps(
                {"video.mkv": {"source_video": source, "clips": [str(clip_uuid)]}}
            ).encode(),
            f"{output_root}/metas/v0/{clip_uuid}.json": json.dumps(
                {
                    "span_uuid": str(clip_uuid),
                    "source_video": source,
                    "duration_span": [0.0, 3.0],
                    "clip_format": "source_span",
                    "valid": True,
                }
            ).encode(),
        }
    )
    monkeypatch.setattr(adapters, "get_storage_client", lambda *_args, **_kwargs: client)

    task = SourceSpanDatasetAdapter(output_root).discover()[0]

    assert isinstance(task.video.input_video, S3Prefix)
    assert task.video.input_path == source
    assert task.video.clips[0].uuid == clip_uuid


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ({"id": "missing-path"}, "field 'path'"),
        ({"path": "video.mp4", "unexpected": True}, "unknown JSONL fields"),
        ({"path": "video.mp4", "span": [3, 2]}, "0 <= start < end"),
        ({"path": "video.mp4", "stream_index": -1}, "non-negative integer"),
        ({"path": "video.mp4", "rotation_degrees_clockwise": 45}, "multiple of 90"),
        ({"path": "video.mp4", "metadata": []}, "metadata must be an object"),
    ],
)
def test_jsonl_adapter_reports_invalid_rows_with_line_number(
    tmp_path: Path,
    row: dict[str, object],
    match: str,
) -> None:
    """JSONL validation failures should identify the offending line."""
    (tmp_path / "video.mp4").touch()
    input_list = tmp_path / "invalid.jsonl"
    input_list.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"invalid\.jsonl:1:.*{match}"):
        JsonlDatasetAdapter(input_list).discover()


def test_jsonl_adapter_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Task IDs must remain unique within one JSONL input list."""
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mp4").touch()
    input_list = tmp_path / "duplicate.jsonl"
    input_list.write_text(
        "\n".join(
            (
                json.dumps({"path": "a.mp4", "id": "same"}),
                json.dumps({"path": "b.mp4", "id": "same"}),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate\.jsonl:2:.*duplicate JSONL id"):
        JsonlDatasetAdapter(input_list).discover()


def test_jsonl_adapter_rejects_non_object_lines(tmp_path: Path) -> None:
    """A JSON value is not enough; each row must expose named fields."""
    input_list = tmp_path / "invalid.jsonl"
    input_list.write_text(json.dumps(["video.mp4"]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid\.jsonl:1:.*must contain an object"):
        JsonlDatasetAdapter(input_list).discover()
