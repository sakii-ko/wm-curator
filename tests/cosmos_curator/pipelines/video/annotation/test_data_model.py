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
"""Tests for annotation input task construction."""

import uuid
from pathlib import Path

import pytest

from cosmos_curator.pipelines.video.annotation.data_model import make_annotation_task
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video


def test_make_annotation_task_reuses_split_data_model(tmp_path: Path) -> None:
    """The thin task should use Video and Clip rather than parallel record types."""
    source = (tmp_path / "video.mp4").resolve()
    metadata = {"dataset": "demo"}

    first = make_annotation_task(
        source,
        session_id="sample",
        relative_path="nested/video.mp4",
        stream_index=0,
        rotation_degrees_clockwise=450,
        span=(0.5, 2.0),
        dataset_metadata=metadata,
    )

    assert isinstance(first, SplitPipeTask)
    assert isinstance(first.video, Video)
    assert isinstance(first.video.clips[0], Clip)
    assert first.input_span == (0.5, 2.0)
    assert first.rotation_degrees_clockwise == 90
    assert first.dataset_metadata == metadata
    assert first.dataset_metadata is not metadata


def test_make_annotation_task_uses_stable_source_span_uuid(tmp_path: Path) -> None:
    """The same source stream and normalized span should keep one artifact identity."""
    source = (tmp_path / "video.mp4").resolve()
    first = make_annotation_task(
        source,
        session_id="first-label",
        relative_path="first/video.mp4",
        stream_index=None,
        span=(0.5, 2.0),
    )
    second = make_annotation_task(
        source,
        session_id="second-label",
        relative_path="renamed/video.mp4",
        stream_index=0,
        span=(0.5, 2.0),
    )
    other_stream = make_annotation_task(
        source,
        session_id="first-label",
        relative_path="first/video.mp4",
        stream_index=1,
        span=(0.5, 2.0),
    )

    assert first.video.clips[0].uuid == second.video.clips[0].uuid
    assert first.video.clips[0].uuid != other_stream.video.clips[0].uuid


def test_make_annotation_task_preserves_explicit_clip_uuid(tmp_path: Path) -> None:
    """A source-span reader should be able to retain an existing Cosmos span UUID."""
    clip_uuid = uuid.uuid4()
    task = make_annotation_task(
        tmp_path / "video.mp4",
        session_id="sample",
        relative_path="video.mp4",
        span=(0.0, 1.0),
        clip_uuid=str(clip_uuid),
    )

    assert task.video.clips[0].uuid == clip_uuid


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"stream_index": True}, "non-negative integer"),
        ({"stream_index": -1}, "non-negative integer"),
        ({"rotation_degrees_clockwise": 1}, "multiple of 90"),
        ({"span": (1.0, 1.0)}, "0 <= start < end"),
        ({"span": (float("nan"), 1.0)}, "finite"),
        ({"relative_path": "../video.mp4"}, "must not be absolute"),
        ({"span": (0.0, 1.0), "clip_uuid": "not-a-uuid"}, "valid UUID"),
        ({"clip_uuid": str(uuid.uuid4())}, "requires an explicit span"),
    ],
)
def test_make_annotation_task_validates_source_hints(
    tmp_path: Path,
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Invalid decode hints should fail before entering a pipeline."""
    task_kwargs: dict[str, object] = {"relative_path": "video.mp4"}
    task_kwargs.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        make_annotation_task(
            tmp_path / "video.mp4",
            session_id="sample",
            **task_kwargs,  # type: ignore[arg-type]
        )
