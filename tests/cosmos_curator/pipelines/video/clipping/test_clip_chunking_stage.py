# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for source-backed clip rechunking."""

import pathlib
import uuid

import numpy as np
import pytest

from cosmos_curator.pipelines.video.annotation.data_model import AnnotationTask
from cosmos_curator.pipelines.video.clipping.clip_extraction_stages import ClipChunkingStage
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, VideoMetadata


def _make_task() -> SplitPipeTask:
    clips = [
        Clip(uuid=uuid.uuid4(), source_video="source.mkv", span=(float(index * 10), float((index + 1) * 10)))
        for index in range(3)
    ]
    video = Video(
        input_video=pathlib.Path("/shared/source.mkv"),
        encoded_data=np.arange(8, dtype=np.uint8),
        frame_array=np.zeros((2, 2, 2, 3), dtype=np.uint8),
        timestamps=np.array([0.0, 1.0], dtype=np.float32),
        metadata=VideoMetadata(size=8, duration=30.0),
        clips=clips,
    )
    return SplitPipeTask(session_id="source", videos=[video])


def test_clip_chunking_stage_keeps_source_spans_and_drops_large_buffers() -> None:
    """The source-backed fan-out should retain only lightweight path/span state."""
    task = _make_task()

    output = ClipChunkingStage(num_clips_per_chunk=1).process_data([task])

    assert len(output) == 3
    assert [chunk.video.clips[0].span for chunk in output] == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    for chunk_index, chunk in enumerate(output):
        assert chunk.video.input_path == "/shared/source.mkv"
        assert chunk.video.clip_chunk_index == chunk_index
        assert chunk.video.num_total_clips == 3
        assert chunk.video.num_clip_chunks == 3
        assert chunk.video.encoded_data.resolve() is None
        assert chunk.video.frame_array.resolve() is None
        assert chunk.video.timestamps is None
        assert chunk.video.clips[0].encoded_data.resolve() is None


def test_clip_chunking_stage_rejects_non_positive_chunk_size() -> None:
    """A non-positive rechunk target should fail at construction."""
    with pytest.raises(ValueError, match="must be positive"):
        ClipChunkingStage(num_clips_per_chunk=0)


def test_clip_chunking_stage_preserves_annotation_decode_hints() -> None:
    """Rechunking should preserve the concrete task type and source decode hints."""
    base_task = _make_task()
    task = AnnotationTask(
        session_id=base_task.session_id,
        videos=base_task.videos,
        stream_index=2,
        rotation_degrees_clockwise=90,
        dataset_metadata={"dataset": "example"},
    )

    output = ClipChunkingStage(num_clips_per_chunk=1).process_data([task])

    assert all(isinstance(chunk, AnnotationTask) for chunk in output)
    assert [chunk.stream_index for chunk in output] == [2, 2, 2]
    assert [chunk.rotation_degrees_clockwise for chunk in output] == [90, 90, 90]
    assert [chunk.dataset_metadata for chunk in output] == [{"dataset": "example"}] * 3
