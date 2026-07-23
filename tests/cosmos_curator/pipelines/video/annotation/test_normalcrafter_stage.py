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

"""CPU tests for NormalCrafter source alignment and artifact output."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from cosmos_curator.models.normalcrafter import (
    NORMALCRAFTER_VAE_CHUNK_SIZE,
    NormalCrafterModel,
    NormalCrafterRawChunk,
)
from cosmos_curator.pipelines.video.annotation import normalcrafter_stage
from cosmos_curator.pipelines.video.annotation.data_model import make_annotation_task
from cosmos_curator.pipelines.video.annotation.normalcrafter_stage import (
    DecodedNormalCrafterClip,
    NormalCrafterStage,
)


class RecordingRuntime:
    """Fake resident runtime that records rotated model input."""

    def __init__(self) -> None:
        """Initialize without inference calls."""
        self.frames: np.ndarray | None = None
        self.closed = False

    def infer(
        self,
        frames: np.ndarray,
    ) -> Iterator[NormalCrafterRawChunk]:
        """Emit the fixed seven-frame chunk contract."""
        self.frames = frames.copy()
        for start in range(0, len(frames), NORMALCRAFTER_VAE_CHUNK_SIZE):
            stop = min(start + NORMALCRAFTER_VAE_CHUNK_SIZE, len(frames))
            values = np.empty((*frames[start:stop].shape[:3], 3), dtype=np.float32)
            values[...] = (1.0, 2.0, 2.0)
            yield NormalCrafterRawChunk(frame_start=start, values=values)

    def close(self) -> None:
        """Record teardown."""
        self.closed = True


class RecordingDecoder:
    """Return one probed whole-video result and record the source request."""

    def __init__(self, decoded: DecodedNormalCrafterClip) -> None:
        """Store the fake decoded clip."""
        self.decoded = decoded
        self.calls: list[dict[str, object]] = []

    def __call__(  # noqa: PLR0913
        self,
        source: object,
        span: tuple[float, float] | None,
        *,
        stream_index: int,
        sample_fps: float,
        min_frames: int,
        max_frames: int,
    ) -> DecodedNormalCrafterClip:
        """Record source identity, span, and safety bounds."""
        self.calls.append(
            {
                "source": source,
                "span": span,
                "stream_index": stream_index,
                "sample_fps": sample_fps,
                "min_frames": min_frames,
                "max_frames": max_frames,
            }
        )
        return self.decoded


def _make_model(
    tmp_path: Path,
    runtime: RecordingRuntime,
    *,
    max_frames: int = 20,
) -> NormalCrafterModel:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    return NormalCrafterModel(
        checkpoint_path=checkpoint,
        max_frames=max_frames,
        runtime_factory=lambda _path: runtime,
    )


def test_whole_video_probe_rotation_chunk_alignment_and_metadata(tmp_path: Path) -> None:
    """A zero-clip filesystem task becomes one aligned, chunked annotation."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "annotations" / "normals" / "normalcrafter-v1"
    frames = np.arange(15 * 2 * 3 * 3, dtype=np.uint8).reshape(15, 2, 3, 3)
    timestamps_ns = np.arange(15, dtype=np.int64) * 66_666_667
    decoded = DecodedNormalCrafterClip(
        frames=frames,
        timestamps_ns=timestamps_ns,
        source_span=(0.0, 1.0),
        decoder_backend="fake-indexed",
    )
    decoder = RecordingDecoder(decoded)
    runtime = RecordingRuntime()
    model = _make_model(tmp_path, runtime)
    stage = NormalCrafterStage(output, model, decoder=decoder)
    task = make_annotation_task(
        source,
        session_id="sample",
        relative_path="source.mp4",
        rotation_degrees_clockwise=90,
    )

    stage.stage_setup()
    assert stage.process_data([task]) == [task]

    assert len(task.video.clips) == 1
    clip = task.video.clips[0]
    assert clip.span == (0.0, 1.0)
    assert task.video.num_total_clips == 1
    assert decoder.calls == [
        {
            "source": source.resolve(),
            "span": None,
            "stream_index": 0,
            "sample_fps": 15.0,
            "min_frames": 14,
            "max_frames": 20,
        }
    ]
    assert runtime.frames is not None
    np.testing.assert_array_equal(
        runtime.frames,
        np.rot90(frames, k=-1, axes=(1, 2)),
    )

    chunk_dir = output / "chunks" / "v1" / str(clip.uuid)
    chunks = sorted(chunk_dir.glob("*.npz"))
    assert [path.name for path in chunks] == [
        "frames-000000000-000000007.npz",
        "frames-000000007-000000014.npz",
        "frames-000000014-000000015.npz",
    ]
    with np.load(chunks[0]) as chunk:
        assert set(chunk.files) == {"normal", "valid", "timestamps_ns"}
        assert chunk["normal"].shape == (7, 3, 2, 3)
        assert chunk["normal"].dtype == np.float16
        assert chunk["valid"].shape == (7, 3, 2)
        np.testing.assert_array_equal(chunk["timestamps_ns"], timestamps_ns[:7])
        np.testing.assert_allclose(
            chunk["normal"][0, 0, 0],
            np.asarray((-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0), dtype=np.float16),
            atol=1.0e-3,
        )

    metadata_path = output / "metas" / "v1" / f"{clip.uuid}.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert document["frame_count"] == 15
    assert document["chunk_frames"] == 7
    metadata = document["metadata"]
    assert metadata["alignment"] == {
        "sample_fps": 15.0,
        "timestamp_array": "timestamps_ns",
        "timestamp_origin": "source_stream_start",
        "timestamp_unit": "nanosecond",
    }
    assert metadata["recipe"]["window_size"] == 14
    assert metadata["recipe"]["window_stride"] == 10
    assert metadata["recipe"]["max_frames"] == 20
    assert metadata["arrays"]["normal"]["shape"] == [15, 3, 2, 3]
    assert metadata["arrays"]["valid"]["shape"] == [15, 3, 2]
    assert metadata["arrays"]["timestamps_ns"]["shape"] == [15]
    assert task.dataset_metadata["normal_annotation"] == {
        "format": "npz-temporal-v1",
        "metadata_uri": str(metadata_path),
        "producer_release": "normalcrafter-v1",
    }

    stage.destroy()
    assert runtime.closed


def test_existing_clip_span_is_passed_directly(tmp_path: Path) -> None:
    """A pre-split task decodes its source path and Clip.span without buffers."""
    source = tmp_path / "source.mp4"
    source.touch()
    span = (0.25, 1.25)
    decoded = DecodedNormalCrafterClip(
        frames=np.zeros((15, 1, 1, 3), dtype=np.uint8),
        timestamps_ns=np.arange(15, dtype=np.int64),
        source_span=span,
        decoder_backend="fake",
    )
    decoder = RecordingDecoder(decoded)
    runtime = RecordingRuntime()
    stage = NormalCrafterStage(
        tmp_path / "output",
        _make_model(tmp_path, runtime),
        decoder=decoder,
    )
    task = make_annotation_task(
        source,
        session_id="sample",
        relative_path="source.mp4",
        span=span,
    )

    stage.stage_setup()
    stage.process_data([task])
    assert decoder.calls[0]["source"] == source.resolve()
    assert decoder.calls[0]["span"] == span


def test_stage_rejects_decoder_output_past_model_limit(tmp_path: Path) -> None:
    """A decoder cannot bypass the explicit full-clip safety limit."""
    source = tmp_path / "source.mp4"
    source.touch()
    decoded = DecodedNormalCrafterClip(
        frames=np.zeros((15, 1, 1, 3), dtype=np.uint8),
        timestamps_ns=np.arange(15, dtype=np.int64),
        source_span=(0.0, 1.0),
        decoder_backend="fake",
    )
    runtime = RecordingRuntime()
    stage = NormalCrafterStage(
        tmp_path / "output",
        _make_model(tmp_path, runtime, max_frames=14),
        decoder=RecordingDecoder(decoded),
    )
    task = make_annotation_task(
        source,
        session_id="sample",
        relative_path="source.mp4",
    )

    stage.stage_setup()
    with pytest.raises(ValueError, match="max_frames=14"):
        stage.process_data([task])
    assert runtime.frames is None


def test_index_failure_uses_pyav_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Weakly indexed and mixed containers retain a direct PyAV path."""
    expected = DecodedNormalCrafterClip(
        frames=np.zeros((14, 1, 1, 3), dtype=np.uint8),
        timestamps_ns=np.arange(14, dtype=np.int64),
        source_span=(0.0, 1.0),
        decoder_backend="pyav",
    )

    def fail_indexed(*_args: object, **_kwargs: object) -> None:
        message = "no header index"
        raise normalcrafter_stage._IndexedDecodeUnavailableError(message)

    def fake_pyav(*_args: object, **_kwargs: object) -> DecodedNormalCrafterClip:
        return expected

    monkeypatch.setattr(normalcrafter_stage, "_decode_with_camera_sensor", fail_indexed)
    monkeypatch.setattr(normalcrafter_stage, "_decode_with_pyav", fake_pyav)

    assert (
        normalcrafter_stage.decode_normalcrafter_clip(
            Path("mixed.ts"),
            None,
            stream_index=0,
            sample_fps=15.0,
            min_frames=14,
            max_frames=20,
        )
        is expected
    )
