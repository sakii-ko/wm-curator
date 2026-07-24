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
from uuid import UUID

import numpy as np
import pytest

from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.models.normalcrafter import (
    NORMALCRAFTER_VAE_CHUNK_SIZE,
    NormalCrafterModel,
    NormalCrafterRawChunk,
)
from cosmos_curator.pipelines.video.annotation import normalcrafter_stage
from cosmos_curator.pipelines.video.annotation.data_model import (
    full_source_clip_uuid,
    make_annotation_task,
)
from cosmos_curator.pipelines.video.annotation.decode_grid import (
    AnnotationGrid,
    DecodedAnnotationClip,
    annotation_grid_frame_count,
    make_raster_transform,
)
from cosmos_curator.pipelines.video.annotation.normalcrafter_stage import (
    NormalCrafterStage,
)

_TEST_GRID = AnnotationGrid(sample_fps=15.0, width=3, height=2)


class RecordingRuntime:
    """Fake resident runtime that records annotation-grid model input."""

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

    def __init__(self, decoded: DecodedAnnotationClip) -> None:
        """Store the fake decoded clip."""
        self.decoded = decoded
        self.calls: list[dict[str, object]] = []

    def __call__(  # noqa: PLR0913
        self,
        source: object,
        span: tuple[float, float] | None,
        *,
        stream_index: int,
        rotation_degrees_clockwise: int,
        grid: AnnotationGrid,
        min_frames: int,
        max_frames: int,
    ) -> DecodedAnnotationClip:
        """Record source identity, span, and safety bounds."""
        self.calls.append(
            {
                "source": source,
                "span": span,
                "stream_index": stream_index,
                "rotation_degrees_clockwise": rotation_degrees_clockwise,
                "grid": grid,
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


def _grid_timestamps(
    span: tuple[float, float],
    grid: AnnotationGrid = _TEST_GRID,
) -> np.ndarray:
    start_ns = round(span[0] * 1_000_000_000)
    stop_ns = round(span[1] * 1_000_000_000)
    _, _, timestamps_ns = make_ts_grid(
        start_ns,
        exclusive_end_ns=stop_ns,
        sample_rate_hz=grid.sample_fps,
    )
    return timestamps_ns.copy()


def _decoded_clip(
    frames: np.ndarray,
    *,
    span: tuple[float, float],
    rotation_degrees_clockwise: int,
    grid: AnnotationGrid = _TEST_GRID,
    decoder_backend: str = "fake-grid",
) -> DecodedAnnotationClip:
    timestamps_ns = _grid_timestamps(span, grid)
    source_timestamps_ns = timestamps_ns + 1_000_000
    return DecodedAnnotationClip(
        frames=frames,
        timestamps_ns=timestamps_ns,
        source_timestamps_ns=source_timestamps_ns,
        source_span=span,
        raster=make_raster_transform(
            2,
            3,
            rotation_degrees_clockwise=rotation_degrees_clockwise,
            grid=grid,
        ),
        decoder_backend=decoder_backend,
    )


def _write_completion(
    output: Path,
    *,
    clip_uuid: UUID,
    source: Path,
    span: tuple[float, float],
    grid: AnnotationGrid,
) -> Path:
    frame_count = annotation_grid_frame_count(span, grid)
    metadata_path = output / "metas" / "v1" / f"{clip_uuid}.json"
    metadata_path.parent.mkdir(parents=True)
    raster = make_raster_transform(
        3,
        2,
        rotation_degrees_clockwise=0,
        grid=grid,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "cosmos-curator.temporal-annotation/v1",
                "clip_uuid": str(clip_uuid),
                "format": "npz",
                "frame_count": frame_count,
                "chunk_frames": frame_count,
                "chunk_path_template": (f"chunks/v1/{clip_uuid}/frames-{{frame_start:09d}}-{{frame_stop:09d}}.npz"),
                "metadata": {
                    "producer": {
                        "id": "normalcrafter",
                        "release": "normalcrafter-grid-v1",
                        "model_id": "Yanrui95/NormalCrafter",
                    },
                    "source": {
                        "path": source.resolve().as_posix(),
                        "span_seconds": [span[0], span[1]],
                        "stream_index": 0,
                        "rotation_degrees_clockwise": 0,
                    },
                    "grid": grid.metadata(raster),
                    "arrays": {
                        "normal": {
                            "axes": "THWC",
                            "dtype": "float16",
                            "shape": [frame_count, grid.height, grid.width, 3],
                        },
                        "valid": {
                            "axes": "THW",
                            "dtype": "bool",
                            "shape": [frame_count, grid.height, grid.width],
                        },
                        "timestamps_ns": {
                            "axes": "T",
                            "dtype": "int64",
                            "shape": [frame_count],
                        },
                        "source_timestamps_ns": {
                            "axes": "T",
                            "dtype": "int64",
                            "shape": [frame_count],
                        },
                    },
                    "alignment": {
                        "sample_fps": grid.sample_fps,
                        "timestamp_array": "timestamps_ns",
                        "source_timestamp_array": "source_timestamps_ns",
                        "timestamp_unit": "nanosecond",
                        "timestamp_origin": "source_stream_start",
                    },
                    "recipe": {
                        "axis_transform_from_raw": [-1, 1, 1],
                        "normalization": "unit_length",
                        "normal_coordinate_space": "annotation_grid_camera_opencv",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return metadata_path


def test_whole_video_probe_rotation_chunk_alignment_and_metadata(tmp_path: Path) -> None:
    """A zero-clip filesystem task becomes one aligned, chunked annotation."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "annotations" / "normals" / "normalcrafter-grid-v1"
    frames = np.arange(15 * 2 * 3 * 3, dtype=np.uint8).reshape(15, 2, 3, 3)
    decoded = _decoded_clip(
        frames,
        span=(0.0, 1.0),
        rotation_degrees_clockwise=90,
        decoder_backend="fake-indexed",
    )
    timestamps_ns = decoded.timestamps_ns
    source_timestamps_ns = decoded.source_timestamps_ns
    decoder = RecordingDecoder(decoded)
    runtime = RecordingRuntime()
    model = _make_model(tmp_path, runtime)
    stage = NormalCrafterStage(
        output,
        model,
        annotation_grid=_TEST_GRID,
        decoder=decoder,
    )
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
            "rotation_degrees_clockwise": 90,
            "grid": _TEST_GRID,
            "min_frames": 14,
            "max_frames": 20,
        }
    ]
    np.testing.assert_array_equal(runtime.frames, frames)

    chunk_dir = output / "chunks" / "v1" / str(clip.uuid)
    chunks = sorted(chunk_dir.glob("*.npz"))
    assert [path.name for path in chunks] == [
        "frames-000000000-000000007.npz",
        "frames-000000007-000000014.npz",
        "frames-000000014-000000015.npz",
    ]
    with np.load(chunks[0]) as chunk:
        assert set(chunk.files) == {
            "normal",
            "valid",
            "timestamps_ns",
            "source_timestamps_ns",
        }
        assert chunk["normal"].shape == (7, 2, 3, 3)
        assert chunk["normal"].dtype == np.float16
        assert chunk["valid"].shape == (7, 2, 3)
        np.testing.assert_array_equal(chunk["timestamps_ns"], timestamps_ns[:7])
        np.testing.assert_array_equal(
            chunk["source_timestamps_ns"],
            source_timestamps_ns[:7],
        )
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
        "source_timestamp_array": "source_timestamps_ns",
        "timestamp_origin": "source_stream_start",
        "timestamp_unit": "nanosecond",
    }
    assert metadata["grid"] == _TEST_GRID.metadata(decoded.raster)
    assert metadata["recipe"]["window_size"] == 14
    assert metadata["recipe"]["window_stride"] == 10
    assert metadata["recipe"]["max_frames"] == 20
    assert metadata["recipe"]["normal_coordinate_space"] == "annotation_grid_camera_opencv"
    assert metadata["arrays"]["normal"]["shape"] == [15, 2, 3, 3]
    assert metadata["arrays"]["valid"]["shape"] == [15, 2, 3]
    assert metadata["arrays"]["timestamps_ns"]["shape"] == [15]
    assert metadata["arrays"]["source_timestamps_ns"]["shape"] == [15]
    assert task.dataset_metadata["normal_annotation"] == {
        "format": "npz-temporal-v1",
        "metadata_uri": str(metadata_path),
        "producer_release": "normalcrafter-grid-v1",
    }

    stage.destroy()
    assert runtime.closed


def test_existing_clip_span_is_passed_directly(tmp_path: Path) -> None:
    """A pre-split task decodes its source path and Clip.span without buffers."""
    source = tmp_path / "source.mp4"
    source.touch()
    span = (0.25, 1.25)
    decoded = _decoded_clip(
        np.zeros((15, 2, 3, 3), dtype=np.uint8),
        span=span,
        rotation_degrees_clockwise=0,
    )
    decoder = RecordingDecoder(decoded)
    runtime = RecordingRuntime()
    stage = NormalCrafterStage(
        tmp_path / "output",
        _make_model(tmp_path, runtime),
        annotation_grid=_TEST_GRID,
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


def test_existing_completion_marker_skips_decode_and_inference(tmp_path: Path) -> None:
    """A completed clip is reused without touching the decoder or model."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output"
    span = (0.25, 1.25)
    task = make_annotation_task(
        source,
        session_id="sample",
        relative_path="source.mp4",
        span=span,
    )
    clip = task.video.clips[0]
    grid = _TEST_GRID
    metadata_path = _write_completion(
        output,
        clip_uuid=clip.uuid,
        source=source,
        span=span,
        grid=grid,
    )

    decoded = _decoded_clip(
        np.zeros((15, 2, 3, 3), dtype=np.uint8),
        span=span,
        rotation_degrees_clockwise=0,
    )
    decoder = RecordingDecoder(decoded)
    runtime = RecordingRuntime()
    stage = NormalCrafterStage(
        output,
        _make_model(tmp_path, runtime),
        annotation_grid=grid,
        decoder=decoder,
    )

    stage.stage_setup()
    assert stage.process_data([task]) == [task]
    assert decoder.calls == []
    assert runtime.frames is None
    assert task.dataset_metadata["normal_annotation"] == {
        "format": "npz-temporal-v1",
        "metadata_uri": str(metadata_path),
        "producer_release": "normalcrafter-grid-v1",
    }


def test_full_source_completion_skips_decode_and_restores_clip(tmp_path: Path) -> None:
    """A full-source retry should resolve its stable UUID before video decode."""
    source = tmp_path / "whole-complete.mp4"
    source.touch()
    output = tmp_path / "output"
    span = (0.0, 1.0)
    clip_uuid = full_source_clip_uuid(source.resolve(), 0)
    metadata_path = _write_completion(
        output,
        clip_uuid=clip_uuid,
        source=source,
        span=span,
        grid=_TEST_GRID,
    )
    frames = np.zeros((15, 2, 3, 3), dtype=np.uint8)
    decoder = RecordingDecoder(
        _decoded_clip(
            frames,
            span=(0.0, 1.0),
            rotation_degrees_clockwise=0,
        )
    )
    runtime = RecordingRuntime()
    stage = NormalCrafterStage(
        output,
        _make_model(tmp_path, runtime),
        annotation_grid=_TEST_GRID,
        decoder=decoder,
    )
    task = make_annotation_task(
        source,
        session_id="whole-complete",
        relative_path="whole-complete.mp4",
    )

    stage.stage_setup()
    assert stage.process_data([task]) == [task]

    assert decoder.calls == []
    assert runtime.frames is None
    assert len(task.video.clips) == 1
    assert task.video.clips[0].uuid == clip_uuid
    assert task.video.clips[0].span == span
    assert task.dataset_metadata["normal_annotation"]["metadata_uri"] == str(metadata_path)


def test_completion_with_incomplete_output_contract_is_not_reused(tmp_path: Path) -> None:
    """Resume must not accept an artifact missing a required normal array."""
    source = tmp_path / "incomplete.mp4"
    source.touch()
    output = tmp_path / "output"
    span = (0.25, 1.25)
    task = make_annotation_task(
        source,
        session_id="incomplete",
        relative_path="incomplete.mp4",
        span=span,
    )
    metadata_path = _write_completion(
        output,
        clip_uuid=task.video.clips[0].uuid,
        source=source,
        span=span,
        grid=_TEST_GRID,
    )
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["metadata"]["arrays"]["source_timestamps_ns"]["dtype"] = "float32"
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    runtime = RecordingRuntime()
    decoder = RecordingDecoder(
        _decoded_clip(
            np.zeros((15, 2, 3, 3), dtype=np.uint8),
            span=span,
            rotation_degrees_clockwise=0,
        )
    )
    stage = NormalCrafterStage(
        output,
        _make_model(tmp_path, runtime),
        annotation_grid=_TEST_GRID,
        decoder=decoder,
    )

    stage.stage_setup()
    with pytest.raises(ValueError, match="incompatible output contract"):
        stage.process_data([task])

    assert decoder.calls == []
    assert runtime.frames is None


def test_existing_completion_with_different_grid_is_not_overwritten(tmp_path: Path) -> None:
    """A semantic mismatch should require a new root before any chunk is replaced."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output"
    span = (0.25, 1.25)
    task = make_annotation_task(
        source,
        session_id="sample",
        relative_path="source.mp4",
        span=span,
    )
    _write_completion(
        output,
        clip_uuid=task.video.clips[0].uuid,
        source=source,
        span=span,
        grid=AnnotationGrid(sample_fps=12.0, width=3, height=2),
    )
    runtime = RecordingRuntime()
    decoder = RecordingDecoder(
        _decoded_clip(
            np.zeros((15, 2, 3, 3), dtype=np.uint8),
            span=span,
            rotation_degrees_clockwise=0,
        )
    )
    stage = NormalCrafterStage(
        output,
        _make_model(tmp_path, runtime),
        annotation_grid=_TEST_GRID,
        decoder=decoder,
    )

    stage.stage_setup()
    with pytest.raises(ValueError, match="different source, producer, or annotation grid"):
        stage.process_data([task])
    assert decoder.calls == []
    assert runtime.frames is None


def test_stage_rejects_decoder_output_past_model_limit(tmp_path: Path) -> None:
    """A decoder cannot bypass the explicit full-clip safety limit."""
    source = tmp_path / "source.mp4"
    source.touch()
    decoded = _decoded_clip(
        np.zeros((15, 2, 3, 3), dtype=np.uint8),
        span=(0.0, 1.0),
        rotation_degrees_clockwise=0,
    )
    runtime = RecordingRuntime()
    stage = NormalCrafterStage(
        tmp_path / "output",
        _make_model(tmp_path, runtime, max_frames=14),
        annotation_grid=_TEST_GRID,
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


def test_stage_rejects_decoder_rotation_mismatch_before_inference(tmp_path: Path) -> None:
    """An injected decoder must honor the requested display rotation."""
    source = tmp_path / "wrong-rotation.mp4"
    source.touch()
    span = (0.0, 1.0)
    decoder = RecordingDecoder(
        _decoded_clip(
            np.zeros((15, 2, 3, 3), dtype=np.uint8),
            span=span,
            rotation_degrees_clockwise=90,
        )
    )
    runtime = RecordingRuntime()
    stage = NormalCrafterStage(
        tmp_path / "output",
        _make_model(tmp_path, runtime),
        annotation_grid=_TEST_GRID,
        decoder=decoder,
    )
    task = make_annotation_task(
        source,
        session_id="wrong-rotation",
        relative_path="wrong-rotation.mp4",
        span=span,
    )

    with pytest.raises(ValueError, match="returned rotation_degrees_clockwise"):
        stage.process_data([task])

    assert runtime.frames is None


def test_decoder_forwards_to_shared_annotation_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The producer-specific decoder remains a thin shared-grid adapter."""
    grid = AnnotationGrid(sample_fps=12.0, width=4, height=3)
    expected = _decoded_clip(
        np.zeros((15, 3, 4, 3), dtype=np.uint8),
        span=(0.0, 1.25),
        rotation_degrees_clockwise=270,
        grid=grid,
    )
    calls: list[dict[str, object]] = []

    def fake_shared_decoder(
        source: Path,
        span: tuple[float, float] | None,
        **kwargs: object,
    ) -> DecodedAnnotationClip:
        calls.append({"source": source, "span": span, **kwargs})
        return expected

    monkeypatch.setattr(normalcrafter_stage, "decode_annotation_clip", fake_shared_decoder)

    assert (
        normalcrafter_stage.decode_normalcrafter_clip(
            Path("mixed.ts"),
            (0.0, 1.25),
            stream_index=2,
            rotation_degrees_clockwise=270,
            grid=grid,
            min_frames=14,
            max_frames=20,
        )
        is expected
    )
    assert calls == [
        {
            "source": Path("mixed.ts"),
            "span": (0.0, 1.25),
            "stream_index": 2,
            "rotation_degrees_clockwise": 270,
            "grid": grid,
            "min_frames": 14,
            "max_frames": 20,
        }
    ]
