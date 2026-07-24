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
"""CPU tests for ViPE decode, alignment, validation, and chunk publication."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from uuid import UUID, uuid4

import av
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.models.vipe import ViPEFrameResult
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
from cosmos_curator.pipelines.video.annotation.vipe_stage import ViPEStage, decode_local_vipe_clip
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video

_TEST_GRID = AnnotationGrid(sample_fps=10.0, width=24, height=16)


class _FakeViPEModel(ModelInterface):
    def __init__(
        self,
        *,
        bad_first_index: bool = False,
        depth_overrides: dict[int, npt.NDArray[np.float32]] | None = None,
    ) -> None:
        self.bad_first_index = bad_first_index
        self.depth_overrides = depth_overrides or {}
        self.setup_called = False
        self.closed = False
        self.infer_called = False
        self.produced = 0
        self.received_shape: tuple[int, ...] | None = None
        self.received_fps: float | None = None

    @property
    def conda_env_name(self) -> str:
        return "test"

    @property
    def model_id_names(self) -> list[str]:
        return []

    def setup(self) -> None:
        self.setup_called = True

    def infer(
        self,
        frames: npt.NDArray[np.uint8],
        *,
        name: str,
        fps: float,
    ) -> Iterator[ViPEFrameResult]:
        del name
        self.infer_called = True
        self.received_shape = frames.shape
        self.received_fps = fps
        height, width = frames.shape[1:3]
        for index in range(len(frames)):
            self.produced = index + 1
            depth = np.full((height, width), index + 1, dtype=np.float32)
            if index in self.depth_overrides:
                depth = self.depth_overrides[index].copy()
            elif index == 0:
                depth[0, 0] = np.nan
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = index
            yield ViPEFrameResult(
                raw_frame_idx=index + int(self.bad_first_index and index == 0),
                metric_depth=depth,
                intrinsics=np.asarray((100.0 + index, 101.0 + index, 12.0, 8.0), dtype=np.float32),
                camera_to_world=pose,
            )

    def close(self) -> None:
        self.closed = True


class _FakeDecoder:
    def __init__(
        self,
        *,
        frame_count: int = 10,
        source_height: int = 16,
        source_width: int = 24,
        source_span: tuple[float, float] | None = None,
    ) -> None:
        self.frame_count = frame_count
        self.source_height = source_height
        self.source_width = source_width
        self.source_span = source_span or (1.0, 1.0 + frame_count / _TEST_GRID.sample_fps)
        self.frames: npt.NDArray[np.uint8] | None = None
        self.timestamps: npt.NDArray[np.int64] | None = None
        self.source_timestamps: npt.NDArray[np.int64] | None = None
        self.calls: list[dict[str, object]] = []

    def __call__(  # noqa: PLR0913
        self,
        source_path: Path,
        span: tuple[float, float] | None,
        *,
        stream_index: int,
        rotation_degrees_clockwise: int,
        grid: AnnotationGrid,
        min_frames: int,
        max_frames: int,
    ) -> DecodedAnnotationClip:
        self.calls.append(
            {
                "source_path": source_path,
                "span": span,
                "stream_index": stream_index,
                "rotation": rotation_degrees_clockwise,
                "grid": grid,
                "min_frames": min_frames,
                "max_frames": max_frames,
            }
        )
        start_ns = round(self.source_span[0] * 1_000_000_000)
        stop_ns = round(self.source_span[1] * 1_000_000_000)
        _, _, timestamps = make_ts_grid(
            start_ns,
            exclusive_end_ns=stop_ns,
            sample_rate_hz=grid.sample_fps,
        )
        assert len(timestamps) == self.frame_count
        self.frames = np.zeros((self.frame_count, grid.height, grid.width, 3), dtype=np.uint8)
        self.timestamps = timestamps
        self.source_timestamps = timestamps + round(0.01 * 1_000_000_000)
        return DecodedAnnotationClip(
            frames=self.frames,
            timestamps_ns=self.timestamps,
            source_timestamps_ns=self.source_timestamps,
            source_span=self.source_span,
            raster=make_raster_transform(
                self.source_width,
                self.source_height,
                rotation_degrees_clockwise=rotation_degrees_clockwise,
                grid=grid,
            ),
            decoder_backend="fake_grid",
        )


class _FakeWriter:
    def __init__(self, root: Path, model: _FakeViPEModel) -> None:
        self.root = root
        self.model = model
        self.chunks: list[tuple[int, int, int, dict[str, npt.NDArray[np.generic]]]] = []
        self.completion: dict[str, object] | None = None

    @contextmanager
    def open_chunk(
        self,
        clip_uuid: str | UUID,
        frame_start: int,
        frame_stop: int,
    ) -> Iterator[Path]:
        path = self.root / f"{clip_uuid}-{frame_start}-{frame_stop}.npz"
        yield path
        with np.load(path) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        self.chunks.append((frame_start, frame_stop, self.model.produced, arrays))

    def complete_clip(
        self,
        clip_uuid: str | UUID,
        *,
        frame_count: int,
        chunk_frames: int,
        metadata: dict[str, object],
    ) -> str:
        self.completion = {
            "clip_uuid": str(clip_uuid),
            "frame_count": frame_count,
            "chunk_frames": chunk_frames,
            "metadata": metadata,
        }
        return "memory://complete"


def _make_stage(
    tmp_path: Path,
    *,
    decoder: _FakeDecoder,
    model: _FakeViPEModel | None = None,
    chunk_frames: int = 4,
    max_frames: int = 32,
) -> tuple[ViPEStage, _FakeViPEModel, _FakeWriter]:
    actual_model = model or _FakeViPEModel()
    writer = _FakeWriter(tmp_path, actual_model)
    stage = ViPEStage(
        tmp_path / "output",
        actual_model,
        chunk_frames=chunk_frames,
        max_frames=max_frames,
        annotation_grid=_TEST_GRID,
        decoder=decoder,
        writer=writer,
    )
    return stage, actual_model, writer


def _write_completion(
    output: Path,
    *,
    clip_uuid: UUID,
    source: Path,
    span: tuple[float, float],
    grid: AnnotationGrid = _TEST_GRID,
) -> Path:
    frame_count = annotation_grid_frame_count(span, grid)
    metadata_path = output / "metas" / "v1" / f"{clip_uuid}.json"
    metadata_path.parent.mkdir(parents=True)
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
                        "id": "vipe",
                        "pipeline": "dav3",
                        "release": "vipe-dav3-grid-v1",
                    },
                    "source": {
                        "path": str(source.resolve()),
                        "span_seconds": [span[0], span[1]],
                        "stream_index": 0,
                        "rotation_degrees_clockwise": 0,
                        "decoder_backend": "fake_grid",
                    },
                    "grid": grid.metadata(
                        make_raster_transform(
                            24,
                            16,
                            rotation_degrees_clockwise=0,
                            grid=grid,
                        )
                    ),
                    "arrays": {
                        "depth": {
                            "axes": "THW",
                            "dtype": "float16",
                            "shape": [frame_count, grid.height, grid.width],
                        },
                        "valid": {
                            "axes": "THW",
                            "dtype": "bool",
                            "shape": [frame_count, grid.height, grid.width],
                        },
                        "K": {
                            "dtype": "float32",
                            "shape": [frame_count, 3, 3],
                        },
                        "camera_to_world": {
                            "dtype": "float32",
                            "shape": [frame_count, 4, 4],
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
                        "depth_representation": "z_depth",
                        "depth_scale": "metric",
                        "K_coordinate_space": "annotation_grid",
                        "min_valid_fraction_per_frame": 0.5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return metadata_path


def test_stage_writes_aligned_bounded_chunks(tmp_path: Path) -> None:
    """Results should stay frame-aligned and flush before inference completes."""
    source = tmp_path / "clip.mkv"
    source.touch()
    task = make_annotation_task(
        source,
        session_id="clip",
        relative_path="clip.mkv",
        stream_index=2,
        rotation_degrees_clockwise=90,
        span=(1.0, 2.0),
    )
    decoder = _FakeDecoder()
    stage, model, writer = _make_stage(tmp_path, decoder=decoder)

    returned = stage.process_data([task])

    assert returned == [task]
    assert decoder.calls == [
        {
            "source_path": source.resolve(),
            "span": (1.0, 2.0),
            "stream_index": 2,
            "rotation": 90,
            "grid": _TEST_GRID,
            "min_frames": 8,
            "max_frames": 32,
        }
    ]
    assert model.received_shape == (10, 16, 24, 3)
    assert model.received_fps == _TEST_GRID.sample_fps
    assert [(start, stop, produced) for start, stop, produced, _ in writer.chunks] == [
        (0, 4, 4),
        (4, 8, 8),
        (8, 10, 10),
    ]

    for _, _, _, arrays in writer.chunks:
        assert set(arrays) == {
            "depth",
            "valid",
            "K",
            "camera_to_world",
            "timestamps_ns",
            "source_timestamps_ns",
        }
    depth = np.concatenate([chunk[3]["depth"] for chunk in writer.chunks])
    valid = np.concatenate([chunk[3]["valid"] for chunk in writer.chunks])
    camera_k = np.concatenate([chunk[3]["K"] for chunk in writer.chunks])
    camera_to_world = np.concatenate([chunk[3]["camera_to_world"] for chunk in writer.chunks])
    timestamps = np.concatenate([chunk[3]["timestamps_ns"] for chunk in writer.chunks])
    source_timestamps = np.concatenate([chunk[3]["source_timestamps_ns"] for chunk in writer.chunks])
    assert depth.shape == (10, 16, 24)
    assert depth.dtype == np.float16
    assert valid.shape == depth.shape
    assert valid.dtype == np.bool_
    assert depth[0, 0, 0] == 0
    assert not valid[0, 0, 0]
    np.testing.assert_array_equal(depth[:, 1, 1], np.arange(1, 11, dtype=np.float16))
    np.testing.assert_array_equal(camera_k[:, 0, 0], np.arange(100, 110, dtype=np.float32))
    np.testing.assert_array_equal(camera_to_world[:, 0, 3], np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(timestamps, decoder.timestamps)
    np.testing.assert_array_equal(source_timestamps, decoder.source_timestamps)
    np.testing.assert_array_equal(np.diff(timestamps), np.full(9, 100_000_000, dtype=np.int64))

    assert writer.completion is not None
    assert writer.completion["frame_count"] == 10
    metadata = writer.completion["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["source"]["decoder_backend"] == "fake_grid"  # type: ignore[index]
    assert metadata["alignment"]["timestamp_unit"] == "nanosecond"  # type: ignore[index]
    assert metadata["alignment"]["timestamp_origin"] == "source_stream_start"  # type: ignore[index]
    assert metadata["alignment"]["sample_fps"] == _TEST_GRID.sample_fps  # type: ignore[index]
    assert metadata["alignment"]["source_timestamp_array"] == "source_timestamps_ns"  # type: ignore[index]
    assert metadata["grid"]["output_size"] == {"width": 24, "height": 16}  # type: ignore[index]
    assert metadata["grid"]["spatial_transform"] == "center_crop_resize"  # type: ignore[index]
    assert metadata["grid"]["rotation_degrees_clockwise"] == 90  # type: ignore[index]
    assert metadata["recipe"]["K_coordinate_space"] == "annotation_grid"  # type: ignore[index]
    assert metadata["arrays"]["depth"]["shape"] == [10, 16, 24]  # type: ignore[index]
    assert metadata["arrays"]["source_timestamps_ns"]["shape"] == [10]  # type: ignore[index]
    assert task.dataset_metadata["vipe_annotation"] == {
        "format": "npz-temporal-v1",
        "metadata_uri": "memory://complete",
        "producer_release": "vipe-dav3-grid-v1",
    }


def test_existing_completion_marker_skips_decode_and_inference(tmp_path: Path) -> None:
    """A completed clip is reused without touching the decoder or model."""
    source = tmp_path / "complete.mkv"
    source.touch()
    task = make_annotation_task(
        source,
        session_id="complete",
        relative_path="complete.mkv",
        span=(1.0, 2.0),
    )
    clip = task.video.clips[0]
    output = tmp_path / "output"
    metadata_path = _write_completion(
        output,
        clip_uuid=clip.uuid,
        source=source,
        span=(1.0, 2.0),
    )
    decoder = _FakeDecoder()
    stage, model, writer = _make_stage(tmp_path, decoder=decoder)

    stage.stage_setup()
    assert stage.process_data([task]) == [task]

    assert decoder.calls == []
    assert not model.infer_called
    assert writer.chunks == []
    assert writer.completion is None
    assert task.dataset_metadata["vipe_annotation"] == {
        "format": "npz-temporal-v1",
        "metadata_uri": str(metadata_path),
        "producer_release": "vipe-dav3-grid-v1",
    }


def test_full_source_completion_skips_decode_and_restores_clip(tmp_path: Path) -> None:
    """A full-source retry should resolve its stable UUID before video decode."""
    source = tmp_path / "whole-complete.mkv"
    source.touch()
    output = tmp_path / "output"
    span = (0.0, 1.0)
    clip_uuid = full_source_clip_uuid(source.resolve(), 0)
    metadata_path = _write_completion(
        output,
        clip_uuid=clip_uuid,
        source=source,
        span=span,
    )
    task = make_annotation_task(
        source,
        session_id="whole-complete",
        relative_path="whole-complete.mkv",
    )
    decoder = _FakeDecoder()
    stage, model, writer = _make_stage(tmp_path, decoder=decoder)

    stage.stage_setup()
    assert stage.process_data([task]) == [task]

    assert decoder.calls == []
    assert not model.infer_called
    assert writer.chunks == []
    assert len(task.video.clips) == 1
    assert task.video.clips[0].uuid == clip_uuid
    assert task.video.clips[0].span == span
    assert task.dataset_metadata["vipe_annotation"]["metadata_uri"] == str(metadata_path)


def test_completion_with_incomplete_output_contract_is_not_reused(tmp_path: Path) -> None:
    """Resume must not accept an artifact missing a required ViPE array."""
    source = tmp_path / "incomplete.mkv"
    source.touch()
    span = (1.0, 2.0)
    task = make_annotation_task(
        source,
        session_id="incomplete",
        relative_path="incomplete.mkv",
        span=span,
    )
    metadata_path = _write_completion(
        tmp_path / "output",
        clip_uuid=task.video.clips[0].uuid,
        source=source,
        span=span,
    )
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["metadata"]["arrays"]["source_timestamps_ns"]["dtype"] = "float32"
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    decoder = _FakeDecoder()
    stage, model, _ = _make_stage(tmp_path, decoder=decoder)

    stage.stage_setup()
    with pytest.raises(ValueError, match="incompatible output contract"):
        stage.process_data([task])

    assert decoder.calls == []
    assert not model.infer_called


@pytest.mark.parametrize("stored_span", [(0.0, 2.0), (1.0, 2.0)])
def test_full_source_completion_requires_consistent_whole_span(
    tmp_path: Path,
    stored_span: tuple[float, float],
) -> None:
    """A full-source completion must start at zero and match its grid frame count."""
    source = tmp_path / "wrong-whole-span.mkv"
    source.touch()
    output = tmp_path / "output"
    clip_uuid = full_source_clip_uuid(source.resolve(), 0)
    metadata_path = _write_completion(
        output,
        clip_uuid=clip_uuid,
        source=source,
        span=(0.0, 1.0),
    )
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["metadata"]["source"]["span_seconds"] = list(stored_span)
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    task = make_annotation_task(
        source,
        session_id="wrong-whole-span",
        relative_path="wrong-whole-span.mkv",
    )
    decoder = _FakeDecoder()
    stage, model, _ = _make_stage(tmp_path, decoder=decoder)

    stage.stage_setup()
    with pytest.raises(ValueError, match="incompatible output contract"):
        stage.process_data([task])

    assert decoder.calls == []
    assert not model.infer_called


def test_whole_video_annotation_task_creates_a_reusable_clip(tmp_path: Path) -> None:
    """An adapter task without clips should mean one full-source annotation."""
    source = tmp_path / "whole.mp4"
    source.touch()
    task = make_annotation_task(source, session_id="whole", relative_path="whole.mp4")
    decoder = _FakeDecoder(frame_count=8, source_span=(0.0, 0.8))
    stage, _, writer = _make_stage(tmp_path, decoder=decoder)

    stage.process_data([task])

    assert decoder.calls[0]["span"] is None
    assert len(task.video.clips) == 1
    assert task.video.clips[0].span == (0.0, 0.8)
    assert task.video.clips[0].source_video == str(source.resolve())
    assert task.video.num_total_clips == 1
    assert writer.completion is not None
    assert writer.completion["clip_uuid"] == str(task.video.clips[0].uuid)


def test_stage_records_odd_source_dimensions_but_infers_on_fixed_grid(tmp_path: Path) -> None:
    """Source dimensions should only affect raster metadata, not model input."""
    source = tmp_path / "odd-size.mp4"
    source.touch()
    task = make_annotation_task(
        source,
        session_id="odd-size",
        relative_path="odd-size.mp4",
        span=(0.0, 0.8),
    )
    decoder = _FakeDecoder(
        frame_count=8,
        source_height=15,
        source_width=23,
        source_span=(0.0, 0.8),
    )
    stage, model, writer = _make_stage(tmp_path, decoder=decoder)

    stage.process_data([task])

    assert model.received_shape == (8, 16, 24, 3)
    assert writer.completion is not None
    metadata = writer.completion["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["grid"]["source_size"] == {"width": 23, "height": 15}  # type: ignore[index]


def test_plain_split_pipe_task_uses_default_decode_hints(tmp_path: Path) -> None:
    """The base split task should decode its clip with stream zero and no rotation."""
    source = tmp_path / "split.mp4"
    source.touch()
    clip = Clip(uuid=uuid4(), source_video=str(source), span=(2.0, 2.8))
    task = SplitPipeTask(
        session_id="split",
        video=Video(input_video=source, clips=[clip], num_total_clips=1),
    )
    decoder = _FakeDecoder(frame_count=8, source_span=(2.0, 2.8))
    stage, _, _ = _make_stage(tmp_path, decoder=decoder)

    stage.process_data([task])

    assert decoder.calls[0]["stream_index"] == 0
    assert decoder.calls[0]["rotation"] == 0
    assert decoder.calls[0]["span"] == clip.span


@pytest.mark.parametrize(
    ("decoder", "max_frames", "message"),
    [
        (_FakeDecoder(frame_count=7), 32, "at least 8 grid frames"),
        (_FakeDecoder(frame_count=9), 8, "bounded full clip"),
    ],
)
def test_stage_rejects_unsupported_input_before_inference(
    tmp_path: Path,
    decoder: _FakeDecoder,
    max_frames: int,
    message: str,
) -> None:
    """Input constraints should fail before any GPU inference or output write."""
    source = tmp_path / "invalid.mp4"
    source.touch()
    task = make_annotation_task(
        source,
        session_id="invalid",
        relative_path="invalid.mp4",
        span=(1.0, 1.8),
    )
    stage, model, writer = _make_stage(tmp_path, decoder=decoder, max_frames=max_frames)

    with pytest.raises(ValueError, match=message):
        stage.process_data([task])

    assert not model.infer_called
    assert writer.chunks == []
    assert writer.completion is None


def test_stage_rejects_decoder_span_mismatch_before_inference(tmp_path: Path) -> None:
    """An injected decoder must honor the requested source interval."""
    source = tmp_path / "wrong-span.mp4"
    source.touch()
    task = make_annotation_task(
        source,
        session_id="wrong-span",
        relative_path="wrong-span.mp4",
        span=(1.0, 2.0),
    )
    decoder = _FakeDecoder(frame_count=8, source_span=(1.0, 1.8))
    stage, model, writer = _make_stage(tmp_path, decoder=decoder)

    with pytest.raises(ValueError, match="returned source_span"):
        stage.process_data([task])

    assert not model.infer_called
    assert writer.completion is None


def test_stage_rejects_changed_frame_indices_before_writing(tmp_path: Path) -> None:
    """ViPE must not silently drop or reorder the source frame timeline."""
    source = tmp_path / "misaligned.mp4"
    source.touch()
    task = make_annotation_task(
        source,
        session_id="misaligned",
        relative_path="misaligned.mp4",
        span=(1.0, 1.8),
    )
    decoder = _FakeDecoder(frame_count=8)
    model = _FakeViPEModel(bad_first_index=True)
    stage, _, writer = _make_stage(tmp_path, decoder=decoder, model=model)

    with pytest.raises(ValueError, match="changed or dropped frame indices"):
        stage.process_data([task])

    assert writer.chunks == []
    assert writer.completion is None


def test_stage_bounds_depth_before_float16_conversion(tmp_path: Path) -> None:
    """Out-of-range depth must become invalid zero rather than float16 infinity."""
    source = tmp_path / "depth-range.mp4"
    source.touch()
    task = make_annotation_task(
        source,
        session_id="depth-range",
        relative_path="depth-range.mp4",
        span=(1.0, 1.8),
    )
    depth = np.ones((16, 24), dtype=np.float32)
    depth[0, :6] = np.asarray(
        [np.nan, np.inf, -1.0, 1.0e-5, 1.0e8, 1.0e3],
        dtype=np.float32,
    )
    model = _FakeViPEModel(depth_overrides={0: depth})
    stage, _, writer = _make_stage(
        tmp_path,
        decoder=_FakeDecoder(frame_count=8),
        model=model,
    )

    stage.process_data([task])

    first_chunk = writer.chunks[0][3]
    stored_depth = first_chunk["depth"][0, 0, :6]
    stored_valid = first_chunk["valid"][0, 0, :6]
    np.testing.assert_array_equal(
        stored_valid,
        np.asarray([False, False, False, False, False, True]),
    )
    np.testing.assert_array_equal(
        stored_depth,
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0e3], dtype=np.float16),
    )
    assert np.isfinite(first_chunk["depth"]).all()
    assert writer.completion is not None
    metadata = writer.completion["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["recipe"]["valid_depth_range_meters"] == [1.0e-4, 1.0e4]  # type: ignore[index]
    assert metadata["recipe"]["min_valid_fraction_per_frame"] == 0.5  # type: ignore[index]


def test_stage_rejects_one_frame_below_valid_depth_fraction(tmp_path: Path) -> None:
    """The quality threshold applies independently to every output frame."""
    source = tmp_path / "low-valid-depth.mp4"
    source.touch()
    task = make_annotation_task(
        source,
        session_id="low-valid-depth",
        relative_path="low-valid-depth.mp4",
        span=(1.0, 1.8),
    )
    low_valid_depth = np.zeros((16, 24), dtype=np.float32)
    low_valid_depth.reshape(-1)[:191] = 1.0
    model = _FakeViPEModel(depth_overrides={3: low_valid_depth})
    stage, _, writer = _make_stage(
        tmp_path,
        decoder=_FakeDecoder(frame_count=8),
        model=model,
    )

    with pytest.raises(
        ValueError,
        match=r"frame 3 valid depth fraction .* below the required 0\.5000",
    ):
        stage.process_data([task])

    assert writer.completion is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_valid_fraction": -0.1}, "min_valid_fraction"),
        ({"min_valid_fraction": 1.1}, "min_valid_fraction"),
        ({"min_valid_fraction": float("nan")}, "min_valid_fraction"),
        ({"gpus_per_worker": 0.0}, "gpus_per_worker"),
        ({"gpus_per_worker": 1.1}, "gpus_per_worker"),
        ({"gpus_per_worker": float("inf")}, "gpus_per_worker"),
    ],
)
def test_stage_rejects_invalid_quality_or_resource_configuration(
    tmp_path: Path,
    kwargs: dict[str, float],
    message: str,
) -> None:
    """Resource fractions and quality thresholds should fail at construction."""
    with pytest.raises(ValueError, match=message):
        ViPEStage(
            tmp_path / "output",
            _FakeViPEModel(),
            decoder=_FakeDecoder(),
            **kwargs,
        )


def test_stage_exposes_configured_gpu_fraction(tmp_path: Path) -> None:
    """Fractional scheduling should be explicit and retain the safe default."""
    default_stage = ViPEStage(
        tmp_path / "default-output",
        _FakeViPEModel(),
        decoder=_FakeDecoder(),
    )
    shared_h100_stage = ViPEStage(
        tmp_path / "shared-output",
        _FakeViPEModel(),
        gpus_per_worker=0.5,
        decoder=_FakeDecoder(),
    )

    assert default_stage.resources.gpus == 1.0
    assert shared_h100_stage.resources.gpus == 0.5


def test_pyav_decode_builds_regular_rotated_annotation_grid(tmp_path: Path) -> None:
    """Direct decode should emit the configured grid while retaining source PTS."""
    source = tmp_path / "native.ts"
    fps = 10
    with av.open(str(source), mode="w", format="mpegts") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 24
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, fps)
        for index in range(10):
            array = np.full((stream.height, stream.width, 3), index * 10, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    decoded = decode_local_vipe_clip(
        source,
        None,
        stream_index=0,
        rotation_degrees_clockwise=90,
        grid=_TEST_GRID,
        min_frames=8,
        max_frames=16,
    )

    assert decoded.decoder_backend in {"pyav_seek_grid", "pyav_sequential_grid"}
    assert decoded.frames.shape == (10, 16, 24, 3)
    assert decoded.timestamps_ns[0] == 0
    np.testing.assert_array_equal(
        decoded.timestamps_ns,
        np.arange(10, dtype=np.int64) * 100_000_000,
    )
    np.testing.assert_array_equal(decoded.source_timestamps_ns, decoded.timestamps_ns)
    assert decoded.raster.rotation_degrees_clockwise == 90
    assert (decoded.raster.oriented_width, decoded.raster.oriented_height) == (16, 24)
    assert decoded.source_span[0] == 0.0
    assert decoded.source_span[1] == pytest.approx(1.0)
