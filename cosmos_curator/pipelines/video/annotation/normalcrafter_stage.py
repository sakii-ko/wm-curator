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

"""Direct-decode NormalCrafter stage with bounded, chunked annotation output."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    PipelineTask,
)
from cosmos_curator.models.normalcrafter import (
    NORMALCRAFTER_CONDITIONING_FPS,
    NORMALCRAFTER_PADDING_MULTIPLE,
    NORMALCRAFTER_VAE_CHUNK_SIZE,
    NORMALCRAFTER_WINDOW_SIZE,
    NORMALCRAFTER_WINDOW_STRIDE,
    NormalCrafterModel,
)
from cosmos_curator.pipelines.video.annotation.artifact_writer import (
    TemporalAnnotationWriter,
)
from cosmos_curator.pipelines.video.annotation.data_model import (
    make_full_source_clip,
    normalize_span,
    resolve_source_clip_request,
    source_path_string,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask

_NANOSECONDS_PER_SECOND = 1_000_000_000
_SAMPLE_FPS = 15.0
_PRODUCER_RELEASE = "normalcrafter-v1"
_VIDEO_NDIM = 4
_RGB_CHANNELS = 3


class _IndexedDecodeUnavailableError(RuntimeError):
    """Signal that the PyAV timestamp fallback should be used."""


@dataclass(frozen=True, slots=True)
class DecodedNormalCrafterClip:
    """Sampled RGB frames and exact selected source timestamps."""

    frames: npt.NDArray[np.uint8]
    timestamps_ns: npt.NDArray[np.int64]
    source_span: tuple[float, float]
    decoder_backend: str


class NormalCrafterClipDecoder(Protocol):
    """Decode one source span at NormalCrafter's fixed sampling rate."""

    def __call__(  # noqa: PLR0913
        self,
        source: Path,
        span: tuple[float, float] | None,
        *,
        stream_index: int,
        sample_fps: float,
        min_frames: int,
        max_frames: int,
    ) -> DecodedNormalCrafterClip:
        """Decode the requested span, or probe and decode the whole source."""


class TemporalWriter(Protocol):
    """Writer surface used by this stage."""

    def open_chunk(
        self,
        clip_uuid: str | UUID,
        frame_start: int,
        frame_stop: int,
    ) -> AbstractContextManager[Path]:
        """Open a temporary chunk path."""

    def complete_clip(
        self,
        clip_uuid: str | UUID,
        *,
        frame_count: int,
        chunk_frames: int,
        metadata: dict[str, object],
    ) -> str:
        """Publish the completion record after all chunks."""


class NormalCrafterStage(CuratorStage):
    """Run one bounded NormalCrafter clip and publish seven-frame NPZ chunks."""

    def __init__(  # noqa: PLR0913
        self,
        output_path: str | Path,
        inference_model: NormalCrafterModel,
        *,
        profile_name: str = "default",
        tmp_dir: str | Path | None = None,
        decoder: NormalCrafterClipDecoder | None = None,
        writer: TemporalWriter | None = None,
    ) -> None:
        """Configure lightweight actor state; model loading happens in setup."""
        if not str(output_path).strip():
            message = "output_path must be non-empty"
            raise ValueError(message)
        if not profile_name.strip():
            message = "profile_name must be non-empty"
            raise ValueError(message)
        self._output_path = output_path
        self._profile_name = profile_name
        self._tmp_dir = tmp_dir
        self._inference_model = inference_model
        self._decoder = decoder or decode_normalcrafter_clip
        self._writer = writer

    @property
    def resources(self) -> CuratorStageResource:
        """Reserve one GPU for the resident model and CPU capacity for decode."""
        return CuratorStageResource(cpus=4.0, gpus=1.0)

    @property
    def model(self) -> ModelInterface:
        """Expose the lazy model to the standard stage lifecycle."""
        return self._inference_model

    def stage_setup(self) -> None:
        """Load the actor-local model and writer."""
        super().stage_setup()
        if self._writer is None:
            self._writer = TemporalAnnotationWriter(
                self._output_path,
                profile_name=self._profile_name,
                tmp_dir=self._tmp_dir,
            )

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Decode and annotate one source-backed task."""
        if len(tasks) != 1:
            message = f"NormalCrafterStage requires batch size one, got {len(tasks)}"
            raise ValueError(message)
        task = tasks[0]
        if not isinstance(task, SplitPipeTask):
            message = f"NormalCrafterStage requires AnnotationTask or SplitPipeTask, got {type(task).__name__}"
            raise TypeError(message)
        request = resolve_source_clip_request(task)
        source = _validated_source(request.source)
        clip = request.clip
        decoded = self._decoder(
            source,
            request.span,
            stream_index=request.stream_index,
            sample_fps=_SAMPLE_FPS,
            min_frames=NORMALCRAFTER_WINDOW_SIZE,
            max_frames=self._inference_model.max_frames,
        )
        frames, timestamps_ns, decoded_span = _validate_decoded_clip(
            decoded,
            max_frames=self._inference_model.max_frames,
        )
        if request.rotation_degrees_clockwise:
            frames = np.rot90(
                frames,
                k=-(request.rotation_degrees_clockwise // 90),
                axes=(1, 2),
            ).copy()

        span = request.span or decoded_span
        if clip is None:
            clip = make_full_source_clip(source, span, request.stream_index)
            task.video.clips.append(clip)
            task.video.num_total_clips = max(task.video.num_total_clips, 1)

        metadata_uri = self._infer_and_write(
            clip,
            source=source,
            span=span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
            decoder_backend=decoded.decoder_backend,
            frames=frames,
            timestamps_ns=timestamps_ns,
        )
        dataset_metadata = getattr(task, "dataset_metadata", None)
        if isinstance(dataset_metadata, dict):
            dataset_metadata["normal_annotation"] = {
                "format": "npz-temporal-v1",
                "metadata_uri": metadata_uri,
                "producer_release": _PRODUCER_RELEASE,
            }
        return tasks

    def destroy(self) -> None:
        """Release the resident model."""
        self._inference_model.close()

    def _infer_and_write(  # noqa: PLR0913
        self,
        clip: Clip,
        *,
        source: Path,
        span: tuple[float, float],
        stream_index: int,
        rotation_degrees_clockwise: int,
        decoder_backend: str,
        frames: npt.NDArray[np.uint8],
        timestamps_ns: npt.NDArray[np.int64],
    ) -> str:
        frame_count, height, width, _ = frames.shape
        expected_start = 0
        valid_count = 0
        chunks = iter(self._inference_model.infer(frames))
        try:
            for chunk in chunks:
                if chunk.frame_start != expected_start:
                    message = (
                        "NormalCrafter output chunks lost temporal alignment: "
                        f"expected={expected_start}, observed={chunk.frame_start}"
                    )
                    raise ValueError(message)
                frame_stop = chunk.frame_stop
                if frame_stop > frame_count:
                    message = f"NormalCrafter output stops at {frame_stop}, past input frame_count={frame_count}"
                    raise ValueError(message)
                valid_count += int(chunk.valid.sum())
                writer = self._require_writer()
                with writer.open_chunk(
                    clip.uuid,
                    chunk.frame_start,
                    frame_stop,
                ) as chunk_path:
                    np.savez(
                        chunk_path,
                        normal=chunk.normal,
                        valid=chunk.valid,
                        timestamps_ns=np.ascontiguousarray(
                            timestamps_ns[chunk.frame_start : frame_stop],
                            dtype=np.int64,
                        ),
                    )
                expected_start = frame_stop
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()

        if expected_start != frame_count:
            message = f"NormalCrafter produced {expected_start} frames, expected {frame_count}"
            raise ValueError(message)

        return self._require_writer().complete_clip(
            clip.uuid,
            frame_count=frame_count,
            chunk_frames=NORMALCRAFTER_VAE_CHUNK_SIZE,
            metadata={
                "producer": {
                    "id": "normalcrafter",
                    "release": _PRODUCER_RELEASE,
                    "model_id": self._inference_model.model_id,
                },
                "source": {
                    "path": source_path_string(source),
                    "span_seconds": [span[0], span[1]],
                    "stream_index": stream_index,
                    "rotation_degrees_clockwise": rotation_degrees_clockwise,
                    "decoder_backend": decoder_backend,
                },
                "alignment": {
                    "sample_fps": _SAMPLE_FPS,
                    "timestamp_array": "timestamps_ns",
                    "timestamp_unit": "nanosecond",
                    "timestamp_origin": "source_stream_start",
                },
                "recipe": {
                    "window_size": NORMALCRAFTER_WINDOW_SIZE,
                    "window_stride": NORMALCRAFTER_WINDOW_STRIDE,
                    "vae_chunk_size": NORMALCRAFTER_VAE_CHUNK_SIZE,
                    "padding": f"centered_white_to_multiple_{NORMALCRAFTER_PADDING_MULTIPLE}",
                    "attention_backend": "sdpa",
                    "conditioning_fps": NORMALCRAFTER_CONDITIONING_FPS,
                    "axis_transform_from_raw": [-1, 1, 1],
                    "normalization": "unit_length",
                    "invalid_value": [0.0, 0.0, 0.0],
                    "input_buffering": "full_clip_bounded",
                    "max_frames": self._inference_model.max_frames,
                },
                "valid_fraction": valid_count / (frame_count * height * width),
                "arrays": {
                    "normal": {
                        "axes": "THWC",
                        "dtype": "float16",
                        "shape": [frame_count, height, width, 3],
                    },
                    "valid": {
                        "axes": "THW",
                        "dtype": "bool",
                        "shape": [frame_count, height, width],
                    },
                    "timestamps_ns": {
                        "axes": "T",
                        "dtype": "int64",
                        "shape": [frame_count],
                    },
                },
            },
        )

    def _require_writer(self) -> TemporalWriter:
        if self._writer is None:
            message = "NormalCrafterStage.stage_setup() must be called before process_data()"
            raise RuntimeError(message)
        return self._writer


def decode_normalcrafter_clip(  # noqa: PLR0913
    source: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> DecodedNormalCrafterClip:
    """Prefer indexed sampling and fall back to PyAV timestamp/frame-id decode."""
    try:
        return _decode_with_camera_sensor(
            source,
            span,
            stream_index=stream_index,
            sample_fps=sample_fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
    except _IndexedDecodeUnavailableError as indexed_error:
        try:
            return _decode_with_pyav(
                source,
                span,
                stream_index=stream_index,
                sample_fps=sample_fps,
                min_frames=min_frames,
                max_frames=max_frames,
            )
        except Exception as fallback_error:
            message = (
                "both indexed CameraSensor and PyAV fallback failed: "
                f"CameraSensor={indexed_error}; PyAV={fallback_error}"
            )
            raise RuntimeError(message) from fallback_error


def _decode_with_camera_sensor(  # noqa: PLR0913
    source: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> DecodedNormalCrafterClip:
    from cosmos_curator.core.sensors.sampling.grid import (  # noqa: PLC0415
        SamplingGrid,
        make_ts_grid,
    )
    from cosmos_curator.core.sensors.sampling.spec import SamplingSpec  # noqa: PLC0415
    from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor  # noqa: PLC0415

    try:
        sensor = CameraSensor(source, stream_idx=stream_index)
    except Exception as error:
        message = "CameraSensor could not index the source container"
        raise _IndexedDecodeUnavailableError(message) from error

    source_duration_ns = _timeline_duration_ns(
        sensor.timestamps_ns,
        fallback_period_ns=max(1, round(_NANOSECONDS_PER_SECOND / sample_fps)),
    )
    resolved_span = _resolve_span(span, source_duration_ns)
    absolute_start_ns = sensor.start_ns + round(resolved_span[0] * _NANOSECONDS_PER_SECOND)
    absolute_stop_ns = sensor.start_ns + round(resolved_span[1] * _NANOSECONDS_PER_SECOND)
    grid_start_ns, exclusive_end_ns, sample_timestamps_ns = make_ts_grid(
        absolute_start_ns,
        exclusive_end_ns=absolute_stop_ns,
        sample_rate_hz=sample_fps,
    )
    _validate_sample_count(
        len(sample_timestamps_ns),
        min_frames=min_frames,
        max_frames=max_frames,
        span=resolved_span,
    )
    duration_ns = exclusive_end_ns - grid_start_ns
    grid = SamplingGrid(
        start_ns=grid_start_ns,
        exclusive_end_ns=exclusive_end_ns,
        timestamps_ns=sample_timestamps_ns,
        stride_ns=duration_ns,
        duration_ns=duration_ns,
    )
    try:
        batches = sensor.sample(SamplingSpec(grid=grid))
        batch = next(batches)
        try:
            next(batches)
        except StopIteration:
            pass
        else:
            message = "CameraSensor returned multiple batches for one clip"
            raise RuntimeError(message)
    except Exception as error:
        message = "CameraSensor could not decode the indexed sample grid"
        raise _IndexedDecodeUnavailableError(message) from error

    frames = np.ascontiguousarray(batch.frames, dtype=np.uint8)
    timestamps_ns = np.ascontiguousarray(
        batch.sensor_timestamps_ns - sensor.start_ns,
        dtype=np.int64,
    )
    if len(frames) != len(sample_timestamps_ns):
        message = f"CameraSensor returned {len(frames)} frames for {len(sample_timestamps_ns)} requested timestamps"
        raise _IndexedDecodeUnavailableError(message)
    return DecodedNormalCrafterClip(
        frames=frames,
        timestamps_ns=timestamps_ns,
        source_span=resolved_span,
        decoder_backend="camera_sensor",
    )


def _decode_with_pyav(  # noqa: PLR0913
    source: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> DecodedNormalCrafterClip:
    from cosmos_curator.pipelines.video.utils.decoder_utils import (  # noqa: PLC0415
        decode_video_cpu_frame_ids,
        get_video_timestamps,
        sample_closest,
        save_stream_position,
    )

    with save_stream_position(source):
        source_timestamps = get_video_timestamps(source, stream_idx=stream_index)
    if len(source_timestamps) == 0:
        message = "source video contains no displayable frames"
        raise ValueError(message)
    origin = float(source_timestamps[0])
    relative_timestamps = source_timestamps.astype(np.float64) - origin
    source_duration_ns = _timeline_duration_ns(
        np.rint(relative_timestamps * _NANOSECONDS_PER_SECOND).astype(np.int64),
        fallback_period_ns=max(1, round(_NANOSECONDS_PER_SECOND / sample_fps)),
    )
    resolved_span = _resolve_span(span, source_duration_ns)
    frame_ids, counts, _ = sample_closest(
        source_timestamps,
        sample_rate=sample_fps,
        start=origin + resolved_span[0],
        stop=origin + resolved_span[1],
        endpoint=False,
        dedup=True,
    )
    frame_count = int(counts.sum())
    _validate_sample_count(
        frame_count,
        min_frames=min_frames,
        max_frames=max_frames,
        span=resolved_span,
    )
    with save_stream_position(source):
        frames = decode_video_cpu_frame_ids(
            source,
            frame_ids,
            counts,
            stream_idx=stream_index,
        )
    selected_timestamps = np.repeat(source_timestamps[frame_ids], counts)
    timestamps_ns = np.rint((selected_timestamps.astype(np.float64) - origin) * _NANOSECONDS_PER_SECOND).astype(
        np.int64
    )
    return DecodedNormalCrafterClip(
        frames=np.ascontiguousarray(frames, dtype=np.uint8),
        timestamps_ns=np.ascontiguousarray(timestamps_ns),
        source_span=resolved_span,
        decoder_backend="pyav",
    )


def _validate_decoded_clip(
    decoded: DecodedNormalCrafterClip,
    *,
    max_frames: int,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.int64],
    tuple[float, float],
]:
    frames = decoded.frames
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != _VIDEO_NDIM
        or frames.shape[-1] != _RGB_CHANNELS
    ):
        message = "decoded NormalCrafter frames must be uint8 [T,H,W,3] RGB"
        raise ValueError(message)
    frame_count = frames.shape[0]
    _validate_sample_count(
        frame_count,
        min_frames=NORMALCRAFTER_WINDOW_SIZE,
        max_frames=max_frames,
        span=decoded.source_span,
    )
    timestamps_ns = decoded.timestamps_ns
    if (
        not isinstance(timestamps_ns, np.ndarray)
        or timestamps_ns.dtype != np.int64
        or timestamps_ns.ndim != 1
        or len(timestamps_ns) != frame_count
    ):
        message = f"decoded timestamps_ns must be int64 [T] aligned with {frame_count} frames"
        raise ValueError(message)
    if timestamps_ns[0] < 0 or (frame_count > 1 and bool(np.any(np.diff(timestamps_ns) < 0))):
        message = "decoded timestamps_ns must be non-negative and non-decreasing"
        raise ValueError(message)
    source_span = _validated_span(decoded.source_span)
    if not decoded.decoder_backend.strip():
        message = "decoded decoder_backend must be non-empty"
        raise ValueError(message)
    return (
        np.ascontiguousarray(frames),
        np.ascontiguousarray(timestamps_ns),
        source_span,
    )


def _timeline_duration_ns(
    timestamps_ns: npt.NDArray[np.int64],
    *,
    fallback_period_ns: int,
) -> int:
    if len(timestamps_ns) == 0:
        message = "source timeline must be non-empty"
        raise ValueError(message)
    differences = np.diff(timestamps_ns)
    positive_differences = differences[differences > 0]
    period_ns = int(np.median(positive_differences)) if len(positive_differences) else fallback_period_ns
    return int(timestamps_ns[-1]) - int(timestamps_ns[0]) + max(1, period_ns)


def _resolve_span(
    span: tuple[float, float] | None,
    source_duration_ns: int,
) -> tuple[float, float]:
    source_duration = source_duration_ns / _NANOSECONDS_PER_SECOND
    if span is None:
        return 0.0, source_duration
    start, stop = _validated_span(span)
    tolerance = 1.0 / _NANOSECONDS_PER_SECOND
    if start >= source_duration or stop > source_duration + tolerance:
        message = f"clip span {span} exceeds source duration {source_duration:.9f} seconds"
        raise ValueError(message)
    return start, min(stop, source_duration)


def _validate_sample_count(
    frame_count: int,
    *,
    min_frames: int,
    max_frames: int,
    span: tuple[float, float],
) -> None:
    if frame_count < min_frames:
        message = f"NormalCrafter requires at least {min_frames} sampled frames, got {frame_count} in span {span}"
        raise ValueError(message)
    if frame_count > max_frames:
        message = (
            "NormalCrafter retains full-clip conditioning latents and this "
            f"runtime is explicitly bounded to max_frames={max_frames}; "
            f"span {span} sampled {frame_count} frames, split it upstream"
        )
        raise ValueError(message)


def _validated_source(source: object) -> Path:
    if isinstance(source, Path):
        resolved = source.expanduser().resolve(strict=True)
        if not resolved.is_file():
            message = f"NormalCrafter source is not a file: {resolved}"
            raise ValueError(message)
        return resolved
    message = f"NormalCrafter source must be a local pathlib.Path, got {type(source).__name__}"
    raise TypeError(message)


def _validated_span(span: object) -> tuple[float, float]:
    normalized = normalize_span(span)
    if normalized is None:
        message = "Clip.span must not be empty"
        raise ValueError(message)
    return normalized


__all__ = [
    "DecodedNormalCrafterClip",
    "NormalCrafterClipDecoder",
    "NormalCrafterStage",
    "decode_normalcrafter_clip",
]
