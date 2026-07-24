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
"""Direct-decode ViPE annotation stage with bounded full-clip input memory."""

import math
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID

import av
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, PipelineTask
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.models.vipe import ViPEFrameResult
from cosmos_curator.pipelines.video.annotation.artifact_writer import (
    TemporalAnnotationReader,
    TemporalAnnotationWriter,
)
from cosmos_curator.pipelines.video.annotation.data_model import (
    make_full_source_clip,
    normalize_span,
    resolve_source_clip_request,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask

_NANOSECONDS_PER_SECOND = 1_000_000_000
_FRAME_ARRAY_NDIM = 4
_RGB_CHANNELS = 3
_MIN_VALID_DEPTH_METERS = 1.0e-4
_MAX_VALID_DEPTH_METERS = 1.0e4
VIPE_MIN_FRAMES = 8


class _IndexedDecodeUnavailableError(RuntimeError):
    """Signal that the sequential PyAV compatibility path should be used."""


@dataclass(frozen=True, slots=True)
class DecodedViPEClip:
    """Native-rate RGB frames and their exact source-relative timestamps."""

    frames: npt.NDArray[np.uint8]
    timestamps_ns: npt.NDArray[np.int64]
    fps: float
    source_span: tuple[float, float]
    decoder_backend: str


class ViPEInferenceModel(Protocol):
    """Inference portion of the ViPE model used by this stage."""

    @property
    def conda_env_name(self) -> str:
        """Return the model environment."""

    @property
    def model_id_names(self) -> list[str]:
        """Return model IDs managed by Curator."""

    def setup(self) -> None:
        """Load the resident model."""

    def infer(
        self,
        frames: npt.NDArray[np.uint8],
        *,
        name: str,
        fps: float,
    ) -> Iterator[ViPEFrameResult]:
        """Yield aligned ViPE results."""

    def close(self) -> None:
        """Release resident model references."""


class ViPEClipDecoder(Protocol):
    """Decode one native-rate source span into bounded host memory."""

    def __call__(  # noqa: PLR0913
        self,
        source_path: Path,
        span: tuple[float, float] | None,
        *,
        stream_index: int,
        rotation_degrees_clockwise: int,
        min_frames: int,
        max_frames: int,
    ) -> DecodedViPEClip:
        """Decode the requested span."""


class TemporalWriter(Protocol):
    """Subset of TemporalAnnotationWriter used by the stage."""

    def open_chunk(
        self,
        clip_uuid: str | UUID,
        frame_start: int,
        frame_stop: int,
    ) -> AbstractContextManager[Path]:
        """Open a staged chunk path."""

    def complete_clip(
        self,
        clip_uuid: str | UUID,
        *,
        frame_count: int,
        chunk_frames: int,
        metadata: dict[str, object],
    ) -> str:
        """Publish the completion record."""


class ViPEStage(CuratorStage):
    """Run ViPE for one source-backed clip and publish temporal NPZ chunks."""

    def __init__(  # noqa: PLR0913
        self,
        output_path: str | Path,
        inference_model: ViPEInferenceModel,
        *,
        profile_name: str = "default",
        tmp_dir: str | Path | None = None,
        chunk_frames: int = 16,
        max_frames: int = 2048,
        min_valid_fraction: float = 0.5,
        gpus_per_worker: float = 1.0,
        decoder: ViPEClipDecoder | None = None,
        writer: TemporalWriter | None = None,
    ) -> None:
        """Configure one batch-one actor.

        ViPE's SLAM and depth alignment are global over a clip. This adapter
        therefore decodes one complete clip, rejects clips above ``max_frames``
        before inference, and streams only model outputs into bounded chunks.
        """
        if not str(output_path).strip():
            msg = "output_path must be non-empty"
            raise ValueError(msg)
        if not profile_name.strip():
            msg = "profile_name must be non-empty"
            raise ValueError(msg)
        self._chunk_frames = _positive_int(chunk_frames, field_name="chunk_frames")
        self._max_frames = _positive_int(max_frames, field_name="max_frames")
        if self._max_frames < VIPE_MIN_FRAMES:
            msg = f"max_frames must be at least {VIPE_MIN_FRAMES}"
            raise ValueError(msg)
        if (
            isinstance(min_valid_fraction, bool)
            or not isinstance(min_valid_fraction, (int, float))
            or not math.isfinite(min_valid_fraction)
            or not 0.0 <= min_valid_fraction <= 1.0
        ):
            msg = "min_valid_fraction must be finite and between 0 and 1"
            raise ValueError(msg)
        if (
            isinstance(gpus_per_worker, bool)
            or not isinstance(gpus_per_worker, (int, float))
            or not math.isfinite(gpus_per_worker)
            or not 0.0 < gpus_per_worker <= 1.0
        ):
            msg = "gpus_per_worker must be finite and in the interval (0, 1]"
            raise ValueError(msg)

        self._output_path = output_path
        self._profile_name = profile_name
        self._tmp_dir = tmp_dir
        self._min_valid_fraction = float(min_valid_fraction)
        self._gpus_per_worker = float(gpus_per_worker)
        self._inference_model = inference_model
        self._decoder = decoder or decode_local_vipe_clip
        self._writer = writer
        self._reader: TemporalAnnotationReader | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Reserve the configured GPU scheduling share for the resident model."""
        return CuratorStageResource(cpus=2.0, gpus=self._gpus_per_worker)

    @property
    def model(self) -> ModelInterface:
        """Expose the inference model to the Curator stage lifecycle."""
        return cast("ModelInterface", self._inference_model)

    def stage_setup(self) -> None:
        """Load ViPE in the target environment and construct the actor-local writer."""
        super().stage_setup()
        if self._writer is None:
            self._writer = TemporalAnnotationWriter(
                self._output_path,
                profile_name=self._profile_name,
                tmp_dir=self._tmp_dir,
            )
        self._reader = TemporalAnnotationReader(
            self._output_path,
            profile_name=self._profile_name,
        )

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Decode one task's clip, treating an empty clip list as the full video."""
        if len(tasks) != 1:
            msg = f"ViPEStage requires batch size one, got {len(tasks)}"
            raise ValueError(msg)
        task = tasks[0]
        if not isinstance(task, SplitPipeTask):
            msg = f"ViPEStage requires AnnotationTask or SplitPipeTask, got {type(task).__name__}"
            raise TypeError(msg)
        request = resolve_source_clip_request(task)
        if not isinstance(request.source, Path):
            msg = "ViPEStage currently requires a local pathlib.Path source"
            raise TypeError(msg)
        source_path = request.source.expanduser().resolve(strict=True)
        if not source_path.is_file():
            msg = f"ViPE source is not a file: {source_path}"
            raise ValueError(msg)
        clip = request.clip
        if clip is not None and self._reuse_completed(task, clip):
            return tasks
        decoded = self._decoder(
            source_path,
            request.span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
            min_frames=VIPE_MIN_FRAMES,
            max_frames=self._max_frames,
        )
        frames, timestamps_ns, fps, decoded_span = self._validate_decoded(decoded)
        span = request.span or decoded_span
        if clip is None:
            clip = make_full_source_clip(source_path, span, request.stream_index)
            task.video.clips.append(clip)
            task.video.num_total_clips = max(task.video.num_total_clips, 1)
        if self._reuse_completed(task, clip):
            return tasks
        metadata_uri = self._infer_and_write(
            clip,
            source_path=source_path,
            span=span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
            frames=frames,
            timestamps_ns=timestamps_ns,
            fps=fps,
            decoder_backend=decoded.decoder_backend,
        )
        self._set_annotation_reference(task, metadata_uri)
        return tasks

    def destroy(self) -> None:
        """Release the resident ViPE model."""
        self._inference_model.close()

    def _validate_decoded(
        self,
        decoded: DecodedViPEClip,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64], float, tuple[float, float]]:
        frames = decoded.frames
        timestamps_ns = decoded.timestamps_ns
        fps = float(decoded.fps)
        if (
            not isinstance(frames, np.ndarray)
            or frames.dtype != np.uint8
            or frames.ndim != _FRAME_ARRAY_NDIM
            or frames.shape[-1] != _RGB_CHANNELS
        ):
            msg = "decoded ViPE frames must be uint8 [T,H,W,3] RGB"
            raise ValueError(msg)
        frame_count = frames.shape[0]
        if frame_count < VIPE_MIN_FRAMES:
            msg = f"ViPE requires at least {VIPE_MIN_FRAMES} frames, got {frame_count}"
            raise ValueError(msg)
        if frame_count > self._max_frames:
            msg = (
                "ViPE requires full-clip SLAM and this adapter does not claim streaming input support: "
                f"got {frame_count} frames, max_frames={self._max_frames}; split the source span upstream"
            )
            raise ValueError(msg)
        if (
            not isinstance(timestamps_ns, np.ndarray)
            or timestamps_ns.dtype != np.int64
            or timestamps_ns.ndim != 1
            or len(timestamps_ns) != frame_count
        ):
            msg = f"decoded timestamps_ns must be int64 [T] aligned with {frame_count} frames"
            raise ValueError(msg)
        if frame_count > 1 and bool(np.any(np.diff(timestamps_ns) <= 0)):
            msg = "decoded timestamps_ns must be strictly increasing"
            raise ValueError(msg)
        if not math.isfinite(fps) or fps <= 0:
            msg = "decoded source fps must be finite and positive"
            raise ValueError(msg)
        source_span = normalize_span(decoded.source_span)
        if source_span is None:
            msg = "decoded source_span must not be empty"
            raise ValueError(msg)
        if not isinstance(decoded.decoder_backend, str) or not decoded.decoder_backend:
            msg = "decoded decoder_backend must be a non-empty string"
            raise ValueError(msg)
        return np.ascontiguousarray(frames), np.ascontiguousarray(timestamps_ns), fps, source_span

    def _infer_and_write(  # noqa: PLR0913
        self,
        clip: Clip,
        *,
        source_path: Path,
        span: tuple[float, float],
        stream_index: int,
        rotation_degrees_clockwise: int,
        frames: npt.NDArray[np.uint8],
        timestamps_ns: npt.NDArray[np.int64],
        fps: float,
        decoder_backend: str,
    ) -> str:
        frame_count, height, width, _ = frames.shape
        chunk_depth: list[npt.NDArray[np.float16]] = []
        chunk_valid: list[npt.NDArray[np.bool_]] = []
        chunk_k: list[npt.NDArray[np.float32]] = []
        chunk_c2w: list[npt.NDArray[np.float32]] = []
        chunk_start = 0
        valid_count = 0
        result_count = 0

        results = iter(self._inference_model.infer(frames, name=str(clip.uuid), fps=fps))
        try:
            for result in results:
                if result_count >= frame_count:
                    msg = f"ViPE returned more than the expected {frame_count} frames"
                    raise ValueError(msg)
                depth, valid, camera_k, camera_to_world = self._canonicalize_result(
                    result,
                    expected_index=result_count,
                    height=height,
                    width=width,
                )
                chunk_depth.append(depth)
                chunk_valid.append(valid)
                chunk_k.append(camera_k)
                chunk_c2w.append(camera_to_world)
                valid_count += int(valid.sum())
                result_count += 1
                if len(chunk_depth) == self._chunk_frames:
                    self._write_chunk(
                        clip,
                        frame_start=chunk_start,
                        frame_stop=result_count,
                        depth=chunk_depth,
                        valid=chunk_valid,
                        camera_k=chunk_k,
                        camera_to_world=chunk_c2w,
                        timestamps_ns=timestamps_ns[chunk_start:result_count],
                    )
                    chunk_start = result_count
                    chunk_depth.clear()
                    chunk_valid.clear()
                    chunk_k.clear()
                    chunk_c2w.clear()
        finally:
            close = getattr(results, "close", None)
            if close is not None:
                close()

        if result_count != frame_count:
            msg = f"ViPE returned {result_count} frames; expected {frame_count}"
            raise ValueError(msg)
        if chunk_depth:
            self._write_chunk(
                clip,
                frame_start=chunk_start,
                frame_stop=result_count,
                depth=chunk_depth,
                valid=chunk_valid,
                camera_k=chunk_k,
                camera_to_world=chunk_c2w,
                timestamps_ns=timestamps_ns[chunk_start:result_count],
            )

        valid_fraction = valid_count / (frame_count * height * width)
        writer = self._require_writer()
        return writer.complete_clip(
            clip.uuid,
            frame_count=frame_count,
            chunk_frames=self._chunk_frames,
            metadata={
                "producer": {"id": "vipe", "pipeline": "dav3"},
                "source": {
                    "path": str(source_path),
                    "span_seconds": [span[0], span[1]],
                    "stream_index": stream_index,
                    "rotation_degrees_clockwise": rotation_degrees_clockwise,
                    "decoder_backend": decoder_backend,
                },
                "alignment": {
                    "source_fps": fps,
                    "timestamp_array": "timestamps_ns",
                    "timestamp_unit": "nanosecond",
                    "timestamp_origin": "source_stream_start",
                },
                "recipe": {
                    "depth_representation": "z_depth",
                    "depth_scale": "metric",
                    "length_unit": "meter",
                    "valid_depth_range_meters": [
                        _MIN_VALID_DEPTH_METERS,
                        _MAX_VALID_DEPTH_METERS,
                    ],
                    "min_valid_fraction_per_frame": self._min_valid_fraction,
                    "camera_pose": "camera_to_world",
                    "camera_convention": "opencv",
                    "input_buffering": "full_clip_bounded",
                    "max_frames": self._max_frames,
                },
                "valid_fraction": valid_fraction,
                "arrays": {
                    "depth": {"axes": "THW", "dtype": "float16", "shape": [frame_count, height, width]},
                    "valid": {"axes": "THW", "dtype": "bool", "shape": [frame_count, height, width]},
                    "K": {"dtype": "float32", "shape": [frame_count, 3, 3]},
                    "camera_to_world": {
                        "dtype": "float32",
                        "shape": [frame_count, 4, 4],
                    },
                    "timestamps_ns": {"axes": "T", "dtype": "int64", "shape": [frame_count]},
                },
            },
        )

    def _canonicalize_result(  # noqa: C901
        self,
        result: ViPEFrameResult,
        *,
        expected_index: int,
        height: int,
        width: int,
    ) -> tuple[
        npt.NDArray[np.float16],
        npt.NDArray[np.bool_],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        try:
            raw_frame_idx = int(result.raw_frame_idx)
        except (AttributeError, TypeError, ValueError) as error:
            msg = f"ViPE frame {expected_index} has an invalid raw frame index"
            raise ValueError(msg) from error
        if raw_frame_idx != expected_index:
            msg = f"ViPE changed or dropped frame indices: expected {expected_index}, got {raw_frame_idx}"
            raise ValueError(msg)

        depth_f32 = np.asarray(result.metric_depth, dtype=np.float32)
        if depth_f32.shape != (height, width):
            msg = f"ViPE frame {expected_index} depth shape is {depth_f32.shape}; expected {(height, width)}"
            raise ValueError(msg)
        valid = np.isfinite(depth_f32) & (depth_f32 > _MIN_VALID_DEPTH_METERS) & (depth_f32 < _MAX_VALID_DEPTH_METERS)
        valid_fraction = float(valid.mean())
        if valid_fraction < self._min_valid_fraction:
            msg = (
                f"ViPE frame {expected_index} valid depth fraction {valid_fraction:.4f} "
                f"is below the required {self._min_valid_fraction:.4f}"
            )
            raise ValueError(msg)
        depth = np.ascontiguousarray(np.where(valid, depth_f32, 0.0), dtype=np.float16)
        valid = np.ascontiguousarray(valid, dtype=np.bool_)

        raw_intrinsics = np.asarray(result.intrinsics, dtype=np.float32).reshape(-1)
        if raw_intrinsics.size < _FRAME_ARRAY_NDIM:
            msg = f"ViPE frame {expected_index} intrinsics require [fx,fy,cx,cy]"
            raise ValueError(msg)
        fx, fy, cx, cy = raw_intrinsics[:4]
        camera_k = np.asarray(
            ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        )
        camera_to_world = np.ascontiguousarray(result.camera_to_world, dtype=np.float32)
        if camera_to_world.shape != (4, 4):
            msg = f"ViPE frame {expected_index} pose shape is {camera_to_world.shape}; expected (4, 4)"
            raise ValueError(msg)
        if not np.isfinite(camera_k).all() or not np.isfinite(camera_to_world).all():
            msg = f"ViPE frame {expected_index} contains non-finite camera values"
            raise ValueError(msg)
        if fx <= 0 or fy <= 0:
            msg = f"ViPE frame {expected_index} contains non-positive focal lengths"
            raise ValueError(msg)
        expected_last_row = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        if not np.allclose(camera_to_world[3], expected_last_row, atol=1.0e-4, rtol=0.0):
            msg = f"ViPE frame {expected_index} pose is not a homogeneous transform"
            raise ValueError(msg)
        try:
            inverse = np.linalg.inv(camera_to_world)
        except np.linalg.LinAlgError as error:
            msg = f"ViPE frame {expected_index} pose is not invertible"
            raise ValueError(msg) from error
        if not np.allclose(
            inverse @ camera_to_world,
            np.eye(4, dtype=np.float32),
            atol=2.0e-3,
            rtol=0.0,
        ):
            msg = f"ViPE frame {expected_index} pose inverse is inconsistent"
            raise ValueError(msg)
        return depth, valid, camera_k, camera_to_world

    def _write_chunk(  # noqa: PLR0913
        self,
        clip: Clip,
        *,
        frame_start: int,
        frame_stop: int,
        depth: list[npt.NDArray[np.float16]],
        valid: list[npt.NDArray[np.bool_]],
        camera_k: list[npt.NDArray[np.float32]],
        camera_to_world: list[npt.NDArray[np.float32]],
        timestamps_ns: npt.NDArray[np.int64],
    ) -> None:
        writer = self._require_writer()
        with writer.open_chunk(clip.uuid, frame_start, frame_stop) as chunk_path:
            np.savez(
                chunk_path,
                depth=np.stack(depth),
                valid=np.stack(valid),
                K=np.stack(camera_k),
                camera_to_world=np.stack(camera_to_world),
                timestamps_ns=np.ascontiguousarray(timestamps_ns, dtype=np.int64),
            )

    def _require_writer(self) -> TemporalWriter:
        if self._writer is None:
            msg = "ViPEStage.stage_setup() must be called before process_data()"
            raise RuntimeError(msg)
        return self._writer

    def _reuse_completed(self, task: SplitPipeTask, clip: Clip) -> bool:
        if self._reader is None or not self._reader.is_complete(clip.uuid):
            return False
        self._set_annotation_reference(task, self._reader.metadata_uri(clip.uuid))
        return True

    @staticmethod
    def _set_annotation_reference(task: SplitPipeTask, metadata_uri: str) -> None:
        dataset_metadata = getattr(task, "dataset_metadata", None)
        if isinstance(dataset_metadata, dict):
            dataset_metadata["vipe_annotation"] = {
                "format": "npz-temporal-v1",
                "metadata_uri": metadata_uri,
                "producer": "vipe-dav3",
            }


def decode_local_vipe_clip(  # noqa: PLR0913
    source_path: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    rotation_degrees_clockwise: int,
    min_frames: int,
    max_frames: int,
) -> DecodedViPEClip:
    """Decode a native-rate span, falling back to sequential PyAV for weak indexes."""
    try:
        return _decode_with_camera_sensor(
            source_path,
            span,
            stream_index=stream_index,
            rotation_degrees_clockwise=rotation_degrees_clockwise,
            min_frames=min_frames,
            max_frames=max_frames,
        )
    except _IndexedDecodeUnavailableError as sensor_error:
        try:
            return _decode_with_pyav(
                source_path,
                span,
                stream_index=stream_index,
                rotation_degrees_clockwise=rotation_degrees_clockwise,
                min_frames=min_frames,
                max_frames=max_frames,
            )
        except Exception as fallback_error:
            msg = (
                f"both CameraSensor and sequential PyAV failed to decode {source_path}: "
                f"CameraSensor={sensor_error}; PyAV={fallback_error}"
            )
            raise RuntimeError(msg) from fallback_error


def _decode_with_camera_sensor(  # noqa: PLR0913
    source_path: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    rotation_degrees_clockwise: int,
    min_frames: int,
    max_frames: int,
) -> DecodedViPEClip:
    """Use indexed CameraSensor decode when the container supports it."""
    try:
        sensor = CameraSensor(source_path, stream_idx=stream_index)
    except Exception as error:
        msg = f"CameraSensor cannot index {source_path}"
        raise _IndexedDecodeUnavailableError(msg) from error
    selected_timestamps, exclusive_end_ns, source_span, fps = _camera_sensor_selection(
        sensor,
        span,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    frames, decoded_timestamps = _sample_camera_sensor(
        sensor,
        selected_timestamps=selected_timestamps,
        exclusive_end_ns=exclusive_end_ns,
        source_path=source_path,
    )
    frames = _rotate_frames(frames, rotation_degrees_clockwise)
    timestamps = np.ascontiguousarray(decoded_timestamps - sensor.start_ns, dtype=np.int64)
    return DecodedViPEClip(
        frames=frames,
        timestamps_ns=timestamps,
        fps=fps,
        source_span=source_span,
        decoder_backend="camera_sensor",
    )


def _camera_sensor_selection(
    sensor: CameraSensor,
    span: tuple[float, float] | None,
    *,
    min_frames: int,
    max_frames: int,
) -> tuple[npt.NDArray[np.int64], int, tuple[float, float], float]:
    """Select an exact bounded native timeline before indexed decode."""
    fps = float(sensor.video_metadata.avg_frame_rate)
    frame_period_ns = _nominal_frame_period_ns(fps)
    full_exclusive_end_ns = sensor.end_ns + frame_period_ns
    if span is None:
        requested_start_ns = sensor.start_ns
        requested_stop_ns = full_exclusive_end_ns
        source_span = (0.0, (full_exclusive_end_ns - sensor.start_ns) / _NANOSECONDS_PER_SECOND)
    else:
        requested_start_ns = sensor.start_ns + round(span[0] * _NANOSECONDS_PER_SECOND)
        requested_stop_ns = sensor.start_ns + round(span[1] * _NANOSECONDS_PER_SECOND)
        source_span = span
    start_ns = max(sensor.start_ns, requested_start_ns)
    exclusive_end_ns = min(full_exclusive_end_ns, requested_stop_ns)
    if exclusive_end_ns <= start_ns:
        msg = f"clip span {span} does not intersect the source video"
        raise ValueError(msg)

    canonical_timestamps = sensor.timestamps_ns
    frame_start = int(np.searchsorted(canonical_timestamps, start_ns, side="left"))
    frame_stop = int(np.searchsorted(canonical_timestamps, exclusive_end_ns, side="left"))
    selected_timestamps = np.ascontiguousarray(canonical_timestamps[frame_start:frame_stop], dtype=np.int64)
    frame_count = len(selected_timestamps)
    if frame_count < min_frames:
        msg = f"ViPE requires at least {min_frames} frames, got {frame_count} in span {span}"
        raise ValueError(msg)
    if frame_count > max_frames:
        msg = (
            "ViPE requires full-clip SLAM and this adapter does not claim streaming input support: "
            f"span {span} contains {frame_count} frames, max_frames={max_frames}; split the span upstream"
        )
        raise ValueError(msg)
    return selected_timestamps, exclusive_end_ns, source_span, fps


def _sample_camera_sensor(
    sensor: CameraSensor,
    *,
    selected_timestamps: npt.NDArray[np.int64],
    exclusive_end_ns: int,
    source_path: Path,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Decode an already validated exact native timeline."""
    grid_start_ns = int(selected_timestamps[0])
    duration_ns = exclusive_end_ns - grid_start_ns
    grid = SamplingGrid(
        start_ns=grid_start_ns,
        exclusive_end_ns=exclusive_end_ns,
        timestamps_ns=selected_timestamps,
        stride_ns=duration_ns,
        duration_ns=duration_ns,
    )
    try:
        batches = sensor.sample(SamplingSpec(grid=grid))
        try:
            batch = next(batches)
        except StopIteration as error:
            msg = "CameraSensor did not return the requested ViPE clip"
            raise RuntimeError(msg) from error
        try:
            next(batches)
        except StopIteration:
            pass
        else:
            msg = "CameraSensor returned more than one batch for a single ViPE clip"
            raise RuntimeError(msg)
    except Exception as error:
        msg = f"CameraSensor cannot decode {source_path}"
        raise _IndexedDecodeUnavailableError(msg) from error

    decoded_timestamps = np.ascontiguousarray(batch.sensor_timestamps_ns, dtype=np.int64)
    if not np.array_equal(decoded_timestamps, selected_timestamps):
        msg = "CameraSensor decoded timestamps do not match the selected native frame timeline"
        raise _IndexedDecodeUnavailableError(msg)
    frames = np.ascontiguousarray(batch.frames, dtype=np.uint8)
    return frames, decoded_timestamps


def _decode_with_pyav(  # noqa: PLR0913
    source_path: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    rotation_degrees_clockwise: int,
    min_frames: int,
    max_frames: int,
) -> DecodedViPEClip:
    """Sequentially decode native frames without relying on a seekable header index."""
    with av.open(str(source_path)) as container:
        try:
            video_stream = container.streams.video[stream_index]
        except IndexError as error:
            msg = f"video stream_index={stream_index} does not exist"
            raise ValueError(msg) from error
        if video_stream.time_base is None:
            msg = f"video stream_index={stream_index} has no time base"
            raise ValueError(msg)
        average_rate = float(video_stream.average_rate) if video_stream.average_rate is not None else None
        frames, timestamps, last_source_timestamp_ns = _collect_pyav_frames(
            container,
            video_stream,
            span=span,
            max_frames=max_frames,
        )

    frame_count = len(frames)
    if frame_count < min_frames:
        msg = f"ViPE requires at least {min_frames} frames, got {frame_count} in span {span}"
        raise ValueError(msg)
    timestamp_array = np.asarray(timestamps, dtype=np.int64)
    fps = _resolve_pyav_fps(average_rate, timestamp_array)
    if span is None:
        assert last_source_timestamp_ns is not None
        source_span = (0.0, (last_source_timestamp_ns + _nominal_frame_period_ns(fps)) / _NANOSECONDS_PER_SECOND)
    else:
        source_span = span
    try:
        frame_array = np.ascontiguousarray(np.stack(frames), dtype=np.uint8)
    except ValueError as error:
        msg = "PyAV decoded frames with inconsistent spatial shapes"
        raise ValueError(msg) from error
    frame_array = _rotate_frames(frame_array, rotation_degrees_clockwise)
    return DecodedViPEClip(
        frames=frame_array,
        timestamps_ns=np.ascontiguousarray(timestamp_array),
        fps=fps,
        source_span=source_span,
        decoder_backend="pyav_sequential",
    )


def _collect_pyav_frames(
    container: av.container.InputContainer,
    video_stream: av.video.stream.VideoStream,
    *,
    span: tuple[float, float] | None,
    max_frames: int,
) -> tuple[list[npt.NDArray[np.uint8]], list[int], int | None]:
    """Collect only selected RGB frames while scanning a weakly indexed source."""
    frames: list[npt.NDArray[np.uint8]] = []
    timestamps: list[int] = []
    origin_ns: int | None = None
    last_source_timestamp_ns: int | None = None
    span_bounds_ns = (
        None
        if span is None
        else (
            round(span[0] * _NANOSECONDS_PER_SECOND),
            round(span[1] * _NANOSECONDS_PER_SECOND),
        )
    )
    assert video_stream.time_base is not None
    for frame in container.decode(video_stream):
        if frame.pts is None:
            continue
        source_timestamp_ns = round(Fraction(frame.pts) * Fraction(video_stream.time_base) * _NANOSECONDS_PER_SECOND)
        if origin_ns is None:
            origin_ns = source_timestamp_ns
        relative_timestamp_ns = source_timestamp_ns - origin_ns
        last_source_timestamp_ns = relative_timestamp_ns
        if span_bounds_ns is not None and relative_timestamp_ns < span_bounds_ns[0]:
            continue
        if span_bounds_ns is not None and relative_timestamp_ns >= span_bounds_ns[1]:
            break
        if timestamps and relative_timestamp_ns <= timestamps[-1]:
            msg = "PyAV returned non-increasing presentation timestamps"
            raise ValueError(msg)
        if len(frames) >= max_frames:
            msg = (
                "ViPE requires full-clip SLAM and this adapter does not claim streaming input support: "
                f"more than max_frames={max_frames} native frames were selected; split the span upstream"
            )
            raise ValueError(msg)
        frames.append(np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8))
        timestamps.append(relative_timestamp_ns)
    return frames, timestamps, last_source_timestamp_ns


def _rotate_frames(
    frames: npt.NDArray[np.uint8],
    rotation_degrees_clockwise: int,
) -> npt.NDArray[np.uint8]:
    if not rotation_degrees_clockwise:
        return np.ascontiguousarray(frames)
    return np.rot90(
        frames,
        k=-(rotation_degrees_clockwise // 90),
        axes=(1, 2),
    ).copy()


def _resolve_pyav_fps(
    average_rate: float | None,
    timestamps: npt.NDArray[np.int64],
) -> float:
    if average_rate is not None and math.isfinite(average_rate) and average_rate > 0:
        return average_rate
    elapsed_ns = int(timestamps[-1]) - int(timestamps[0])
    if elapsed_ns <= 0:
        msg = "cannot derive source fps from PyAV timestamps"
        raise ValueError(msg)
    return (len(timestamps) - 1) * _NANOSECONDS_PER_SECOND / elapsed_ns


def _nominal_frame_period_ns(fps: float) -> int:
    if not math.isfinite(fps) or fps <= 0:
        msg = "source fps must be finite and positive"
        raise ValueError(msg)
    return max(1, round(_NANOSECONDS_PER_SECOND / fps))


def _positive_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise ValueError(msg)
    return value
