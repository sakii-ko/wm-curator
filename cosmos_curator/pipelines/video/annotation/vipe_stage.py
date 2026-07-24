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
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, PipelineTask
from cosmos_curator.models.vipe import ViPEFrameResult
from cosmos_curator.pipelines.video.annotation.artifact_writer import (
    TemporalAnnotationReader,
    TemporalAnnotationWriter,
)
from cosmos_curator.pipelines.video.annotation.data_model import (
    full_source_clip_uuid,
    make_full_source_clip,
    normalize_span,
    resolve_source_clip_request,
)
from cosmos_curator.pipelines.video.annotation.decode_grid import (
    DEFAULT_ANNOTATION_GRID,
    AnnotationClipDecoder,
    AnnotationGrid,
    DecodedAnnotationClip,
    RasterTransform,
    annotation_grid_configuration_matches,
    annotation_grid_frame_count,
    decode_annotation_clip,
    validate_decoded_annotation_clip,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask

_FRAME_ARRAY_NDIM = 4
_MIN_VALID_DEPTH_METERS = 1.0e-4
_MAX_VALID_DEPTH_METERS = 1.0e4
_PRODUCER_RELEASE = "vipe-dav3-grid-v1"
VIPE_MIN_FRAMES = 8

DecodedViPEClip = DecodedAnnotationClip
ViPEClipDecoder = AnnotationClipDecoder


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
        annotation_grid: AnnotationGrid = DEFAULT_ANNOTATION_GRID,
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
        self._annotation_grid = annotation_grid
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
        if self._reuse_completed(
            task,
            clip,
            source_path=source_path,
            span=request.span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
        ):
            return tasks
        decoded = self._decoder(
            source_path,
            request.span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
            grid=self._annotation_grid,
            min_frames=VIPE_MIN_FRAMES,
            max_frames=self._max_frames,
        )
        frames, timestamps_ns, source_timestamps_ns, decoded_span = self._validate_decoded(decoded)
        if request.span is not None and decoded_span != request.span:
            msg = f"ViPE decoder returned source_span={decoded_span}, expected {request.span}"
            raise ValueError(msg)
        if decoded.raster.rotation_degrees_clockwise != request.rotation_degrees_clockwise:
            msg = (
                "ViPE decoder returned rotation_degrees_clockwise="
                f"{decoded.raster.rotation_degrees_clockwise}, expected {request.rotation_degrees_clockwise}"
            )
            raise ValueError(msg)
        span = decoded_span
        if clip is None:
            clip = make_full_source_clip(source_path, span, request.stream_index)
            task.video.clips.append(clip)
            task.video.num_total_clips = max(task.video.num_total_clips, 1)
        if self._reuse_completed(
            task,
            clip,
            source_path=source_path,
            span=span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
        ):
            return tasks
        metadata_uri = self._infer_and_write(
            clip,
            source_path=source_path,
            span=span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
            frames=frames,
            timestamps_ns=timestamps_ns,
            source_timestamps_ns=source_timestamps_ns,
            raster=decoded.raster,
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
    ) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        tuple[float, float],
    ]:
        return validate_decoded_annotation_clip(
            decoded,
            grid=self._annotation_grid,
            min_frames=VIPE_MIN_FRAMES,
            max_frames=self._max_frames,
            consumer_name="ViPE",
        )

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
        source_timestamps_ns: npt.NDArray[np.int64],
        raster: RasterTransform,
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

        results = iter(
            self._inference_model.infer(
                frames,
                name=str(clip.uuid),
                fps=self._annotation_grid.sample_fps,
            )
        )
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
                        source_timestamps_ns=source_timestamps_ns[chunk_start:result_count],
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
                source_timestamps_ns=source_timestamps_ns[chunk_start:result_count],
            )

        valid_fraction = valid_count / (frame_count * height * width)
        writer = self._require_writer()
        return writer.complete_clip(
            clip.uuid,
            frame_count=frame_count,
            chunk_frames=self._chunk_frames,
            metadata={
                "producer": {
                    "id": "vipe",
                    "pipeline": "dav3",
                    "release": _PRODUCER_RELEASE,
                },
                "source": {
                    "path": str(source_path),
                    "span_seconds": [span[0], span[1]],
                    "stream_index": stream_index,
                    "rotation_degrees_clockwise": rotation_degrees_clockwise,
                    "decoder_backend": decoder_backend,
                },
                "alignment": {
                    "sample_fps": self._annotation_grid.sample_fps,
                    "timestamp_array": "timestamps_ns",
                    "source_timestamp_array": "source_timestamps_ns",
                    "timestamp_unit": "nanosecond",
                    "timestamp_origin": "source_stream_start",
                },
                "grid": self._annotation_grid.metadata(raster),
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
                    "K_coordinate_space": "annotation_grid",
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
                    "source_timestamps_ns": {
                        "axes": "T",
                        "dtype": "int64",
                        "shape": [frame_count],
                    },
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
        source_timestamps_ns: npt.NDArray[np.int64],
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
                source_timestamps_ns=np.ascontiguousarray(
                    source_timestamps_ns,
                    dtype=np.int64,
                ),
            )

    def _require_writer(self) -> TemporalWriter:
        if self._writer is None:
            msg = "ViPEStage.stage_setup() must be called before process_data()"
            raise RuntimeError(msg)
        return self._writer

    def _reuse_completed(  # noqa: PLR0913
        self,
        task: SplitPipeTask,
        clip: Clip | None,
        *,
        source_path: Path,
        span: tuple[float, float] | None,
        stream_index: int,
        rotation_degrees_clockwise: int,
    ) -> bool:
        if self._reader is None:
            return False
        clip_uuid = clip.uuid if clip is not None else full_source_clip_uuid(source_path, stream_index)
        document = self._reader.read_metadata_if_complete(clip_uuid)
        if document is None:
            return False
        metadata = document.get("metadata")
        frame_count = document.get("frame_count")
        expected_span = None if span is None else [span[0], span[1]]
        expected_producer = {
            "id": "vipe",
            "pipeline": "dav3",
            "release": _PRODUCER_RELEASE,
        }
        completed_span = _source_span_from_metadata(metadata.get("source")) if isinstance(metadata, Mapping) else None
        frame_count_matches = (
            isinstance(frame_count, int)
            and not isinstance(frame_count, bool)
            and completed_span is not None
            and annotation_grid_frame_count(completed_span, self._annotation_grid) == frame_count
        )
        full_source_span_matches = clip is not None or (completed_span is not None and completed_span[0] == 0.0)
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("producer") != expected_producer
            or not _source_metadata_matches(
                metadata.get("source"),
                source_path=source_path,
                span=expected_span,
                stream_index=stream_index,
                rotation_degrees_clockwise=rotation_degrees_clockwise,
            )
            or not annotation_grid_configuration_matches(
                metadata.get("grid"),
                self._annotation_grid,
            )
            or not frame_count_matches
            or not full_source_span_matches
            or not _output_contract_matches(
                metadata,
                frame_count=frame_count,
                grid=self._annotation_grid,
                min_valid_fraction=self._min_valid_fraction,
            )
        ):
            msg = (
                f"completed ViPE annotation {clip_uuid} uses a different source, producer, or annotation grid, "
                "or an incompatible output contract; use a new output_path instead of overwriting it"
            )
            raise ValueError(msg)
        if clip is None:
            assert completed_span is not None
            clip = make_full_source_clip(source_path, completed_span, stream_index)
            task.video.clips.append(clip)
            task.video.num_total_clips = max(task.video.num_total_clips, 1)
        self._set_annotation_reference(task, self._reader.metadata_uri(clip.uuid))
        return True

    @staticmethod
    def _set_annotation_reference(task: SplitPipeTask, metadata_uri: str) -> None:
        dataset_metadata = getattr(task, "dataset_metadata", None)
        if isinstance(dataset_metadata, dict):
            dataset_metadata["vipe_annotation"] = {
                "format": "npz-temporal-v1",
                "metadata_uri": metadata_uri,
                "producer_release": _PRODUCER_RELEASE,
            }


def decode_local_vipe_clip(  # noqa: PLR0913
    source_path: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    rotation_degrees_clockwise: int,
    grid: AnnotationGrid,
    min_frames: int,
    max_frames: int,
) -> DecodedViPEClip:
    """Decode ViPE input on the shared, low-resolution annotation grid."""
    return decode_annotation_clip(
        source_path,
        span,
        stream_index=stream_index,
        rotation_degrees_clockwise=rotation_degrees_clockwise,
        grid=grid,
        min_frames=min_frames,
        max_frames=max_frames,
    )


def _source_metadata_matches(
    value: object,
    *,
    source_path: Path,
    span: list[float] | None,
    stream_index: int,
    rotation_degrees_clockwise: int,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("path") == str(source_path)
        and (span is None or value.get("span_seconds") == span)
        and value.get("stream_index") == stream_index
        and value.get("rotation_degrees_clockwise") == rotation_degrees_clockwise
    )


def _output_contract_matches(
    metadata: Mapping[object, object],
    *,
    frame_count: object,
    grid: AnnotationGrid,
    min_valid_fraction: float,
) -> bool:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        return False
    alignment = metadata.get("alignment")
    arrays = metadata.get("arrays")
    recipe = metadata.get("recipe")
    if not isinstance(alignment, Mapping) or not isinstance(arrays, Mapping) or not isinstance(recipe, Mapping):
        return False
    return (
        alignment.get("timestamp_array") == "timestamps_ns"
        and alignment.get("source_timestamp_array") == "source_timestamps_ns"
        and alignment.get("sample_fps") == grid.sample_fps
        and alignment.get("timestamp_unit") == "nanosecond"
        and alignment.get("timestamp_origin") == "source_stream_start"
        and arrays.get("depth")
        == {
            "axes": "THW",
            "dtype": "float16",
            "shape": [frame_count, grid.height, grid.width],
        }
        and arrays.get("valid")
        == {
            "axes": "THW",
            "dtype": "bool",
            "shape": [frame_count, grid.height, grid.width],
        }
        and arrays.get("K") == {"dtype": "float32", "shape": [frame_count, 3, 3]}
        and arrays.get("camera_to_world")
        == {
            "dtype": "float32",
            "shape": [frame_count, 4, 4],
        }
        and arrays.get("timestamps_ns")
        == {
            "axes": "T",
            "dtype": "int64",
            "shape": [frame_count],
        }
        and arrays.get("source_timestamps_ns")
        == {
            "axes": "T",
            "dtype": "int64",
            "shape": [frame_count],
        }
        and recipe.get("depth_representation") == "z_depth"
        and recipe.get("depth_scale") == "metric"
        and recipe.get("K_coordinate_space") == "annotation_grid"
        and recipe.get("min_valid_fraction_per_frame") == min_valid_fraction
    )


def _source_span_from_metadata(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return normalize_span(value.get("span_seconds"))
    except ValueError:
        return None


def _positive_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise ValueError(msg)
    return value
