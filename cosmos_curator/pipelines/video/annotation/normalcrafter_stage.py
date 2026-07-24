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

from collections.abc import Mapping
from contextlib import AbstractContextManager
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
    TemporalAnnotationReader,
    TemporalAnnotationWriter,
)
from cosmos_curator.pipelines.video.annotation.data_model import (
    full_source_clip_uuid,
    make_full_source_clip,
    normalize_span,
    resolve_source_clip_request,
    source_path_string,
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

_PRODUCER_RELEASE = "normalcrafter-grid-v1"

DecodedNormalCrafterClip = DecodedAnnotationClip
NormalCrafterClipDecoder = AnnotationClipDecoder


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
        annotation_grid: AnnotationGrid = DEFAULT_ANNOTATION_GRID,
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
        self._annotation_grid = annotation_grid
        self._inference_model = inference_model
        self._decoder = decoder or decode_normalcrafter_clip
        self._writer = writer
        self._reader: TemporalAnnotationReader | None = None

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
        self._reader = TemporalAnnotationReader(
            self._output_path,
            profile_name=self._profile_name,
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
        if self._reuse_completed(
            task,
            clip,
            source=source,
            span=request.span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
        ):
            return tasks
        decoded = self._decoder(
            source,
            request.span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
            grid=self._annotation_grid,
            min_frames=NORMALCRAFTER_WINDOW_SIZE,
            max_frames=self._inference_model.max_frames,
        )
        frames, timestamps_ns, source_timestamps_ns, decoded_span = validate_decoded_annotation_clip(
            decoded,
            grid=self._annotation_grid,
            min_frames=NORMALCRAFTER_WINDOW_SIZE,
            max_frames=self._inference_model.max_frames,
            consumer_name="NormalCrafter",
        )
        if request.span is not None and decoded_span != request.span:
            message = f"NormalCrafter decoder returned source_span={decoded_span}, expected {request.span}"
            raise ValueError(message)
        if decoded.raster.rotation_degrees_clockwise != request.rotation_degrees_clockwise:
            message = (
                "NormalCrafter decoder returned rotation_degrees_clockwise="
                f"{decoded.raster.rotation_degrees_clockwise}, expected {request.rotation_degrees_clockwise}"
            )
            raise ValueError(message)
        span = decoded_span
        if clip is None:
            clip = make_full_source_clip(source, span, request.stream_index)
            task.video.clips.append(clip)
            task.video.num_total_clips = max(task.video.num_total_clips, 1)
        if self._reuse_completed(
            task,
            clip,
            source=source,
            span=span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
        ):
            return tasks

        metadata_uri = self._infer_and_write(
            clip,
            source=source,
            span=span,
            stream_index=request.stream_index,
            rotation_degrees_clockwise=request.rotation_degrees_clockwise,
            decoder_backend=decoded.decoder_backend,
            frames=frames,
            timestamps_ns=timestamps_ns,
            source_timestamps_ns=source_timestamps_ns,
            raster=decoded.raster,
        )
        self._set_annotation_reference(task, metadata_uri)
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
        source_timestamps_ns: npt.NDArray[np.int64],
        raster: RasterTransform,
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
                        source_timestamps_ns=np.ascontiguousarray(
                            source_timestamps_ns[chunk.frame_start : frame_stop],
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
                    "sample_fps": self._annotation_grid.sample_fps,
                    "timestamp_array": "timestamps_ns",
                    "source_timestamp_array": "source_timestamps_ns",
                    "timestamp_unit": "nanosecond",
                    "timestamp_origin": "source_stream_start",
                },
                "grid": self._annotation_grid.metadata(raster),
                "recipe": {
                    "window_size": NORMALCRAFTER_WINDOW_SIZE,
                    "window_stride": NORMALCRAFTER_WINDOW_STRIDE,
                    "vae_chunk_size": NORMALCRAFTER_VAE_CHUNK_SIZE,
                    "padding": f"centered_white_to_multiple_{NORMALCRAFTER_PADDING_MULTIPLE}",
                    "attention_backend": "sdpa",
                    "conditioning_fps": NORMALCRAFTER_CONDITIONING_FPS,
                    "axis_transform_from_raw": [-1, 1, 1],
                    "normalization": "unit_length",
                    "normal_coordinate_space": "annotation_grid_camera_opencv",
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
                    "source_timestamps_ns": {
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

    def _reuse_completed(  # noqa: PLR0913
        self,
        task: SplitPipeTask,
        clip: Clip | None,
        *,
        source: Path,
        span: tuple[float, float] | None,
        stream_index: int,
        rotation_degrees_clockwise: int,
    ) -> bool:
        if self._reader is None:
            return False
        clip_uuid = clip.uuid if clip is not None else full_source_clip_uuid(source, stream_index)
        document = self._reader.read_metadata_if_complete(clip_uuid)
        if document is None:
            return False
        metadata = document.get("metadata")
        frame_count = document.get("frame_count")
        expected_producer = {
            "id": "normalcrafter",
            "release": _PRODUCER_RELEASE,
            "model_id": self._inference_model.model_id,
        }
        expected_span = None if span is None else [span[0], span[1]]
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
                source=source,
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
            )
        ):
            message = (
                f"completed NormalCrafter annotation {clip_uuid} uses a different source, producer, or annotation "
                "grid, or an incompatible output contract; use a new output_path instead of overwriting it"
            )
            raise ValueError(message)
        if clip is None:
            assert completed_span is not None
            clip = make_full_source_clip(source, completed_span, stream_index)
            task.video.clips.append(clip)
            task.video.num_total_clips = max(task.video.num_total_clips, 1)
        self._set_annotation_reference(task, self._reader.metadata_uri(clip.uuid))
        return True

    @staticmethod
    def _set_annotation_reference(task: SplitPipeTask, metadata_uri: str) -> None:
        dataset_metadata = getattr(task, "dataset_metadata", None)
        if isinstance(dataset_metadata, dict):
            dataset_metadata["normal_annotation"] = {
                "format": "npz-temporal-v1",
                "metadata_uri": metadata_uri,
                "producer_release": _PRODUCER_RELEASE,
            }


def decode_normalcrafter_clip(  # noqa: PLR0913
    source: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    rotation_degrees_clockwise: int,
    grid: AnnotationGrid,
    min_frames: int,
    max_frames: int,
) -> DecodedNormalCrafterClip:
    """Decode NormalCrafter input on the shared, low-resolution annotation grid."""
    return decode_annotation_clip(
        source,
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
    source: Path,
    span: list[float] | None,
    stream_index: int,
    rotation_degrees_clockwise: int,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("path") == source_path_string(source)
        and (span is None or value.get("span_seconds") == span)
        and value.get("stream_index") == stream_index
        and value.get("rotation_degrees_clockwise") == rotation_degrees_clockwise
    )


def _output_contract_matches(
    metadata: Mapping[object, object],
    *,
    frame_count: object,
    grid: AnnotationGrid,
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
        and arrays.get("normal")
        == {
            "axes": "THWC",
            "dtype": "float16",
            "shape": [frame_count, grid.height, grid.width, 3],
        }
        and arrays.get("valid")
        == {
            "axes": "THW",
            "dtype": "bool",
            "shape": [frame_count, grid.height, grid.width],
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
        and recipe.get("axis_transform_from_raw") == [-1, 1, 1]
        and recipe.get("normalization") == "unit_length"
        and recipe.get("normal_coordinate_space") == "annotation_grid_camera_opencv"
    )


def _source_span_from_metadata(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return normalize_span(value.get("span_seconds"))
    except ValueError:
        return None


def _validated_source(source: object) -> Path:
    if isinstance(source, Path):
        resolved = source.expanduser().resolve(strict=True)
        if not resolved.is_file():
            message = f"NormalCrafter source is not a file: {resolved}"
            raise ValueError(message)
        return resolved
    message = f"NormalCrafter source must be a local pathlib.Path, got {type(source).__name__}"
    raise TypeError(message)


__all__ = [
    "DecodedNormalCrafterClip",
    "NormalCrafterClipDecoder",
    "NormalCrafterStage",
    "decode_normalcrafter_clip",
]
