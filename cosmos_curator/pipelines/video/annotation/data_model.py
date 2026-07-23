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
"""Small additions to the existing split-pipeline task model for annotation inputs."""

import math
import pathlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import attrs

from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video

type SourcePath = pathlib.Path | StoragePrefix
type TimeSpan = tuple[float, float]

_SPAN_BOUND_COUNT = 2


def normalize_stream_index(value: object) -> int | None:
    """Validate an optional zero-based media stream index."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"stream_index must be a non-negative integer, got {value!r}"
        raise ValueError(msg)
    return value


def normalize_rotation_degrees(value: object) -> int | None:
    """Normalize an optional clockwise, right-angle rotation to ``[0, 360)``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value % 90:
        msg = f"rotation_degrees_clockwise must be a multiple of 90, got {value!r}"
        raise ValueError(msg)
    return value % 360


def normalize_span(value: object) -> TimeSpan | None:
    """Validate an optional half-open source time span in seconds."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != _SPAN_BOUND_COUNT:
        msg = f"span must contain exactly [start_seconds, end_seconds], got {value!r}"
        raise ValueError(msg)
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        msg = f"span values must be numbers, got {value!r}"
        raise ValueError(msg)

    start, end = float(value[0]), float(value[1])
    if not math.isfinite(start) or not math.isfinite(end):
        msg = f"span values must be finite, got {value!r}"
        raise ValueError(msg)
    if start < 0 or end <= start:
        msg = f"span must satisfy 0 <= start < end, got {(start, end)!r}"
        raise ValueError(msg)
    return start, end


def normalize_relative_path(value: str) -> str:
    """Return a portable, non-empty relative POSIX path."""
    if not isinstance(value, str) or not value.strip():
        msg = "relative_path must be a non-empty string"
        raise ValueError(msg)
    candidate = pathlib.PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        msg = f"relative_path must not be absolute or contain '..', got {value!r}"
        raise ValueError(msg)
    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        msg = "relative_path must identify a file"
        raise ValueError(msg)
    return normalized


def _copy_dataset_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"dataset_metadata must be a mapping, got {type(value).__name__}"
        raise TypeError(msg)
    if any(not isinstance(key, str) for key in value):
        msg = "dataset_metadata keys must be strings"
        raise TypeError(msg)
    return dict(value)


def source_path_string(source: SourcePath) -> str:
    """Return the local POSIX path or remote object URI for a source."""
    if isinstance(source, StoragePrefix):
        return source.path
    return source.as_posix()


@dataclass(frozen=True, slots=True)
class SourceClipRequest:
    """Source, span, and decode hints carried by one split-pipeline task."""

    source: SourcePath
    clip: Clip | None
    span: TimeSpan | None
    stream_index: int
    rotation_degrees_clockwise: int


def resolve_source_clip_request(task: SplitPipeTask) -> SourceClipRequest:
    """Resolve the source-side fields shared by temporal annotation stages."""
    if len(task.videos) != 1:
        msg = f"source-backed annotation requires one video per task, got {len(task.videos)}"
        raise ValueError(msg)

    source = task.video.input_video
    if len(task.video.clips) > 1:
        msg = f"source-backed annotation supports at most one Clip.span per task, got {len(task.video.clips)} clips"
        raise ValueError(msg)

    clip = task.video.clips[0] if task.video.clips else None
    span = normalize_span(clip.span) if clip is not None else None
    stream_index = normalize_stream_index(getattr(task, "stream_index", None))
    rotation = normalize_rotation_degrees(getattr(task, "rotation_degrees_clockwise", None))
    return SourceClipRequest(
        source=source,
        clip=clip,
        span=span,
        stream_index=0 if stream_index is None else stream_index,
        rotation_degrees_clockwise=0 if rotation is None else rotation,
    )


def make_full_source_clip(source: SourcePath, span: TimeSpan, stream_index: int) -> Clip:
    """Create the stable clip identity used when a task denotes the whole source."""
    normalized_span = normalize_span(span)
    if normalized_span is None:
        msg = "full-source clip requires a span"
        raise ValueError(msg)
    normalized_stream_index = normalize_stream_index(stream_index)
    if normalized_stream_index is None:
        msg = "full-source clip requires a stream index"
        raise ValueError(msg)
    source_path = source_path_string(source)
    clip_uuid = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{source_path}#video-stream={normalized_stream_index}#full-source",
    )
    return Clip(uuid=clip_uuid, source_video=source_path, span=normalized_span)


def normalize_clip_uuid(value: object) -> uuid.UUID:
    """Validate and canonicalize a clip UUID."""
    if isinstance(value, uuid.UUID):
        return value
    if not isinstance(value, str):
        msg = f"clip_uuid must be a UUID or UUID string, got {type(value).__name__}"
        raise TypeError(msg)
    try:
        return uuid.UUID(value)
    except ValueError as error:
        msg = f"clip_uuid must be a valid UUID, got {value!r}"
        raise ValueError(msg) from error


def _make_source_span_uuid(source_path: str, stream_index: int | None, span: TimeSpan) -> uuid.UUID:
    stream = 0 if stream_index is None else stream_index
    start, end = span
    identity = f"{source_path}#video-stream={stream}#span={start.hex()}:{end.hex()}"
    return uuid.uuid5(uuid.NAMESPACE_URL, identity)


@attrs.define
class AnnotationTask(SplitPipeTask):
    """A ``SplitPipeTask`` carrying only source-side annotation hints.

    The media itself remains represented by the inherited ``Video`` and an
    optional requested span remains represented by its first ``Clip``. These
    fields describe decode choices that the existing data model cannot express.
    """

    stream_index: int | None = attrs.field(default=None, converter=normalize_stream_index)
    rotation_degrees_clockwise: int | None = attrs.field(default=None, converter=normalize_rotation_degrees)
    dataset_metadata: dict[str, Any] = attrs.field(factory=dict, converter=_copy_dataset_metadata)

    @property
    def input_span(self) -> TimeSpan | None:
        """Return the adapter-provided span, if the task contains exactly one clip."""
        if not self.video.clips:
            return None
        if len(self.video.clips) != 1:
            msg = "input_span is only defined while an annotation input has at most one clip"
            raise ValueError(msg)
        return self.video.clips[0].span


def make_annotation_task(  # noqa: PLR0913
    source: SourcePath,
    *,
    session_id: str,
    relative_path: str,
    stream_index: int | None = None,
    rotation_degrees_clockwise: int | None = None,
    span: TimeSpan | None = None,
    clip_uuid: str | uuid.UUID | None = None,
    dataset_metadata: Mapping[str, Any] | None = None,
) -> AnnotationTask:
    """Build an annotation task using the existing ``Video`` and ``Clip`` types."""
    if not isinstance(session_id, str) or not session_id.strip():
        msg = "session_id must be a non-empty string"
        raise ValueError(msg)

    normalized_relative_path = normalize_relative_path(relative_path)
    normalized_span = normalize_span(span)
    normalized_stream_index = normalize_stream_index(stream_index)
    normalized_rotation = normalize_rotation_degrees(rotation_degrees_clockwise)
    source_path = source_path_string(source)

    clips: list[Clip] = []
    if normalized_span is not None:
        normalized_clip_uuid = (
            _make_source_span_uuid(source_path, normalized_stream_index, normalized_span)
            if clip_uuid is None
            else normalize_clip_uuid(clip_uuid)
        )
        clips.append(
            Clip(
                uuid=normalized_clip_uuid,
                source_video=source_path,
                span=normalized_span,
            )
        )
    elif clip_uuid is not None:
        msg = "clip_uuid requires an explicit span"
        raise ValueError(msg)

    video = Video(
        input_video=source,
        relative_path=normalized_relative_path,
        clips=clips,
        num_total_clips=len(clips),
    )
    return AnnotationTask(
        session_id=session_id,
        video=video,
        stream_index=normalized_stream_index,
        rotation_degrees_clockwise=normalized_rotation,
        dataset_metadata={} if dataset_metadata is None else dataset_metadata,
    )
