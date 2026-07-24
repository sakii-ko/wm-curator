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

"""Read and write temporal annotation chunks without a catalog.

The caller owns the annotation namespace through ``output_path`` (for example,
``annotations/normals/normalcrafter-grid-v1``). Each chunk is published independently,
and the per-clip metadata JSON is published last as the logical completion record.
"""

import contextlib
import io
import json
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_utils import StorageWriter

_SCHEMA = "cosmos-curator.temporal-annotation/v1"
_CHUNK_PATH_TEMPLATE = "chunks/v1/{clip_uuid}/frames-{frame_start:09d}-{frame_stop:09d}.npz"
_METADATA_PATH_TEMPLATE = "metas/v1/{clip_uuid}.json"


@dataclass
class _ClipWriteState:
    published_ranges: set[tuple[int, int]] = field(default_factory=set)
    pending_ranges: set[tuple[int, int]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class TemporalAnnotationChunk:
    """One materialized temporal chunk returned by the reader."""

    frame_start: int
    frame_stop: int
    arrays: Mapping[str, npt.NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class _ArraySpec:
    dtype: np.dtype[np.generic]
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _AnnotationDocument:
    raw: dict[str, Any]
    frame_count: int
    chunk_frames: int
    arrays: Mapping[str, _ArraySpec]
    timestamp_array: str


class TemporalAnnotationReader:
    """Read completed annotations one NPZ chunk at a time.

    The per-clip JSON is the only completion marker. Chunk paths are derived
    from its frame count and chunk size, so no manifest or hash index is
    required.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        profile_name: str = "default",
    ) -> None:
        """Create a reader rooted at one annotation producer release."""
        output_path_str = str(output_path)
        if not output_path_str.strip():
            message = "output_path must be non-empty"
            raise ValueError(message)
        self._base_path = output_path_str.rstrip("/")
        self._client = storage_utils.get_storage_client(
            self._base_path,
            profile_name=profile_name,
        )

    def metadata_uri(self, clip_uuid: str | UUID) -> str:
        """Return the completion-record location for one clip."""
        canonical_uuid = _canonical_clip_uuid(clip_uuid)
        return _join_location(self._base_path, _metadata_sub_path(canonical_uuid))

    def is_complete(self, clip_uuid: str | UUID) -> bool:
        """Return whether a valid final metadata record exists."""
        return self.read_metadata_if_complete(clip_uuid) is not None

    def read_metadata_if_complete(
        self,
        clip_uuid: str | UUID,
    ) -> dict[str, Any] | None:
        """Read one valid completion record, or return ``None`` if absent."""
        canonical_uuid = _canonical_clip_uuid(clip_uuid)
        if not storage_utils.path_exists(
            self.metadata_uri(canonical_uuid),
            client=self._client,
        ):
            return None
        return self._read_document(canonical_uuid).raw

    def read_metadata(self, clip_uuid: str | UUID) -> dict[str, Any]:
        """Read and validate the final metadata document."""
        canonical_uuid = _canonical_clip_uuid(clip_uuid)
        return self._read_document(canonical_uuid).raw

    def iter_chunks(
        self,
        clip_uuid: str | UUID,
    ) -> Iterator[TemporalAnnotationChunk]:
        """Load, validate, and yield one bounded NPZ chunk at a time."""
        canonical_uuid = _canonical_clip_uuid(clip_uuid)
        document = self._read_document(canonical_uuid)
        previous_timestamp: int | None = None
        for frame_start in range(0, document.frame_count, document.chunk_frames):
            frame_stop = min(frame_start + document.chunk_frames, document.frame_count)
            chunk_uri = _join_location(
                self._base_path,
                _chunk_sub_path(canonical_uuid, frame_start, frame_stop),
            )
            arrays = self._read_chunk(
                chunk_uri,
                frame_start=frame_start,
                frame_stop=frame_stop,
                specs=document.arrays,
            )
            timestamps = cast(
                "npt.NDArray[np.int64]",
                arrays[document.timestamp_array],
            )
            if timestamps.ndim != 1:
                message = (
                    f"annotation timestamp array {document.timestamp_array!r} "
                    f"must be one-dimensional, got {timestamps.shape}"
                )
                raise ValueError(message)
            if len(timestamps) > 1 and np.any(timestamps[1:] <= timestamps[:-1]):
                message = f"annotation timestamps are not strictly increasing in {chunk_uri}"
                raise ValueError(message)
            if len(timestamps) and previous_timestamp is not None and int(timestamps[0]) <= previous_timestamp:
                message = f"annotation timestamps are not strictly increasing across chunks at {chunk_uri}"
                raise ValueError(message)
            if len(timestamps):
                previous_timestamp = int(timestamps[-1])
            yield TemporalAnnotationChunk(
                frame_start=frame_start,
                frame_stop=frame_stop,
                arrays=arrays,
            )

    def _read_document(self, canonical_uuid: str) -> _AnnotationDocument:
        metadata_uri = _join_location(
            self._base_path,
            _metadata_sub_path(canonical_uuid),
        )
        try:
            value = json.loads(
                storage_utils.read_bytes(metadata_uri, client=self._client).decode("utf-8"),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            message = f"invalid annotation metadata JSON: {metadata_uri}"
            raise ValueError(message) from error
        return _parse_annotation_document(
            value,
            canonical_uuid=canonical_uuid,
            metadata_uri=metadata_uri,
        )

    def _read_chunk(
        self,
        chunk_uri: str,
        *,
        frame_start: int,
        frame_stop: int,
        specs: Mapping[str, _ArraySpec],
    ) -> dict[str, npt.NDArray[np.generic]]:
        source: Path | io.BytesIO
        if self._client is None:
            source = Path(chunk_uri)
        else:
            source = io.BytesIO(storage_utils.read_bytes(chunk_uri, client=self._client))
        loaded = np.load(source, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            message = f"annotation chunk is not an NPZ archive: {chunk_uri}"
            raise TypeError(message)
        try:
            if set(loaded.files) != set(specs):
                message = (
                    f"annotation chunk arrays do not match metadata in {chunk_uri}: "
                    f"expected={sorted(specs)}, observed={sorted(loaded.files)}"
                )
                raise ValueError(message)
            arrays: dict[str, npt.NDArray[np.generic]] = {}
            chunk_length = frame_stop - frame_start
            for name, spec in specs.items():
                array = loaded[name]
                expected_shape = (chunk_length, *spec.shape[1:])
                if array.dtype != spec.dtype:
                    message = f"annotation array {name!r} has dtype {array.dtype} in {chunk_uri}, expected {spec.dtype}"
                    raise ValueError(message)
                if array.shape != expected_shape:
                    message = (
                        f"annotation array {name!r} has shape {array.shape} in {chunk_uri}, expected {expected_shape}"
                    )
                    raise ValueError(message)
                arrays[name] = array
            return arrays
        finally:
            loaded.close()


def _parse_annotation_document(
    value: object,
    *,
    canonical_uuid: str,
    metadata_uri: str,
) -> _AnnotationDocument:
    if not isinstance(value, dict):
        message = f"annotation metadata must be a JSON object: {metadata_uri}"
        raise TypeError(message)
    document = cast("dict[str, Any]", value)
    if document.get("schema") != _SCHEMA:
        message = f"unsupported annotation schema in {metadata_uri}: {document.get('schema')!r}"
        raise ValueError(message)
    if document.get("clip_uuid") != canonical_uuid:
        message = f"annotation metadata UUID does not match {canonical_uuid}: {document.get('clip_uuid')!r}"
        raise ValueError(message)
    if document.get("format") != "npz":
        message = f"unsupported annotation format in {metadata_uri}: {document.get('format')!r}"
        raise ValueError(message)

    frame_count = _positive_int(document.get("frame_count"), field_name="frame_count")
    chunk_frames = _positive_int(document.get("chunk_frames"), field_name="chunk_frames")
    expected_template = f"chunks/v1/{canonical_uuid}/frames-{{frame_start:09d}}-{{frame_stop:09d}}.npz"
    if document.get("chunk_path_template") != expected_template:
        message = f"unsupported annotation chunk path template in {metadata_uri}"
        raise ValueError(message)

    metadata = _string_mapping(document.get("metadata"), field_name="metadata")
    arrays = _parse_array_specs(metadata.get("arrays"), frame_count=frame_count)
    timestamp_array = _parse_timestamp_array(
        metadata.get("alignment"),
        arrays=arrays,
        frame_count=frame_count,
    )
    return _AnnotationDocument(
        raw=document,
        frame_count=frame_count,
        chunk_frames=chunk_frames,
        arrays=arrays,
        timestamp_array=timestamp_array,
    )


def _parse_array_specs(value: object, *, frame_count: int) -> dict[str, _ArraySpec]:
    array_values = _string_mapping(value, field_name="metadata.arrays")
    if not array_values:
        message = "metadata.arrays must not be empty"
        raise ValueError(message)
    arrays: dict[str, _ArraySpec] = {}
    for name, spec_value in array_values.items():
        if not name:
            message = "metadata.arrays keys must be non-empty strings"
            raise ValueError(message)
        arrays[name] = _parse_array_spec(
            spec_value,
            name=name,
            frame_count=frame_count,
        )
    return arrays


def _parse_array_spec(value: object, *, name: str, frame_count: int) -> _ArraySpec:
    spec = _string_mapping(value, field_name=f"metadata.arrays.{name}")
    dtype_value = spec.get("dtype")
    if not isinstance(dtype_value, str):
        message = f"metadata.arrays.{name}.dtype must be a string"
        raise TypeError(message)
    try:
        dtype = np.dtype(dtype_value)
    except TypeError as error:
        message = f"metadata.arrays.{name}.dtype is invalid: {dtype_value!r}"
        raise ValueError(message) from error
    if dtype.name != dtype_value:
        message = f"metadata.arrays.{name}.dtype must use canonical NumPy name {dtype.name!r}"
        raise ValueError(message)

    shape_value = spec.get("shape")
    if not isinstance(shape_value, list) or not shape_value:
        message = f"metadata.arrays.{name}.shape must be a non-empty list"
        raise ValueError(message)
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape_value):
        message = f"metadata.arrays.{name}.shape must contain positive integers"
        raise ValueError(message)
    shape = tuple(shape_value)
    axes = spec.get("axes")
    if axes is not None and (not isinstance(axes, str) or len(axes) != len(shape) or not axes.startswith("T")):
        message = f"metadata.arrays.{name}.axes must describe a leading temporal axis"
        raise ValueError(message)
    if shape[0] != frame_count:
        message = f"metadata.arrays.{name}.shape starts with {shape[0]}, expected frame_count={frame_count}"
        raise ValueError(message)
    return _ArraySpec(dtype=dtype, shape=shape)


def _parse_timestamp_array(
    value: object,
    *,
    arrays: Mapping[str, _ArraySpec],
    frame_count: int,
) -> str:
    alignment = _string_mapping(value, field_name="metadata.alignment")
    timestamp_array = alignment.get("timestamp_array")
    if not isinstance(timestamp_array, str) or timestamp_array not in arrays:
        message = "metadata.alignment.timestamp_array must name a declared array"
        raise ValueError(message)
    timestamp_spec = arrays[timestamp_array]
    if timestamp_spec.dtype != np.dtype(np.int64) or timestamp_spec.shape != (frame_count,):
        message = f"timestamp array {timestamp_array!r} must be int64 with shape [{frame_count}]"
        raise ValueError(message)
    return timestamp_array


class TemporalAnnotationWriter:
    """Publish NumPy temporal chunks and one final per-clip metadata record.

    This is deliberately a write-only helper, not a repository or catalog. A
    pipeline must assign at most one active writer to a given
    ``(output_path, clip_uuid)``.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        profile_name: str = "default",
        tmp_dir: str | Path | None = None,
    ) -> None:
        """Create a writer rooted at one annotation kind and producer release."""
        output_path_str = str(output_path)
        if not output_path_str.strip():
            message = "output_path must be non-empty"
            raise ValueError(message)
        self._writer = StorageWriter(
            output_path_str,
            profile_name=profile_name,
            tmp_dir=tmp_dir,
        )
        self._tmp_dir = Path(tmp_dir) if tmp_dir is not None else None
        self._states: dict[str, _ClipWriteState] = {}

    @contextlib.contextmanager
    def open_chunk(
        self,
        clip_uuid: str | UUID,
        frame_start: int,
        frame_stop: int,
    ) -> Iterator[Path]:
        """Yield a temporary ``.npz`` path and publish it on normal exit.

        ``frame_start`` is inclusive and ``frame_stop`` is exclusive. The caller
        writes the path, normally with ``numpy.savez``. Empty files and concurrent
        writes to the same range are rejected. A previously published range may be
        atomically overwritten so a retried pipeline task can start from the first
        chunk again.
        """
        canonical_uuid = _canonical_clip_uuid(clip_uuid)
        frame_range = _validate_frame_range(frame_start, frame_stop)
        state = self._states.setdefault(canonical_uuid, _ClipWriteState())
        if frame_range in state.pending_ranges:
            message = f"annotation clip {canonical_uuid} is already writing frame range [{frame_start}, {frame_stop})"
            raise ValueError(message)

        sub_path = _chunk_sub_path(canonical_uuid, frame_start, frame_stop)
        state.pending_ranges.add(frame_range)
        staging_path: Path | None = None
        try:
            staging_path = self._create_staging_path(sub_path)
            yield staging_path
            self._publish_staged_file(staging_path, sub_path)
        except BaseException:
            if staging_path is not None:
                staging_path.unlink(missing_ok=True)
            raise
        else:
            state.published_ranges.add(frame_range)
        finally:
            state.pending_ranges.discard(frame_range)
            if staging_path is not None:
                staging_path.unlink(missing_ok=True)
            if not state.pending_ranges and not state.published_ranges:
                self._states.pop(canonical_uuid, None)

    def complete_clip(
        self,
        clip_uuid: str | UUID,
        *,
        frame_count: int,
        chunk_frames: int,
        metadata: Mapping[str, object],
    ) -> str:
        """Validate chunk coverage and publish the per-clip JSON last.

        The JSON is the completion record. Expected chunk paths are derived from
        ``frame_count`` and ``chunk_frames``; no chunk manifest is written.

        Returns:
            The local path or remote URI of the metadata JSON.

        """
        canonical_uuid = _canonical_clip_uuid(clip_uuid)
        frame_count = _positive_int(frame_count, field_name="frame_count")
        chunk_frames = _positive_int(chunk_frames, field_name="chunk_frames")
        if any(not isinstance(key, str) or not key for key in metadata):
            message = "metadata keys must be non-empty strings"
            raise ValueError(message)

        state = self._states.get(canonical_uuid)
        if state is None:
            message = f"annotation clip {canonical_uuid} has no published chunks"
            raise ValueError(message)
        if state.pending_ranges:
            message = f"annotation clip {canonical_uuid} still has pending chunks"
            raise RuntimeError(message)

        expected_ranges = [
            (start, min(start + chunk_frames, frame_count)) for start in range(0, frame_count, chunk_frames)
        ]
        observed_ranges = sorted(state.published_ranges)
        if observed_ranges != expected_ranges:
            message = (
                f"annotation clip {canonical_uuid} chunk coverage mismatch: "
                f"expected={expected_ranges}, observed={observed_ranges}"
            )
            raise ValueError(message)

        document = {
            "schema": _SCHEMA,
            "clip_uuid": canonical_uuid,
            "format": "npz",
            "frame_count": frame_count,
            "chunk_frames": chunk_frames,
            "chunk_path_template": (f"chunks/v1/{canonical_uuid}/frames-{{frame_start:09d}}-{{frame_stop:09d}}.npz"),
            "metadata": dict(metadata),
        }
        try:
            payload = json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            message = "metadata must contain only finite JSON values"
            raise ValueError(message) from error

        sub_path = _metadata_sub_path(canonical_uuid)
        staging_path = self._create_staging_path(sub_path)
        try:
            staging_path.write_text(payload + "\n", encoding="utf-8")
            self._publish_staged_file(staging_path, sub_path)
        finally:
            staging_path.unlink(missing_ok=True)
        del self._states[canonical_uuid]
        return _join_location(self._writer.base_path, sub_path)

    def _create_staging_path(self, sub_path: str) -> Path:
        if self._writer.is_remote:
            staging_dir = self._tmp_dir or Path(tempfile.gettempdir())
        else:
            staging_dir = Path(self._writer.base_path) / Path(sub_path).parent
        staging_dir.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{Path(sub_path).name}.partial-",
            suffix=Path(sub_path).suffix,
            dir=staging_dir,
        )
        os.close(descriptor)
        return Path(name)

    def _publish_staged_file(self, staging_path: Path, sub_path: str) -> None:
        try:
            status = staging_path.stat(follow_symlinks=False)
        except FileNotFoundError as error:
            message = f"annotation staging file was not written: {staging_path}"
            raise ValueError(message) from error
        if not stat.S_ISREG(status.st_mode):
            message = f"annotation staging path must be a regular file: {staging_path}"
            raise ValueError(message)
        if status.st_size == 0:
            message = f"annotation staging file must be non-empty: {staging_path}"
            raise ValueError(message)

        with staging_path.open("rb") as staged_file:
            os.fsync(staged_file.fileno())

        if self._writer.is_remote:
            self._writer.upload_file_to(sub_path, staging_path)
            return

        target = Path(self._writer.base_path) / sub_path
        target.parent.mkdir(parents=True, exist_ok=True)
        staging_path.replace(target)


def _canonical_clip_uuid(value: str | UUID) -> str:
    try:
        return str(value if isinstance(value, UUID) else UUID(value))
    except (AttributeError, TypeError, ValueError) as error:
        message = f"clip_uuid must be a valid UUID: {value!r}"
        raise ValueError(message) from error


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{field_name} must be an integer"
        raise TypeError(message)
    if value <= 0:
        message = f"{field_name} must be positive"
        raise ValueError(message)
    return value


def _string_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        message = f"{field_name} must be an object"
        raise TypeError(message)
    if any(not isinstance(key, str) for key in value):
        message = f"{field_name} keys must be strings"
        raise ValueError(message)
    return cast("Mapping[str, object]", value)


def _validate_frame_range(frame_start: int, frame_stop: int) -> tuple[int, int]:
    if isinstance(frame_start, bool) or not isinstance(frame_start, int):
        message = "frame_start must be an integer"
        raise TypeError(message)
    if isinstance(frame_stop, bool) or not isinstance(frame_stop, int):
        message = "frame_stop must be an integer"
        raise TypeError(message)
    if frame_start < 0:
        message = "frame_start must be non-negative"
        raise ValueError(message)
    if frame_stop <= frame_start:
        message = "frame_stop must be greater than frame_start"
        raise ValueError(message)
    return frame_start, frame_stop


def _chunk_sub_path(clip_uuid: str, frame_start: int, frame_stop: int) -> str:
    return _CHUNK_PATH_TEMPLATE.format(
        clip_uuid=clip_uuid,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )


def _metadata_sub_path(clip_uuid: str) -> str:
    return _METADATA_PATH_TEMPLATE.format(clip_uuid=clip_uuid)


def _join_location(base_path: str, sub_path: str) -> str:
    return f"{base_path.rstrip('/')}/{sub_path}"


__all__ = [
    "TemporalAnnotationChunk",
    "TemporalAnnotationReader",
    "TemporalAnnotationWriter",
]
