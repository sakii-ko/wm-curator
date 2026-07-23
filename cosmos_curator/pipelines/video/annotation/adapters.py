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
"""Thin adapters from common dataset layouts to annotation tasks."""

import json
import pathlib
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Never, Protocol, runtime_checkable
from urllib.parse import urlsplit

import attrs

from cosmos_curator.core.utils.storage.storage_client import StorageClient, StoragePrefix
from cosmos_curator.core.utils.storage.storage_utils import (
    get_full_path,
    get_storage_client,
    is_remote_path,
    read_json_file,
)
from cosmos_curator.pipelines.video.annotation.data_model import (
    AnnotationTask,
    SourcePath,
    make_annotation_task,
    normalize_clip_uuid,
    normalize_relative_path,
    normalize_span,
)

DEFAULT_VIDEO_EXTENSIONS = frozenset(
    {
        ".avi",
        ".m2ts",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".ts",
        ".webm",
    }
)

_JSONL_FIELDS = frozenset(
    {
        "id",
        "clip_uuid",
        "metadata",
        "path",
        "relative_path",
        "rotation_degrees_clockwise",
        "span",
        "span_uuid",
        "stream_index",
    }
)

type SourceRoot = pathlib.Path | StoragePrefix


@runtime_checkable
class AnnotationDatasetAdapter(Protocol):
    """Minimal discovery contract, also suitable for a future Parquet adapter."""

    def discover(self) -> list[AnnotationTask]:
        """Return materialized tasks in deterministic adapter order."""


def _copy_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"metadata must be an object, got {type(value).__name__}"
        raise TypeError(msg)
    if any(not isinstance(key, str) for key in value):
        msg = "metadata keys must be strings"
        raise TypeError(msg)
    return dict(value)


def _record_clip_uuid(record: Mapping[str, object]) -> uuid.UUID | None:
    raw_clip_uuid = record.get("clip_uuid")
    raw_span_uuid = record.get("span_uuid")
    if raw_clip_uuid is None and raw_span_uuid is None:
        return None

    clip_uuid = normalize_clip_uuid(raw_clip_uuid) if raw_clip_uuid is not None else None
    span_uuid = normalize_clip_uuid(raw_span_uuid) if raw_span_uuid is not None else None
    if clip_uuid is not None and span_uuid is not None and clip_uuid != span_uuid:
        msg = f"clip_uuid and span_uuid must match when both are provided, got {clip_uuid} and {span_uuid}"
        raise ValueError(msg)
    return clip_uuid or span_uuid


def _normalize_extensions(values: Iterable[str]) -> frozenset[str]:
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            msg = f"video extensions must be non-empty strings, got {value!r}"
            raise ValueError(msg)
        extension = value.strip().lower()
        normalized.add(extension if extension.startswith(".") else f".{extension}")
    if not normalized:
        msg = "at least one video extension is required"
        raise ValueError(msg)
    return frozenset(normalized)


def _normalize_source_root(value: str | pathlib.Path | StoragePrefix) -> SourceRoot:
    if isinstance(value, StoragePrefix):
        return value
    if isinstance(value, str) and is_remote_path(value):
        result = get_full_path(value)
        assert isinstance(result, StoragePrefix)
        return result
    return pathlib.Path(value).expanduser().resolve()


def _remote_relative_path(source: str) -> str:
    name = pathlib.PurePosixPath(urlsplit(source).path).name
    if not name:
        msg = f"remote source path must identify an object, got {source!r}"
        raise ValueError(msg)
    return name


def _resolve_jsonl_source(
    raw_source: str,
    *,
    source_root: SourceRoot,
    require_source_exists: bool,
) -> tuple[SourcePath, str]:
    if is_remote_path(raw_source):
        source = get_full_path(raw_source)
        assert isinstance(source, StoragePrefix)
        return source, _remote_relative_path(raw_source)

    if isinstance(source_root, StoragePrefix):
        relative_path = normalize_relative_path(raw_source)
        source = get_full_path(source_root, relative_path)
        assert isinstance(source, StoragePrefix)
        return source, relative_path

    raw_path = pathlib.Path(raw_source).expanduser()
    unresolved = raw_path if raw_path.is_absolute() else source_root / raw_path
    source_path = unresolved.resolve(strict=require_source_exists)
    if require_source_exists and not source_path.is_file():
        msg = f"source path is not a file: {source_path}"
        raise ValueError(msg)
    try:
        relative_path = source_path.relative_to(source_root).as_posix()
    except ValueError:
        relative_path = source_path.name
    return source_path, normalize_relative_path(relative_path)


def annotation_task_from_mapping(
    record: Mapping[str, object],
    *,
    source_root: SourceRoot,
    require_source_exists: bool = True,
    dataset_metadata: Mapping[str, Any] | None = None,
) -> AnnotationTask:
    """Convert one JSON-like row into a task.

    A Parquet adapter can reuse this function after converting each row to a
    mapping; no catalog or second record model is required.
    """
    if any(not isinstance(key, str) for key in record):
        msg = "JSONL field names must be strings"
        raise TypeError(msg)
    unknown = set(record) - _JSONL_FIELDS
    if unknown:
        msg = f"unknown JSONL fields: {sorted(unknown)}"
        raise ValueError(msg)

    raw_source = record.get("path")
    if not isinstance(raw_source, str) or not raw_source.strip():
        msg = "JSONL field 'path' must be a non-empty string"
        raise ValueError(msg)
    source, inferred_relative_path = _resolve_jsonl_source(
        raw_source,
        source_root=source_root,
        require_source_exists=require_source_exists,
    )

    raw_relative_path = record.get("relative_path")
    if raw_relative_path is None:
        relative_path = inferred_relative_path
    elif isinstance(raw_relative_path, str):
        relative_path = normalize_relative_path(raw_relative_path)
    else:
        msg = "JSONL field 'relative_path' must be a string"
        raise TypeError(msg)

    raw_session_id = record.get("id", relative_path)
    if not isinstance(raw_session_id, str) or not raw_session_id.strip():
        msg = "JSONL field 'id' must be a non-empty string"
        raise ValueError(msg)

    merged_metadata = _copy_metadata({} if dataset_metadata is None else dataset_metadata)
    if "metadata" in record:
        merged_metadata.update(_copy_metadata(record["metadata"]))

    return make_annotation_task(
        source,
        session_id=raw_session_id,
        relative_path=relative_path,
        stream_index=record.get("stream_index"),  # type: ignore[arg-type]
        rotation_degrees_clockwise=record.get("rotation_degrees_clockwise"),  # type: ignore[arg-type]
        span=normalize_span(record.get("span")),
        clip_uuid=_record_clip_uuid(record),
        dataset_metadata=merged_metadata,
    )


@attrs.frozen
class FilesystemDatasetAdapter:
    """Recursively discover local videos below one dataset root."""

    root: pathlib.Path = attrs.field(converter=pathlib.Path)
    extensions: frozenset[str] = attrs.field(
        factory=lambda: DEFAULT_VIDEO_EXTENSIONS,
        converter=_normalize_extensions,
    )
    stream_index: int | None = None
    rotation_degrees_clockwise: int | None = None
    dataset_metadata: dict[str, Any] = attrs.field(factory=dict, converter=_copy_metadata)

    def discover(self) -> list[AnnotationTask]:
        """Return extension-filtered files sorted by relative POSIX path."""
        root = self.root.expanduser().resolve(strict=True)
        if not root.is_dir():
            msg = f"filesystem dataset root is not a directory: {root}"
            raise ValueError(msg)

        paths = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in self.extensions),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        tasks = []
        for path in paths:
            relative_path = path.relative_to(root).as_posix()
            tasks.append(
                make_annotation_task(
                    path.resolve(strict=True),
                    session_id=relative_path,
                    relative_path=relative_path,
                    stream_index=self.stream_index,
                    rotation_degrees_clockwise=self.rotation_degrees_clockwise,
                    dataset_metadata=self.dataset_metadata,
                )
            )
        return tasks


def _normalize_annotation_version(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = "annotation_version must be a non-empty path component"
        raise ValueError(msg)
    if pathlib.PurePosixPath(value).name != value or value in {".", ".."} or "\\" in value:
        msg = f"annotation_version must be one path component, got {value!r}"
        raise ValueError(msg)
    return value


def _resolve_published_source(raw_source: object, *, require_source_exists: bool) -> SourcePath:
    if not isinstance(raw_source, str) or not raw_source.strip():
        msg = "source-span metadata field 'source_video' must be a non-empty string"
        raise ValueError(msg)
    if is_remote_path(raw_source):
        source = get_full_path(raw_source)
        assert isinstance(source, StoragePrefix)
        return source

    source_path = pathlib.Path(raw_source).expanduser().resolve(strict=require_source_exists)
    if require_source_exists and not source_path.is_file():
        msg = f"source-span source path is not a file: {source_path}"
        raise ValueError(msg)
    return source_path


def _read_json_mapping(
    path: StoragePrefix | pathlib.Path,
    client: StorageClient | None,
    *,
    description: str,
) -> Mapping[str, object]:
    value: object = read_json_file(path, client)
    if not isinstance(value, Mapping):
        msg = f"{description} must contain a JSON object: {path}"
        raise TypeError(msg)
    if any(not isinstance(key, str) for key in value):
        msg = f"{description} keys must be strings: {path}"
        raise TypeError(msg)
    return value


@attrs.frozen
class SourceSpanDatasetAdapter:
    """Read source-backed clip references written by the Cosmos split pipeline."""

    output_path: str | pathlib.Path | StoragePrefix
    annotation_version: str = attrs.field(default="v0", converter=_normalize_annotation_version)
    profile_name: str = "default"
    stream_index: int | None = None
    rotation_degrees_clockwise: int | None = None
    require_source_exists: bool = True
    dataset_metadata: dict[str, Any] = attrs.field(factory=dict, converter=_copy_metadata)

    def discover(self) -> list[AnnotationTask]:
        """Return one annotation task per valid source-span clip in ``summary.json``."""
        output_root = _normalize_source_root(self.output_path)
        client = get_storage_client(str(output_root), profile_name=self.profile_name)
        summary_path = get_full_path(output_root, "summary.json")
        summary = _read_json_mapping(summary_path, client, description="Cosmos summary")

        video_records = [
            (relative_path, value)
            for relative_path, value in summary.items()
            if isinstance(value, Mapping) and "clips" in value
        ]
        tasks: list[AnnotationTask] = []
        seen_clip_uuids: set[uuid.UUID] = set()
        for raw_relative_path, video_record in sorted(video_records, key=lambda item: item[0]):
            relative_path = normalize_relative_path(raw_relative_path)
            raw_clip_uuids = video_record["clips"]
            if not isinstance(raw_clip_uuids, list):
                msg = f"Cosmos summary clips for {relative_path!r} must be a list"
                raise TypeError(msg)

            for raw_clip_uuid in raw_clip_uuids:
                clip_uuid = normalize_clip_uuid(raw_clip_uuid)
                if clip_uuid in seen_clip_uuids:
                    msg = f"duplicate span_uuid in Cosmos summary: {clip_uuid}"
                    raise ValueError(msg)
                seen_clip_uuids.add(clip_uuid)

                metadata_path = get_full_path(
                    output_root,
                    "metas",
                    self.annotation_version,
                    f"{clip_uuid}.json",
                )
                clip_metadata = _read_json_mapping(
                    metadata_path,
                    client,
                    description="Cosmos source-span metadata",
                )
                task = self._task_from_clip_metadata(
                    clip_metadata,
                    metadata_path=metadata_path,
                    relative_path=relative_path,
                    clip_uuid=clip_uuid,
                )
                if task is not None:
                    tasks.append(task)
        return tasks

    def _task_from_clip_metadata(
        self,
        clip_metadata: Mapping[str, object],
        *,
        metadata_path: StoragePrefix | pathlib.Path,
        relative_path: str,
        clip_uuid: uuid.UUID,
    ) -> AnnotationTask | None:
        if clip_metadata.get("clip_format") != "source_span":
            msg = f"expected clip_format='source_span' in {metadata_path}"
            raise ValueError(msg)
        metadata_uuid = normalize_clip_uuid(clip_metadata.get("span_uuid"))
        if metadata_uuid != clip_uuid:
            msg = f"summary span_uuid {clip_uuid} does not match metadata span_uuid {metadata_uuid} in {metadata_path}"
            raise ValueError(msg)

        valid = clip_metadata.get("valid")
        if not isinstance(valid, bool):
            msg = f"source-span metadata field 'valid' must be a boolean in {metadata_path}"
            raise TypeError(msg)
        if not valid:
            return None

        span = normalize_span(clip_metadata.get("duration_span"))
        if span is None:
            msg = f"source-span metadata is missing duration_span in {metadata_path}"
            raise ValueError(msg)
        source = _resolve_published_source(
            clip_metadata.get("source_video"),
            require_source_exists=self.require_source_exists,
        )
        return make_annotation_task(
            source,
            session_id=f"{relative_path}#{clip_uuid}",
            relative_path=relative_path,
            stream_index=self.stream_index,
            rotation_degrees_clockwise=self.rotation_degrees_clockwise,
            span=span,
            clip_uuid=clip_uuid,
            dataset_metadata=self.dataset_metadata,
        )


@attrs.frozen
class JsonlDatasetAdapter:
    """Read an optional, plain JSONL input list.

    This is only a convenience for datasets that cannot be discovered from a
    directory. It is not a dataset manifest, catalog, or integrity ledger.
    """

    jsonl_path: pathlib.Path = attrs.field(converter=pathlib.Path)
    source_root: str | pathlib.Path | StoragePrefix | None = None
    require_source_exists: bool = True
    dataset_metadata: dict[str, Any] = attrs.field(factory=dict, converter=_copy_metadata)

    def discover(self) -> list[AnnotationTask]:
        """Return tasks in input-list order, rejecting duplicate IDs."""
        jsonl_path = self.jsonl_path.expanduser().resolve(strict=True)
        if not jsonl_path.is_file():
            msg = f"JSONL input path is not a file: {jsonl_path}"
            raise ValueError(msg)
        source_root = _normalize_source_root(jsonl_path.parent if self.source_root is None else self.source_root)

        tasks: list[AnnotationTask] = []
        session_ids: set[str] = set()
        with jsonl_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    task = _annotation_task_from_json_value(
                        record,
                        source_root=source_root,
                        require_source_exists=self.require_source_exists,
                        dataset_metadata=self.dataset_metadata,
                    )
                except (OSError, TypeError, ValueError) as error:
                    _raise_jsonl_line_error(jsonl_path, line_number, error)
                if task.session_id in session_ids:
                    error_msg = f"duplicate JSONL id: {task.session_id!r}"
                    _raise_jsonl_line_error(jsonl_path, line_number, ValueError(error_msg))
                session_ids.add(task.session_id)
                tasks.append(task)
        return tasks


def _annotation_task_from_json_value(
    value: object,
    *,
    source_root: SourceRoot,
    require_source_exists: bool,
    dataset_metadata: Mapping[str, Any],
) -> AnnotationTask:
    if not isinstance(value, Mapping):
        msg = "each JSONL line must contain an object"
        raise TypeError(msg)
    return annotation_task_from_mapping(
        value,
        source_root=source_root,
        require_source_exists=require_source_exists,
        dataset_metadata=dataset_metadata,
    )


def _raise_jsonl_line_error(
    jsonl_path: pathlib.Path,
    line_number: int,
    error: Exception,
) -> Never:
    msg = f"{jsonl_path}:{line_number}: {error}"
    raise ValueError(msg) from error
