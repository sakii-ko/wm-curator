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

"""Write temporal annotation chunks without content-addressed indirection.

The caller owns the annotation namespace through ``output_path`` (for example,
``annotations/normals/normalcrafter-v1``). Each chunk is published independently,
and the per-clip metadata JSON is published last as the logical completion record.
"""

import contextlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from cosmos_curator.core.utils.storage.storage_utils import StorageWriter

_SCHEMA = "cosmos-curator.temporal-annotation/v1"
_CHUNK_PATH_TEMPLATE = "chunks/v1/{clip_uuid}/frames-{frame_start:09d}-{frame_stop:09d}.npz"
_METADATA_PATH_TEMPLATE = "metas/v1/{clip_uuid}.json"


@dataclass
class _ClipWriteState:
    published_ranges: set[tuple[int, int]] = field(default_factory=set)
    pending_ranges: set[tuple[int, int]] = field(default_factory=set)


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


def _positive_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{field_name} must be an integer"
        raise TypeError(message)
    if value <= 0:
        message = f"{field_name} must be positive"
        raise ValueError(message)
    return value


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


__all__ = ["TemporalAnnotationWriter"]
