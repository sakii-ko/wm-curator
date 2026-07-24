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
"""Decode-time alignment grid shared by geometry and normal annotation stages."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol, cast

import av
import numpy as np
import numpy.typing as npt
from PIL import Image

from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.pipelines.video.annotation.data_model import normalize_span

_NANOSECONDS_PER_SECOND = 1_000_000_000
_RGB_CHANNELS = 3
_VIDEO_NDIM = 4
_GRID_VERSION = 1
_ROTATION_90 = 90
_ROTATION_180 = 180


@dataclass(frozen=True, slots=True)
class AnnotationGrid:
    """Regular temporal grid and fixed RGB raster used before model inference."""

    sample_fps: float = 15.0
    width: int = 832
    height: int = 480

    def __post_init__(self) -> None:
        """Validate the three user-facing grid controls."""
        if (
            isinstance(self.sample_fps, bool)
            or not isinstance(self.sample_fps, (int, float))
            or not math.isfinite(self.sample_fps)
            or self.sample_fps <= 0
        ):
            message = "sample_fps must be finite and positive"
            raise ValueError(message)
        for field_name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                message = f"{field_name} must be a positive integer"
                raise ValueError(message)

    def configuration_metadata(self) -> dict[str, object]:
        """Return only fields that decide whether a completed artifact is reusable."""
        return {
            "version": _GRID_VERSION,
            "sample_fps": float(self.sample_fps),
            "timestamp_kind": "regular_alignment_grid",
            "source_selection": "nearest",
            "span_semantics": "half_open",
            "output_size": {"width": self.width, "height": self.height},
            "spatial_transform": "center_crop_resize",
            "interpolation": "bilinear",
            "pixel_coordinates": "integer_pixel_centers",
        }

    def metadata(self, raster: "RasterTransform") -> dict[str, object]:
        """Describe the temporal grid and exact grid-to-source raster mapping."""
        metadata = self.configuration_metadata()
        metadata.update(raster.metadata())
        return metadata


DEFAULT_ANNOTATION_GRID = AnnotationGrid()


@dataclass(frozen=True, slots=True)
class RasterTransform:
    """Centered aspect crop and its output-pixel-to-source-pixel transform."""

    source_width: int
    source_height: int
    rotation_degrees_clockwise: int
    oriented_width: int
    oriented_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    grid_to_oriented_source: tuple[tuple[float, float, float], ...]
    grid_to_source: tuple[tuple[float, float, float], ...]

    def metadata(self) -> dict[str, object]:
        """Serialize the raster contract without adding an identifier or hash."""
        return {
            "source_size": {"width": self.source_width, "height": self.source_height},
            "rotation_degrees_clockwise": self.rotation_degrees_clockwise,
            "oriented_source_size": {
                "width": self.oriented_width,
                "height": self.oriented_height,
            },
            "crop_xywh": [
                self.crop_left,
                self.crop_top,
                self.crop_width,
                self.crop_height,
            ],
            "grid_to_oriented_source_pixel_center": [list(row) for row in self.grid_to_oriented_source],
            "grid_to_source_pixel_center": [list(row) for row in self.grid_to_source],
        }


@dataclass(frozen=True, slots=True)
class DecodedAnnotationClip:
    """Low-resolution RGB grid plus regular and selected-source timestamps."""

    frames: npt.NDArray[np.uint8]
    timestamps_ns: npt.NDArray[np.int64]
    source_timestamps_ns: npt.NDArray[np.int64]
    source_span: tuple[float, float]
    raster: RasterTransform
    decoder_backend: str


class AnnotationClipDecoder(Protocol):
    """Decode one source span directly into the shared annotation grid."""

    def __call__(  # noqa: PLR0913
        self,
        source: Path,
        span: tuple[float, float] | None,
        *,
        stream_index: int,
        rotation_degrees_clockwise: int,
        grid: AnnotationGrid,
        min_frames: int,
        max_frames: int,
    ) -> DecodedAnnotationClip:
        """Decode the requested source span."""


@dataclass(frozen=True, slots=True)
class _VideoTimeline:
    pts_stream: npt.NDArray[np.int64]
    timestamps_ns: npt.NDArray[np.int64]
    source_width: int
    source_height: int
    nominal_fps: float | None
    duration_ns: int | None


class _SelectedFrameDecodeError(RuntimeError):
    """Signal that seek decode should retry as one sequential scan."""


class _PacketTimelineMismatchError(RuntimeError):
    """Signal that packet metadata must be replaced with decoded-frame PTS."""


def decode_annotation_clip(  # noqa: PLR0913
    source: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    rotation_degrees_clockwise: int,
    grid: AnnotationGrid,
    min_frames: int,
    max_frames: int,
) -> DecodedAnnotationClip:
    """Decode nearest source frames and resize each one before retaining it."""
    try:
        timeline = _probe_packet_timeline(source, stream_index=stream_index)
        return _decode_annotation_timeline(
            source,
            span,
            stream_index=stream_index,
            rotation_degrees_clockwise=rotation_degrees_clockwise,
            grid=grid,
            min_frames=min_frames,
            max_frames=max_frames,
            timeline=timeline,
        )
    except _PacketTimelineMismatchError as packet_error:
        packet_error_message = str(packet_error)
    try:
        timeline = _probe_decoded_timeline(source, stream_index=stream_index)
        return _decode_annotation_timeline(
            source,
            span,
            stream_index=stream_index,
            rotation_degrees_clockwise=rotation_degrees_clockwise,
            grid=grid,
            min_frames=min_frames,
            max_frames=max_frames,
            timeline=timeline,
        )
    except Exception as decoded_error:
        message = (
            "PyAV packet timestamps did not match display frames and decoded-frame "
            f"fallback failed: packet={packet_error_message}; decoded={decoded_error}"
        )
        raise RuntimeError(message) from decoded_error


def _decode_annotation_timeline(  # noqa: PLR0913
    source: Path,
    span: tuple[float, float] | None,
    *,
    stream_index: int,
    rotation_degrees_clockwise: int,
    grid: AnnotationGrid,
    min_frames: int,
    max_frames: int,
    timeline: _VideoTimeline,
) -> DecodedAnnotationClip:
    source_duration_ns = _timeline_duration_ns(timeline, sample_fps=grid.sample_fps)
    source_span = _resolve_span(span, source_duration_ns)
    start_ns = round(source_span[0] * _NANOSECONDS_PER_SECOND)
    stop_ns = round(source_span[1] * _NANOSECONDS_PER_SECOND)
    _, _, timestamps_ns = make_ts_grid(
        start_ns,
        exclusive_end_ns=stop_ns,
        sample_rate_hz=grid.sample_fps,
    )
    _validate_frame_count(
        len(timestamps_ns),
        min_frames=min_frames,
        max_frames=max_frames,
        source_span=source_span,
    )
    selected_indices = _select_nearest_source_indices(
        timeline.timestamps_ns,
        timestamps_ns,
        start_ns=start_ns,
        stop_ns=stop_ns,
    )
    selected_pts_stream = timeline.pts_stream[selected_indices]
    source_timestamps_ns = timeline.timestamps_ns[selected_indices]
    raster = make_raster_transform(
        timeline.source_width,
        timeline.source_height,
        rotation_degrees_clockwise=rotation_degrees_clockwise,
        grid=grid,
    )
    frames, decoder_backend = _decode_selected_frames(
        source,
        stream_index=stream_index,
        selected_pts_stream=selected_pts_stream,
        raster=raster,
        grid=grid,
    )
    return DecodedAnnotationClip(
        frames=frames,
        timestamps_ns=np.ascontiguousarray(timestamps_ns, dtype=np.int64),
        source_timestamps_ns=np.ascontiguousarray(source_timestamps_ns, dtype=np.int64),
        source_span=source_span,
        raster=raster,
        decoder_backend=decoder_backend,
    )


def make_raster_transform(
    source_width: int,
    source_height: int,
    *,
    rotation_degrees_clockwise: int,
    grid: AnnotationGrid,
) -> RasterTransform:
    """Build the center-crop transform used by the streaming frame converter."""
    if source_width <= 0 or source_height <= 0:
        message = "source dimensions must be positive"
        raise ValueError(message)
    if rotation_degrees_clockwise not in {0, 90, 180, 270}:
        message = "rotation_degrees_clockwise must be one of 0, 90, 180, or 270"
        raise ValueError(message)
    if rotation_degrees_clockwise in {90, 270}:
        oriented_width, oriented_height = source_height, source_width
    else:
        oriented_width, oriented_height = source_width, source_height
    crop_left, crop_top, crop_width, crop_height = _center_crop(
        oriented_width,
        oriented_height,
        grid.width,
        grid.height,
    )
    scale_x = crop_width / grid.width
    scale_y = crop_height / grid.height
    grid_to_oriented = (
        (scale_x, 0.0, crop_left + (scale_x - 1.0) / 2.0),
        (0.0, scale_y, crop_top + (scale_y - 1.0) / 2.0),
        (0.0, 0.0, 1.0),
    )
    oriented_to_source = _oriented_to_source_matrix(
        source_width,
        source_height,
        rotation_degrees_clockwise,
    )
    return RasterTransform(
        source_width=source_width,
        source_height=source_height,
        rotation_degrees_clockwise=rotation_degrees_clockwise,
        oriented_width=oriented_width,
        oriented_height=oriented_height,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_width=crop_width,
        crop_height=crop_height,
        grid_to_oriented_source=grid_to_oriented,
        grid_to_source=_matrix_multiply(oriented_to_source, grid_to_oriented),
    )


def validate_decoded_annotation_clip(
    decoded: DecodedAnnotationClip,
    *,
    grid: AnnotationGrid,
    min_frames: int,
    max_frames: int,
    consumer_name: str,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    tuple[float, float],
]:
    """Validate the shared decoder boundary once for both estimator stages."""
    frames = decoded.frames
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != _VIDEO_NDIM
        or frames.shape[1:] != (grid.height, grid.width, _RGB_CHANNELS)
    ):
        message = f"decoded {consumer_name} frames must be uint8 [T,{grid.height},{grid.width},3] RGB"
        raise ValueError(message)
    frame_count = len(frames)
    source_span = normalize_span(decoded.source_span)
    if source_span is None:
        message = "decoded source_span must not be empty"
        raise ValueError(message)
    _validate_frame_count(
        frame_count,
        min_frames=min_frames,
        max_frames=max_frames,
        source_span=source_span,
        consumer_name=consumer_name,
    )
    timestamps_ns = _validated_timestamp_array(
        decoded.timestamps_ns,
        frame_count=frame_count,
        field_name="timestamps_ns",
        strictly_increasing=True,
    )
    start_ns = round(source_span[0] * _NANOSECONDS_PER_SECOND)
    stop_ns = round(source_span[1] * _NANOSECONDS_PER_SECOND)
    _, _, expected_timestamps_ns = make_ts_grid(
        start_ns,
        exclusive_end_ns=stop_ns,
        sample_rate_hz=grid.sample_fps,
    )
    if not np.array_equal(timestamps_ns, expected_timestamps_ns):
        message = "decoded timestamps_ns do not match the configured regular annotation grid"
        raise ValueError(message)
    source_timestamps_ns = _validated_timestamp_array(
        decoded.source_timestamps_ns,
        frame_count=frame_count,
        field_name="source_timestamps_ns",
        strictly_increasing=False,
    )
    if bool(np.any(source_timestamps_ns < start_ns)) or bool(np.any(source_timestamps_ns >= stop_ns)):
        message = "decoded source_timestamps_ns must remain inside the half-open source span"
        raise ValueError(message)
    expected_raster = make_raster_transform(
        decoded.raster.source_width,
        decoded.raster.source_height,
        rotation_degrees_clockwise=decoded.raster.rotation_degrees_clockwise,
        grid=grid,
    )
    if decoded.raster != expected_raster:
        message = "decoded raster does not match the configured annotation grid"
        raise ValueError(message)
    if not isinstance(decoded.decoder_backend, str) or not decoded.decoder_backend.strip():
        message = "decoded decoder_backend must be a non-empty string"
        raise ValueError(message)
    return (
        np.ascontiguousarray(frames),
        np.ascontiguousarray(timestamps_ns),
        np.ascontiguousarray(source_timestamps_ns),
        source_span,
    )


def annotation_grid_configuration_matches(value: object, grid: AnnotationGrid) -> bool:
    """Return whether stored metadata has this grid's semantic configuration."""
    if not isinstance(value, Mapping):
        return False
    expected = grid.configuration_metadata()
    return all(value.get(key) == expected_value for key, expected_value in expected.items())


def annotation_grid_frame_count(
    span: tuple[float, float],
    grid: AnnotationGrid,
) -> int:
    """Return the number of regular half-open timestamps in one source span."""
    normalized_span = normalize_span(span)
    assert normalized_span is not None
    start_ns = round(normalized_span[0] * _NANOSECONDS_PER_SECOND)
    stop_ns = round(normalized_span[1] * _NANOSECONDS_PER_SECOND)
    _, _, timestamps_ns = make_ts_grid(
        start_ns,
        exclusive_end_ns=stop_ns,
        sample_rate_hz=grid.sample_fps,
    )
    return len(timestamps_ns)


def _probe_packet_timeline(source: Path, *, stream_index: int) -> _VideoTimeline:
    with av.open(str(source)) as container:
        stream = _video_stream(container, stream_index)
        assert stream.time_base is not None
        time_base = Fraction(stream.time_base)
        rows = [
            (
                int(packet.pts),
                0 if packet.duration is None else int(packet.duration),
            )
            for packet in container.demux(stream)
            if packet.pts is not None and not packet.is_discard
        ]
        source_width, source_height, nominal_fps = _stream_scalars(stream)
        stream_start_pts = stream.start_time
        stream_duration_pts = stream.duration
        declared_frame_count = int(stream.frames)
    if declared_frame_count > 0 and len(rows) != declared_frame_count:
        message = (
            f"packet timeline has {len(rows)} timestamps but the stream declares {declared_frame_count} display frames"
        )
        raise _PacketTimelineMismatchError(message)
    return _make_timeline(
        rows,
        time_base=time_base,
        source_width=source_width,
        source_height=source_height,
        nominal_fps=nominal_fps,
        stream_start_pts=stream_start_pts,
        stream_duration_pts=stream_duration_pts,
        mismatch_error=_PacketTimelineMismatchError,
    )


def _probe_decoded_timeline(source: Path, *, stream_index: int) -> _VideoTimeline:
    with av.open(str(source)) as container:
        stream = _video_stream(container, stream_index)
        assert stream.time_base is not None
        time_base = Fraction(stream.time_base)
        rows = [
            (
                int(frame.pts),
                0 if frame.duration is None else int(frame.duration),
            )
            for frame in container.decode(stream)
            if frame.pts is not None
        ]
        source_width, source_height, nominal_fps = _stream_scalars(stream)
        stream_start_pts = stream.start_time
        stream_duration_pts = stream.duration
    return _make_timeline(
        rows,
        time_base=time_base,
        source_width=source_width,
        source_height=source_height,
        nominal_fps=nominal_fps,
        stream_start_pts=stream_start_pts,
        stream_duration_pts=stream_duration_pts,
        mismatch_error=ValueError,
    )


def _video_stream(
    container: av.container.InputContainer,
    stream_index: int,
) -> av.video.stream.VideoStream:
    try:
        stream = container.streams.video[stream_index]
    except IndexError as error:
        message = f"video stream_index={stream_index} does not exist"
        raise ValueError(message) from error
    if stream.time_base is None:
        message = f"video stream_index={stream_index} has no time base"
        raise ValueError(message)
    return stream


def _stream_scalars(
    stream: av.video.stream.VideoStream,
) -> tuple[int, int, float | None]:
    nominal_fps = (
        float(stream.average_rate) if stream.average_rate is not None and float(stream.average_rate) > 0 else None
    )
    return int(stream.width), int(stream.height), nominal_fps


def _make_timeline(  # noqa: PLR0913
    rows: list[tuple[int, int]],
    *,
    time_base: Fraction,
    source_width: int,
    source_height: int,
    nominal_fps: float | None,
    stream_start_pts: int | None,
    stream_duration_pts: int | None,
    mismatch_error: type[Exception],
) -> _VideoTimeline:
    if not rows:
        message = "source video contains no displayable frame timestamps"
        raise mismatch_error(message)
    rows.sort()
    pts_array = np.asarray([pts for pts, _ in rows], dtype=np.int64)
    if len(pts_array) > 1 and bool(np.any(np.diff(pts_array) <= 0)):
        message = "source video presentation timestamps must be strictly increasing"
        raise mismatch_error(message)
    origin_pts = int(pts_array[0])
    timestamps_ns = np.asarray(
        [round(Fraction(int(pts) - origin_pts) * time_base * _NANOSECONDS_PER_SECOND) for pts in pts_array],
        dtype=np.int64,
    )
    if len(timestamps_ns) > 1 and bool(np.any(np.diff(timestamps_ns) <= 0)):
        message = "source timestamps collapse after conversion to nanoseconds"
        raise mismatch_error(message)
    duration_ns = _header_or_packet_duration_ns(
        rows,
        origin_pts=origin_pts,
        time_base=time_base,
        stream_start_pts=stream_start_pts,
        stream_duration_pts=stream_duration_pts,
        final_timestamp_ns=int(timestamps_ns[-1]),
    )
    return _VideoTimeline(
        pts_stream=pts_array,
        timestamps_ns=timestamps_ns,
        source_width=source_width,
        source_height=source_height,
        nominal_fps=nominal_fps,
        duration_ns=duration_ns,
    )


def _header_or_packet_duration_ns(  # noqa: PLR0913
    rows: list[tuple[int, int]],
    *,
    origin_pts: int,
    time_base: Fraction,
    stream_start_pts: int | None,
    stream_duration_pts: int | None,
    final_timestamp_ns: int,
) -> int | None:
    final_packet_duration = rows[-1][1]
    if final_packet_duration > 0:
        packet_duration_ns = final_timestamp_ns + round(
            Fraction(final_packet_duration) * time_base * _NANOSECONDS_PER_SECOND
        )
        if packet_duration_ns > final_timestamp_ns:
            return packet_duration_ns
    if stream_duration_pts is not None and stream_duration_pts > 0:
        start_pts = origin_pts if stream_start_pts is None else stream_start_pts
        header_duration_ns = round(
            Fraction(start_pts + stream_duration_pts - origin_pts) * time_base * _NANOSECONDS_PER_SECOND
        )
        if header_duration_ns > final_timestamp_ns:
            return header_duration_ns
    return None


def _timeline_duration_ns(timeline: _VideoTimeline, *, sample_fps: float) -> int:
    if timeline.duration_ns is not None:
        return timeline.duration_ns
    differences = np.diff(timeline.timestamps_ns)
    if len(differences):
        period_ns = int(np.median(differences))
    elif timeline.nominal_fps is not None:
        period_ns = round(_NANOSECONDS_PER_SECOND / timeline.nominal_fps)
    else:
        period_ns = round(_NANOSECONDS_PER_SECOND / sample_fps)
    return int(timeline.timestamps_ns[-1]) + max(1, period_ns)


def _resolve_span(
    span: tuple[float, float] | None,
    source_duration_ns: int,
) -> tuple[float, float]:
    source_duration = source_duration_ns / _NANOSECONDS_PER_SECOND
    if span is None:
        return 0.0, source_duration
    normalized = normalize_span(span)
    assert normalized is not None
    start, stop = normalized
    tolerance = 1.0 / _NANOSECONDS_PER_SECOND
    if start >= source_duration or stop > source_duration + tolerance:
        message = f"clip span {span} exceeds source duration {source_duration:.9f} seconds"
        raise ValueError(message)
    return start, min(stop, source_duration)


def _select_nearest_source_indices(
    source_timestamps_ns: npt.NDArray[np.int64],
    target_timestamps_ns: npt.NDArray[np.int64],
    *,
    start_ns: int,
    stop_ns: int,
) -> npt.NDArray[np.int64]:
    frame_start = int(np.searchsorted(source_timestamps_ns, start_ns, side="left"))
    frame_stop = int(np.searchsorted(source_timestamps_ns, stop_ns, side="left"))
    candidates = source_timestamps_ns[frame_start:frame_stop]
    if len(candidates) == 0:
        message = "source span contains no displayable frames"
        raise ValueError(message)
    right = np.searchsorted(candidates, target_timestamps_ns, side="left")
    right = np.clip(right, 0, len(candidates) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(candidates[right] - target_timestamps_ns) < np.abs(target_timestamps_ns - candidates[left])
    selected = np.where(choose_right, right, left)
    return np.ascontiguousarray(selected + frame_start, dtype=np.int64)


def _decode_selected_frames(
    source: Path,
    *,
    stream_index: int,
    selected_pts_stream: npt.NDArray[np.int64],
    raster: RasterTransform,
    grid: AnnotationGrid,
) -> tuple[npt.NDArray[np.uint8], str]:
    first_pts = int(selected_pts_stream[0])
    try:
        return (
            _decode_selected_frames_once(
                source,
                stream_index=stream_index,
                selected_pts_stream=selected_pts_stream,
                raster=raster,
                grid=grid,
                seek=True,
            ),
            "pyav_seek_grid",
        )
    except (av.FFmpegError, _SelectedFrameDecodeError) as error:
        seek_error_message = str(error)
    try:
        return (
            _decode_selected_frames_once(
                source,
                stream_index=stream_index,
                selected_pts_stream=selected_pts_stream,
                raster=raster,
                grid=grid,
                seek=False,
            ),
            "pyav_sequential_grid",
        )
    except (av.FFmpegError, _SelectedFrameDecodeError) as error:
        sequential_error_message = str(error)
    message = (
        f"PyAV could not decode selected frame PTS starting at {first_pts}: "
        f"seek={seek_error_message}; sequential={sequential_error_message}"
    )
    raise _PacketTimelineMismatchError(message) from None


def _decode_selected_frames_once(  # noqa: PLR0913
    source: Path,
    *,
    stream_index: int,
    selected_pts_stream: npt.NDArray[np.int64],
    raster: RasterTransform,
    grid: AnnotationGrid,
    seek: bool,
) -> npt.NDArray[np.uint8]:
    output_indices_by_pts: dict[int, list[int]] = {}
    for output_index, pts in enumerate(selected_pts_stream):
        output_indices_by_pts.setdefault(int(pts), []).append(output_index)
    frames = np.empty(
        (len(selected_pts_stream), grid.height, grid.width, _RGB_CHANNELS),
        dtype=np.uint8,
    )
    with av.open(str(source)) as container:
        stream = container.streams.video[stream_index]
        stream.thread_type = "AUTO"
        if seek:
            container.seek(
                min(output_indices_by_pts),
                stream=stream,
                any_frame=False,
                backward=True,
            )
        remaining = set(output_indices_by_pts)
        for frame in container.decode(stream):
            if frame.pts is None or int(frame.pts) not in remaining:
                continue
            pts = int(frame.pts)
            converted = _convert_frame(frame, raster=raster, grid=grid)
            frames[output_indices_by_pts[pts]] = converted
            remaining.remove(pts)
            if not remaining:
                break
    if remaining:
        missing = sorted(remaining)
        message = f"decoder did not return {len(missing)} selected frame PTS: {missing[:5]}"
        raise _SelectedFrameDecodeError(message)
    return np.ascontiguousarray(frames)


def _convert_frame(
    frame: av.VideoFrame,
    *,
    raster: RasterTransform,
    grid: AnnotationGrid,
) -> npt.NDArray[np.uint8]:
    array = np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
    expected_shape = (raster.source_height, raster.source_width, _RGB_CHANNELS)
    if array.shape != expected_shape:
        message = f"decoded frame shape {array.shape} does not match stream shape {expected_shape}"
        raise ValueError(message)
    if raster.rotation_degrees_clockwise:
        array = np.rot90(
            array,
            k=-(raster.rotation_degrees_clockwise // 90),
            axes=(0, 1),
        )
        array = np.ascontiguousarray(array)
    box = (
        raster.crop_left,
        raster.crop_top,
        raster.crop_left + raster.crop_width,
        raster.crop_top + raster.crop_height,
    )
    image = Image.fromarray(array)
    resized = image.resize(
        (grid.width, grid.height),
        resample=Image.Resampling.BILINEAR,
        box=box,
    )
    return np.asarray(resized, dtype=np.uint8).copy()


def _center_crop(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    if source_width * target_height >= source_height * target_width:
        crop_height = source_height
        crop_width = min(
            source_width,
            max(
                1,
                (source_height * target_width + target_height // 2) // target_height,
            ),
        )
    else:
        crop_width = source_width
        crop_height = min(
            source_height,
            max(
                1,
                (source_width * target_height + target_width // 2) // target_width,
            ),
        )
    return (
        (source_width - crop_width) // 2,
        (source_height - crop_height) // 2,
        crop_width,
        crop_height,
    )


def _oriented_to_source_matrix(
    source_width: int,
    source_height: int,
    rotation_degrees_clockwise: int,
) -> tuple[tuple[float, float, float], ...]:
    if rotation_degrees_clockwise == 0:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if rotation_degrees_clockwise == _ROTATION_90:
        return (
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, float(source_height - 1)),
            (0.0, 0.0, 1.0),
        )
    if rotation_degrees_clockwise == _ROTATION_180:
        return (
            (-1.0, 0.0, float(source_width - 1)),
            (0.0, -1.0, float(source_height - 1)),
            (0.0, 0.0, 1.0),
        )
    return (
        (0.0, -1.0, float(source_width - 1)),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _matrix_multiply(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    product = np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)
    return cast(
        "tuple[tuple[float, float, float], ...]",
        tuple(tuple(float(value) for value in row) for row in product),
    )


def _validated_timestamp_array(
    value: object,
    *,
    frame_count: int,
    field_name: str,
    strictly_increasing: bool,
) -> npt.NDArray[np.int64]:
    if not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.ndim != 1 or len(value) != frame_count:
        message = f"decoded {field_name} must be int64 [T] aligned with {frame_count} frames"
        raise ValueError(message)
    differences = np.diff(value)
    if len(value) and int(value[0]) < 0:
        message = f"decoded {field_name} must be non-negative"
        raise ValueError(message)
    if strictly_increasing and bool(np.any(differences <= 0)):
        message = f"decoded {field_name} must be strictly increasing"
        raise ValueError(message)
    if not strictly_increasing and bool(np.any(differences < 0)):
        message = f"decoded {field_name} must be non-decreasing"
        raise ValueError(message)
    return value


def _validate_frame_count(
    frame_count: int,
    *,
    min_frames: int,
    max_frames: int,
    source_span: tuple[float, float],
    consumer_name: str = "annotation model",
) -> None:
    if frame_count < min_frames:
        message = f"{consumer_name} requires at least {min_frames} grid frames, got {frame_count} in span {source_span}"
        raise ValueError(message)
    if frame_count > max_frames:
        message = (
            f"{consumer_name} requires a bounded full clip: span {source_span} "
            f"contains {frame_count} grid frames, max_frames={max_frames}; "
            "split the source span upstream"
        )
        raise ValueError(message)


__all__ = [
    "DEFAULT_ANNOTATION_GRID",
    "AnnotationClipDecoder",
    "AnnotationGrid",
    "DecodedAnnotationClip",
    "RasterTransform",
    "annotation_grid_configuration_matches",
    "annotation_grid_frame_count",
    "decode_annotation_clip",
    "make_raster_transform",
    "validate_decoded_annotation_clip",
]
