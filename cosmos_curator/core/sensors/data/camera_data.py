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

"""Camera data structures for cosmos_curator.core.sensors package."""

from typing import Any, Protocol, Self, cast, get_args

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.intrinsics import CameraIntrinsics
from cosmos_curator.core.sensors.data.video import VideoMetadata
from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    nondecreasing_int64_array,
    strictly_increasing_int64_array,
    uint8_frame_batch,
)

_CAMERA_FRAME_NDIM = 4
_RGB_CHANNELS = 3


class _HasCameraFrames(Protocol):
    frames: npt.NDArray[np.uint8]


class _HasCameraBatchFields(_HasCameraFrames, Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    pts_stream: npt.NDArray[np.int64]


class _HasMetadata(Protocol):
    metadata: VideoMetadata


class _HasArrayAttribute(Protocol):
    name: str
    type: Any


def _dtype_from_ndarray_annotation(attribute: _HasArrayAttribute) -> np.dtype[Any]:
    """Return the scalar dtype declared by a ``npt.NDArray[...]`` field annotation."""
    _, dtype_annotation = get_args(attribute.type)
    dtype_arg = get_args(dtype_annotation)[0]
    return np.dtype(cast("type[np.generic]", dtype_arg))


def _motion_vector_array(
    _instance: object,
    attribute: _HasArrayAttribute,
    value: npt.NDArray[Any],
) -> None:
    """Validate one named motion-vector field array."""
    attribute_name = attribute.name
    expected_dtype = _dtype_from_ndarray_annotation(attribute)
    if value.ndim != 1:
        msg = f"{attribute_name} must be 1-D, got ndim={value.ndim}"
        raise ValueError(msg)
    if value.dtype != expected_dtype:
        msg = f"{attribute_name} must have dtype {expected_dtype}, got {value.dtype}"
        raise ValueError(msg)


def _motion_vector_frame_lengths(instance: "MotionVectorFrameData") -> None:
    """Validate all arrays in one motion-vector frame have equal length."""
    lengths = {field.name: len(getattr(instance, field.name)) for field in attrs.fields(type(instance))}
    if len(set(lengths.values())) != 1:
        msg = "all motion-vector fields must have equal length: " + ", ".join(
            f"{name}={length}" for name, length in lengths.items()
        )
        raise ValueError(msg)


def _positive_motion_vector_fields(instance: "MotionVectorFrameData") -> None:
    """Validate positive block dimensions and sub-pixel motion scale."""
    for field_name in ("w", "h", "motion_scale"):
        values = getattr(instance, field_name)
        if np.any(values <= 0):
            msg = f"{field_name} must contain strictly positive values"
            raise ValueError(msg)


def _motion_vectors(
    instance: _HasCameraFrames,
    _attribute: object,
    value: "MotionVectorData | None",
) -> None:
    """Validate optional motion-vector payload length against the RGB frame batch."""
    if value is None:
        return
    if len(value.frames) != len(instance.frames):
        error_msg = f"motion_vectors.frames length {len(value.frames)} != frames length {len(instance.frames)}"
        raise ValueError(error_msg)


def _batch_lengths(
    instance: _HasCameraBatchFields,
    _attribute: object,
    _value: VideoMetadata,
) -> None:
    """Validate shared row-count invariants across camera batch arrays."""
    if not (
        len(instance.align_timestamps_ns)
        == len(instance.sensor_timestamps_ns)
        == len(instance.pts_stream)
        == len(instance.frames)
    ):
        error_msg = (
            "All arrays must be the same length: "
            f"align_timestamps_ns={len(instance.align_timestamps_ns)} "
            f"sensor_timestamps_ns={len(instance.sensor_timestamps_ns)} "
            f"pts_stream={len(instance.pts_stream)} "
            f"frames={len(instance.frames)}"
        )
        raise ValueError(error_msg)


def _metadata_shape(
    instance: _HasCameraFrames,
    _attribute: object,
    value: VideoMetadata,
) -> None:
    """Validate frame geometry against metadata dimensions."""
    expected_shape = (value.height, value.width, _RGB_CHANNELS)
    if instance.frames.shape[1:] != expected_shape:
        msg = f"frames must have shape (N, {value.height}, {value.width}, {_RGB_CHANNELS}), got {instance.frames.shape}"
        raise ValueError(msg)


def _intrinsics_dimensions_match_metadata(
    instance: _HasMetadata,
    _attribute: object,
    value: CameraIntrinsics | None,
) -> None:
    """Validate optional intrinsics dimensions against the batch metadata."""
    if value is None:
        return
    metadata = instance.metadata
    if value.width != metadata.width or value.height != metadata.height:
        msg = (
            f"CameraIntrinsics dimensions ({value.width}x{value.height}) do not match "
            f"VideoMetadata dimensions ({metadata.width}x{metadata.height}). "
            "This may indicate either rig calibration errors or video encoding errors."
        )
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class MotionVectorFrameData:
    """Lossless named motion-vector side data for one decoded video frame.

    PyAV 17.0.0 exposes FFmpeg ``AVMotionVector`` side data as a structured
    NumPy array with fields ``source``, ``w``, ``h``, ``src_x``, ``src_y``,
    ``dst_x``, ``dst_y``, ``flags``, ``motion_x``, ``motion_y``, and
    ``motion_scale``. The sensor-layer contract normalizes those mixed-width
    integer fields into this stable schema: every field is ``int32`` except
    ``flags``, which is ``int64``. No field is dropped.
    """

    __hash__ = None  # type: ignore[assignment]

    source: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    w: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    h: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    src_x: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    src_y: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    dst_x: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    dst_y: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    flags: npt.NDArray[np.int64] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    motion_x: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    motion_y: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)
    motion_scale: npt.NDArray[np.int32] = attrs.field(converter=as_readonly_view, validator=_motion_vector_array)

    def __attrs_post_init__(self) -> None:
        """Validate cross-field frame invariants."""
        _motion_vector_frame_lengths(self)
        _positive_motion_vector_fields(self)

    @classmethod
    def empty(cls) -> Self:
        """Return an empty, well-typed motion-vector payload for one frame."""
        return cls(
            source=np.empty(0, dtype=np.int32),
            w=np.empty(0, dtype=np.int32),
            h=np.empty(0, dtype=np.int32),
            src_x=np.empty(0, dtype=np.int32),
            src_y=np.empty(0, dtype=np.int32),
            dst_x=np.empty(0, dtype=np.int32),
            dst_y=np.empty(0, dtype=np.int32),
            flags=np.empty(0, dtype=np.int64),
            motion_x=np.empty(0, dtype=np.int32),
            motion_y=np.empty(0, dtype=np.int32),
            motion_scale=np.empty(0, dtype=np.int32),
        )


def _motion_vector_frames(
    _instance: object,
    _attribute: object,
    value: tuple[object, ...],
) -> None:
    """Validate that a motion-vector batch contains per-frame containers."""
    for index, frame in enumerate(value):
        if not isinstance(frame, MotionVectorFrameData):
            msg = f"frames[{index}] must be MotionVectorFrameData, got {type(frame).__name__}"
            raise TypeError(msg)


@attrs.define(hash=False, frozen=True)
class MotionVectorData:
    """Motion-vector batch aligned one-to-one with decoded RGB frames.

    ``frames[i]`` contains the named motion-vector payload for
    ``CameraData.frames[i]``. This is the canonical sensor-library
    representation: it preserves all PyAV/FFmpeg fields as named integer
    arrays, including ``source``.

    This intentionally replaces the older positional ``float64`` matrix shape
    ``(N, 10)``.
    """

    __hash__ = None  # type: ignore[assignment]

    frames: tuple[MotionVectorFrameData, ...] = attrs.field(converter=tuple, validator=_motion_vector_frames)


@attrs.define(hash=False, frozen=True)
class CameraData:
    """Decoded RGB video: ``N`` frames, with row ``i`` indexing the same moment across arrays.

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).

    Attributes:
        align_timestamps_ns: 1-D alignment timeline (ns) each sample row is aligned to; length ``N``,
            row ``i`` with ``frames[i]``
        sensor_timestamps_ns: 1-D sensor-reported times (ns); may differ from ``align_timestamps_ns``
            (resampling/grid); length ``N``
        pts_stream: 1-D presentation timestamps in producer-specific int units, length ``N``;
            ``CameraSensor`` uses stream-native ``time_base`` for exact seeks; ``McapCameraSensor``
            uses nanoseconds matching ``sensor_timestamps_ns`` (example sensor only; production
            ``pts_stream`` is expected to follow the ``CameraSensor`` contract)
        frames: decoded RGB, shape ``(N, H, W, 3)``, ``uint8``; row ``i`` is the image at index ``i``
        metadata: stream geometry and related fields (``VideoMetadata``)
        motion_vectors: optional per-frame motion vectors; when set, length ``N`` matches ``frames``
        intrinsics: optional typed camera calibration for the same image geometry as ``metadata``
        extrinsics: optional rigid transform from the camera frame to a caller-defined reference frame

    """

    __hash__ = None  # type: ignore[assignment]
    align_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=strictly_increasing_int64_array,
    )
    sensor_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=nondecreasing_int64_array,
    )
    pts_stream: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=nondecreasing_int64_array,
    )
    frames: npt.NDArray[np.uint8] = attrs.field(
        converter=as_readonly_view,
        validator=uint8_frame_batch,
    )
    # Attach batch-length validation to the last required field so all batch
    # arrays are already set when attrs runs this validator.
    metadata: VideoMetadata = attrs.field(
        validator=attrs.validators.and_(
            _batch_lengths,
            _metadata_shape,
        )
    )
    # Optional; requires decoder with export_mvs
    motion_vectors: MotionVectorData | None = attrs.field(
        default=None,
        validator=_motion_vectors,
    )
    intrinsics: CameraIntrinsics | None = attrs.field(
        default=None,
        validator=_intrinsics_dimensions_match_metadata,
    )
    extrinsics: SensorExtrinsics | None = attrs.field(default=None)
