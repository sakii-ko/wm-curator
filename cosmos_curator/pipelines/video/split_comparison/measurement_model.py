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
"""Measurement-row model for split output comparison.

The measure phase emits one tidy/long table conforming to
:data:`MEASUREMENT_SCHEMA`: every comparable element of the clip metadata
becomes one row whose ``value`` is the comparison outcome (absolute diff for
tolerance fields, cosine similarity for captions, ``1.0``/``0.0`` for
equality). Per-side presence/corruption is carried in four booleans. Construct
rows via :func:`make_measurement` so the value/presence invariants are enforced
at the call site. The evaluate phase applies thresholds; the mode
of each ``measurement_type`` is declared in ``field_spec`` and classified by
:class:`MeasurementMode`. See ``docs/curator/design/split-comparison.md``.
"""

import types
from enum import Enum, auto
from typing import Any, TypedDict, Union, get_args, get_origin, get_type_hints

import pyarrow as pa


class Measurement(TypedDict):
    """Row shape for the measurements table, and the source of truth for :data:`MEASUREMENT_SCHEMA`.

    The Arrow schema is derived from these annotations, so this declaration is
    the one place a column's name, order, type, and nullability live. Carries no
    methods; the Arrow table is the canonical representation. Build rows via
    :func:`make_measurement` so the presence/value invariants hold.

    ``total=True`` (the default): every row carries all ten keys -- ``make_measurement``
    always populates them. ``window_id``/``model``/``value`` are nullable in *value*
    (``| None``), not optional in *presence*.
    """

    video_key: str
    clip_id: str
    window_id: str | None  # null for clip-level rows; "{start_frame}_{end_frame}" otherwise
    model: str | None  # null for non-model measurements; caption/token model otherwise
    measurement_type: str
    value: float | None  # null unless both sides present and neither corrupt
    output_a_present: bool
    output_b_present: bool
    output_a_corrupt: bool
    output_b_corrupt: bool


# Arrow type per scalar Python type used in Measurement. int/float map to the
# 64-bit Arrow types; bool is keyed exactly (not via its int subclassing).
_PY_TO_ARROW: dict[type, pa.DataType] = {
    str: pa.string(),
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
}


def _arrow_type(annotation: Any) -> pa.DataType:  # noqa: ANN401 -- receives a type annotation object
    """Arrow type for one :class:`Measurement` field annotation, unwrapping ``X | None``."""
    if get_origin(annotation) in (types.UnionType, Union):
        annotation = next(arg for arg in get_args(annotation) if arg is not type(None))
    return _PY_TO_ARROW[annotation]


# Derived from Measurement so the row type stays the single source of truth:
# add or reorder a field there and the schema follows. Every field is nullable
# (Arrow default); the value/presence invariants are enforced in make_measurement.
MEASUREMENT_SCHEMA: pa.Schema = pa.schema(
    [(name, _arrow_type(annotation)) for name, annotation in get_type_hints(Measurement).items()],
)


class MeasurementMode(Enum):
    """How a ``measurement_type``'s ``value`` is produced and later evaluated.

    Declared per type in ``field_spec``; the evaluate phase dispatches on it.
    The mode is *not* stored on the row -- ``measurement_type`` is the key and
    the spec maps it back to a mode.
    """

    TOLERANCE = auto()  # value = absolute difference; evaluate flags value > abs_tolerance
    SIMILARITY = auto()  # value = cosine similarity; evaluate flags value < min_similarity
    EQUALITY = auto()  # value = 1.0 (equal) / 0.0 (not equal); evaluate flags value == 0.0


def make_measurement(  # noqa: PLR0913 -- MEASUREMENT_SCHEMA has 10 columns; helper mirrors them as kwargs
    *,
    video_key: str,
    clip_id: str,
    measurement_type: str,
    output_a_present: bool,
    output_b_present: bool,
    output_a_corrupt: bool = False,
    output_b_corrupt: bool = False,
    value: float | None = None,
    window_id: str | None = None,
    model: str | None = None,
) -> Measurement:
    """Build a schema-compatible :class:`Measurement` row, enforcing the invariants.

    Invariants (raise :class:`ValueError` on violation):

    * ``corrupt`` implies ``present`` per side -- ``present=False, corrupt=True``
      is impossible (a side can't be corrupt without existing).
    * ``value`` is non-null **iff** the measurement is comparable, i.e. both
      sides present and neither corrupt. Comparable rows must carry a value;
      non-comparable rows must leave it null.
    """
    if output_a_corrupt and not output_a_present:
        msg = "output_a_corrupt set without output_a_present (corrupt implies present)"
        raise ValueError(msg)
    if output_b_corrupt and not output_b_present:
        msg = "output_b_corrupt set without output_b_present (corrupt implies present)"
        raise ValueError(msg)
    comparable = output_a_present and output_b_present and not output_a_corrupt and not output_b_corrupt
    if comparable and value is None:
        msg = f"comparable measurement {measurement_type!r} requires a non-null value"
        raise ValueError(msg)
    if not comparable and value is not None:
        msg = f"non-comparable measurement {measurement_type!r} must have a null value"
        raise ValueError(msg)
    return Measurement(
        video_key=video_key,
        clip_id=clip_id,
        window_id=window_id,
        model=model,
        measurement_type=measurement_type,
        value=value,
        output_a_present=output_a_present,
        output_b_present=output_b_present,
        output_a_corrupt=output_a_corrupt,
        output_b_corrupt=output_b_corrupt,
    )


def empty_measurements() -> pa.Table:
    """Return an empty measurements table that still carries :data:`MEASUREMENT_SCHEMA`."""
    return pa.Table.from_pylist([], schema=MEASUREMENT_SCHEMA)
