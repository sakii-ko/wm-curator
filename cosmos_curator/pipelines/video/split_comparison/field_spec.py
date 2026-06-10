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
"""Measurement catalog for the split-comparison measure phase.

The measure stage doesn't diff the two clip-metadata dicts directly -- it walks
this catalog, one :class:`FieldSpec` per thing worth comparing. Each spec says
what to compare and how:

- ``measurement_type`` -- the row label (e.g. ``aesthetic_score_diff``).
- ``mode`` -- how two values are compared (tolerance / similarity / equality).
- ``scope`` -- where the field lives (clip, caption window, filtered window),
  which sets how its rows are keyed.
- ``accessor`` -- reads the value off one side's metadata, returning
  :data:`ABSENT` when the field isn't present.
- ``model_qualified`` -- when set, the spec is expanded once per caption-*producer*
  model the measure phase is asked to compare (e.g. ``qwen``); the model name keys
  the per-model field (``{model}_caption``) via :func:`_model_key`. That producer
  list is a measure-phase input, not part of this catalog -- and distinct from the
  single embedding model in ``CaptionPolicy``, which only scores caption similarity.

Walking a fixed catalog -- rather than whatever keys the clip-metadata dict
happens to hold -- keeps the measurement types a closed, stable vocabulary, and
lets fields that legitimately differ be handled deliberately: ``clip_location``
is compared only after its per-side output root is stripped, and
``duration_span`` is split into start/end scalars. See
``docs/curator/design/split-comparison.md``.
"""

from collections.abc import Callable, Mapping, Sequence
from enum import Enum, auto
from typing import Any

import attrs

from cosmos_curator.pipelines.video.split_comparison.measurement_model import MeasurementMode


class Scope(Enum):
    """Where a measured field lives, which sets how its measurement rows are keyed."""

    CLIP = auto()  # clip-level field; window_id is null
    WINDOW = auto()  # per caption window; window_id = "{start_frame}_{end_frame}"
    FILTERED_WINDOW = auto()  # per filtered (rejected) window; window_id = "{start_frame}_{end_frame}"


class _Sentinel(Enum):
    ABSENT = auto()


# Returned by an accessor when the field is not present on that side. Distinct
# from any real value (including ``None`` / ``False`` / ``0`` / ``""``).
ABSENT = _Sentinel.ABSENT

# A membership accessor returns this whenever its scope exists on a side. Both
# sides present return the same marker, so the equality compare yields 1.0.
_MEMBERSHIP_MARKER = True

# (mapping, model, output_root) -> the field's value on that side, or ABSENT.
Accessor = Callable[[Mapping[str, Any], str | None, str], object]


@attrs.define(frozen=True)
class FieldSpec:
    """One catalog entry: a measurement type plus how to read and classify it."""

    measurement_type: str
    mode: MeasurementMode
    scope: Scope
    accessor: Accessor
    model_qualified: bool = False


def _key(name: str) -> Accessor:
    """Accessor for a flat top-level field."""

    def read(data: Mapping[str, Any], _model: str | None, _root: str) -> object:
        return data.get(name, ABSENT)

    return read


def _nested(outer: str, inner: str) -> Accessor:
    """Accessor for a value nested one level under ``outer`` (e.g. motion_score.global_mean)."""

    def read(data: Mapping[str, Any], _model: str | None, _root: str) -> object:
        nested = data.get(outer)
        if not isinstance(nested, Mapping):
            return ABSENT
        return nested.get(inner, ABSENT)

    return read


def _index(name: str, idx: int) -> Accessor:
    """Accessor for one element of a list/tuple field (e.g. duration_span[0])."""

    def read(data: Mapping[str, Any], _model: str | None, _root: str) -> object:
        seq = data.get(name)
        if not isinstance(seq, (list, tuple)) or len(seq) <= idx:
            return ABSENT
        return seq[idx]

    return read


def _count(name: str) -> Accessor:
    """Accessor that measures the size of a collection field.

    An absent key and an explicit ``null`` both count as ``0`` -- intentional:
    these fields (``errors``, ``qwen_rejection_reasons``) follow "missing ==
    none recorded", so the absent/null distinction is deliberately collapsed.
    A present non-collection value is ABSENT (not comparable).
    """

    def read(data: Mapping[str, Any], _model: str | None, _root: str) -> object:
        value = data.get(name)
        if value is None:  # absent key or explicit null -- both empty by design (see docstring)
            return 0
        if isinstance(value, (list, tuple, dict, set)):
            return len(value)
        return ABSENT  # present but not a collection -- treated as not comparable

    return read


def _model_key(suffix: str) -> Accessor:
    """Accessor for a per-model window field keyed ``{model}_{suffix}`` (e.g. qwen_caption)."""

    def read(data: Mapping[str, Any], model: str | None, _root: str) -> object:
        if model is None:
            return ABSENT
        return data.get(f"{model}_{suffix}", ABSENT)

    return read


def _clip_location() -> Accessor:
    """Accessor for ``clip_location`` with the per-side output root stripped before compare."""

    def read(data: Mapping[str, Any], _model: str | None, output_root: str) -> object:
        loc = data.get("clip_location", ABSENT)
        if not isinstance(loc, str):
            return ABSENT
        return loc.removeprefix(output_root.rstrip("/")).lstrip("/")

    return read


def _membership() -> Accessor:
    """Accessor for a membership row: present iff the scope-level mapping exists on the side."""

    def read(_data: Mapping[str, Any], _model: str | None, _root: str) -> object:
        return _MEMBERSHIP_MARKER

    return read


_CLIP_TOLERANCE: tuple[tuple[str, Accessor], ...] = (
    ("aesthetic_score_diff", _key("aesthetic_score")),
    ("motion_global_mean_diff", _nested("motion_score", "global_mean")),
    ("motion_per_patch_min_256_diff", _nested("motion_score", "per_patch_min_256")),
    ("duration_span_start_diff", _index("duration_span", 0)),
    ("duration_span_end_diff", _index("duration_span", 1)),
    ("framerate_source_diff", _key("framerate_source")),
    ("framerate_diff", _key("framerate")),
    ("num_bytes_diff", _key("num_bytes")),
    ("total_prompt_tokens_diff", _key("total_prompt_tokens")),
    ("total_output_tokens_diff", _key("total_output_tokens")),
    ("errors_count_diff", _count("errors")),
)

_CLIP_EQUALITY: tuple[tuple[str, Accessor], ...] = (
    ("span_uuid_equal", _key("span_uuid")),
    ("source_video_equal", _key("source_video")),
    ("width_source_equal", _key("width_source")),
    ("height_source_equal", _key("height_source")),
    ("width_equal", _key("width")),
    ("height_equal", _key("height")),
    ("num_frames_equal", _key("num_frames")),
    ("video_codec_equal", _key("video_codec")),
    ("qwen_type_classification_equal", _key("qwen_type_classification")),
    ("qwen_rejection_stage_equal", _key("qwen_rejection_stage")),
    ("post_production_text_equal", _key("post_production_text")),
    ("sam3_num_instances_equal", _key("sam3_num_instances")),
    ("sam3_num_events_equal", _key("sam3_num_events")),
    ("valid_equal", _key("valid")),
    ("has_caption_equal", _key("has_caption")),
    ("num_caption_windows_equal", _key("num_caption_windows")),
    ("caption_quality_flags_enabled_equal", _key("caption_quality_flags_enabled")),
    ("clip_location_equal", _clip_location()),
)

# Window entries carry, in order: measurement_type, mode, accessor, model_qualified.
_WINDOW: tuple[tuple[str, MeasurementMode, Accessor, bool], ...] = (
    ("caption_similarity", MeasurementMode.SIMILARITY, _model_key("caption"), True),
    ("enhanced_caption_similarity", MeasurementMode.SIMILARITY, _model_key("enhanced_caption"), True),
    ("prompt_tokens_diff", MeasurementMode.TOLERANCE, _model_key("prompt_tokens"), True),
    ("output_tokens_diff", MeasurementMode.TOLERANCE, _model_key("output_tokens"), True),
    ("caption_status_equal", MeasurementMode.EQUALITY, _key("caption_status"), False),
    ("caption_failure_reason_equal", MeasurementMode.EQUALITY, _key("caption_failure_reason"), False),
    ("flag_length_outlier_equal", MeasurementMode.EQUALITY, _key("flag_length_outlier"), False),
    ("flag_repetition_equal", MeasurementMode.EQUALITY, _key("flag_repetition"), False),
    ("flag_near_duplicate_equal", MeasurementMode.EQUALITY, _key("flag_near_duplicate"), False),
    ("window_present", MeasurementMode.EQUALITY, _membership(), False),
)

# Filtered-window entries carry, in order: measurement_type, mode, accessor.
_FILTERED_WINDOW: tuple[tuple[str, MeasurementMode, Accessor], ...] = (
    ("filtered_window_errors_count_diff", MeasurementMode.TOLERANCE, _count("errors")),
    # qwen_rejection_reasons is written as str(dict) by the filter stages, not a
    # collection, so it's compared for equality (like qwen_rejection_stage) rather
    # than counted -- _count would see a non-collection and always return ABSENT.
    ("qwen_rejection_reasons_equal", MeasurementMode.EQUALITY, _key("qwen_rejection_reasons")),
)

# Membership types compare presence, not a field value; they carry EQUALITY mode
# but do not follow the ``*_equal`` naming. Listed so consistency checks can
# exempt them.
MEMBERSHIP_TYPES: frozenset[str] = frozenset({"window_present"})

FIELD_SPECS: tuple[FieldSpec, ...] = (
    *(FieldSpec(name, MeasurementMode.TOLERANCE, Scope.CLIP, acc) for name, acc in _CLIP_TOLERANCE),
    *(FieldSpec(name, MeasurementMode.EQUALITY, Scope.CLIP, acc) for name, acc in _CLIP_EQUALITY),
    *(FieldSpec(name, mode, Scope.WINDOW, acc, model_qualified=mq) for name, mode, acc, mq in _WINDOW),
    *(FieldSpec(name, mode, Scope.FILTERED_WINDOW, acc) for name, mode, acc in _FILTERED_WINDOW),
)

SPECS_BY_TYPE: Mapping[str, FieldSpec] = {spec.measurement_type: spec for spec in FIELD_SPECS}

# Catalog partitioned by scope once at import. ``specs_for_scope`` is called once
# per caption/filtered window per clip in the measure hot loop, and the result
# is the same every time, so the scan + tuple build is precomputed here.
_SPECS_BY_SCOPE: Mapping[Scope, tuple[FieldSpec, ...]] = {
    scope: tuple(spec for spec in FIELD_SPECS if spec.scope is scope) for scope in Scope
}


def specs_for_scope(scope: Scope) -> Sequence[FieldSpec]:
    """Return the catalog entries for a given :class:`Scope`, in catalog order."""
    return _SPECS_BY_SCOPE[scope]
