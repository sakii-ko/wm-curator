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
"""Tests for the declarative measurement catalog (field_spec)."""

from cosmos_curator.pipelines.video.split_comparison.field_spec import (
    ABSENT,
    FIELD_SPECS,
    MEMBERSHIP_TYPES,
    SPECS_BY_TYPE,
    Scope,
    specs_for_scope,
)
from cosmos_curator.pipelines.video.split_comparison.measurement_model import MeasurementMode


def test_catalog_has_no_duplicate_types() -> None:
    """No measurement_type appears twice in the catalog."""
    names = [spec.measurement_type for spec in FIELD_SPECS]
    assert len(names) == len(set(names))


def test_mode_is_consistent_with_naming() -> None:
    """Suffix-to-mode is consistent; membership types are the only non-suffixed equality rows."""
    for spec in FIELD_SPECS:
        name = spec.measurement_type
        if name in MEMBERSHIP_TYPES:
            assert spec.mode is MeasurementMode.EQUALITY
        elif name.endswith("_diff"):
            assert spec.mode is MeasurementMode.TOLERANCE
        elif name.endswith("_similarity"):
            assert spec.mode is MeasurementMode.SIMILARITY
        elif name.endswith("_equal"):
            assert spec.mode is MeasurementMode.EQUALITY
        else:
            msg = f"unclassified measurement_type: {name}"
            raise AssertionError(msg)


def test_specs_for_scope_filters() -> None:
    """specs_for_scope returns only that scope's entries."""
    assert all(spec.scope is Scope.FILTERED_WINDOW for spec in specs_for_scope(Scope.FILTERED_WINDOW))
    assert {spec.measurement_type for spec in specs_for_scope(Scope.FILTERED_WINDOW)} == {
        "filtered_window_errors_count_diff",
        "qwen_rejection_reasons_equal",
    }


def test_flat_accessor_reads_value_and_absent() -> None:
    """A flat-key accessor returns the value when present, ABSENT when missing."""
    acc = SPECS_BY_TYPE["aesthetic_score_diff"].accessor
    assert acc({"aesthetic_score": 4.5}, None, "") == 4.5
    assert acc({}, None, "") is ABSENT


def test_nested_accessor_reads_motion_fields() -> None:
    """The motion accessor reads the nested field and is ABSENT when motion_score is missing."""
    acc = SPECS_BY_TYPE["motion_global_mean_diff"].accessor
    assert acc({"motion_score": {"global_mean": 0.7}}, None, "") == 0.7
    assert acc({}, None, "") is ABSENT


def test_index_accessor_reads_duration_span_endpoints() -> None:
    """duration_span start/end accessors read element 0 and 1."""
    meta = {"duration_span": [1.5, 3.25]}
    assert SPECS_BY_TYPE["duration_span_start_diff"].accessor(meta, None, "") == 1.5
    assert SPECS_BY_TYPE["duration_span_end_diff"].accessor(meta, None, "") == 3.25


def test_count_accessor_covers_documented_cases() -> None:
    """Count accessor: absent key and explicit null count 0; collection -> size; non-collection -> ABSENT."""
    acc = SPECS_BY_TYPE["errors_count_diff"].accessor
    assert acc({}, None, "") == 0  # absent key -- missing == none recorded
    assert acc({"errors": None}, None, "") == 0  # explicit null collapses to 0, same as absent
    assert acc({"errors": ["a", "b"]}, None, "") == 2  # collection -> size
    assert acc({"errors": "not-a-list"}, None, "") is ABSENT  # present non-collection -> not comparable


def test_clip_location_accessor_strips_output_root() -> None:
    """clip_location is compared on the path after the per-side output root."""
    acc = SPECS_BY_TYPE["clip_location_equal"].accessor
    meta = {"clip_location": "s3://bucket/run_a/clips/v0/abc.mp4"}
    assert acc(meta, None, "s3://bucket/run_a/") == "clips/v0/abc.mp4"
    assert acc({}, None, "s3://bucket/run_a/") is ABSENT


def test_clip_location_accessor_strips_regardless_of_trailing_slash() -> None:
    """Strip is trailing-slash invariant, so roots typed differently still reduce alike.

    Otherwise two sides that typed the root with vs without a trailing slash would
    reduce to ``clips/...`` vs ``/clips/...`` and spuriously mismatch on every clip.
    """
    acc = SPECS_BY_TYPE["clip_location_equal"].accessor
    meta_a = {"clip_location": "s3://bucket/run_a/clips/v0/abc.mp4"}
    meta_b = {"clip_location": "s3://bucket/run_b/clips/v0/abc.mp4"}
    # output_a has a trailing slash, output_b does not -- both must reduce identically.
    assert acc(meta_a, None, "s3://bucket/run_a/") == "clips/v0/abc.mp4"
    assert acc(meta_b, None, "s3://bucket/run_b") == "clips/v0/abc.mp4"


def test_model_qualified_accessor_uses_model_name() -> None:
    """A model-qualified accessor keys the window field by the active model."""
    acc = SPECS_BY_TYPE["caption_similarity"].accessor
    assert acc({"qwen_caption": "a cat"}, "qwen", "") == "a cat"
    assert acc({"qwen_caption": "a cat"}, None, "") is ABSENT
    assert acc({}, "qwen", "") is ABSENT


def test_membership_accessor_marks_present_when_window_exists() -> None:
    """The window_present accessor returns a constant marker whenever the window dict exists."""
    acc = SPECS_BY_TYPE["window_present"].accessor
    marker = acc({"start_frame": 0}, None, "")
    assert marker is not ABSENT
    assert acc({"start_frame": 10}, None, "") == marker
