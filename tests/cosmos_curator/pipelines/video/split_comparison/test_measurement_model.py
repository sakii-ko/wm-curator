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
"""Tests for the measurement-row model: MEASUREMENT_SCHEMA + make_measurement invariants."""

import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison.measurement_model import (
    MEASUREMENT_SCHEMA,
    MeasurementMode,
    empty_measurements,
    make_measurement,
)


def test_schema_has_expected_columns_and_types() -> None:
    """MEASUREMENT_SCHEMA carries exactly the ten expected columns and types."""
    fields = {field.name: field.type for field in MEASUREMENT_SCHEMA}
    assert fields == {
        "video_key": pa.string(),
        "clip_id": pa.string(),
        "window_id": pa.string(),
        "model": pa.string(),
        "measurement_type": pa.string(),
        "value": pa.float64(),
        "output_a_present": pa.bool_(),
        "output_b_present": pa.bool_(),
        "output_a_corrupt": pa.bool_(),
        "output_b_corrupt": pa.bool_(),
    }


def test_modes_present() -> None:
    """MeasurementMode covers tolerance, similarity, and equality."""
    assert {mode.name for mode in MeasurementMode} == {"TOLERANCE", "SIMILARITY", "EQUALITY"}


def test_comparable_measurement_carries_value_and_fits_schema() -> None:
    """A comparable measurement keeps its value and materializes under the schema."""
    row = make_measurement(
        video_key="v",
        clip_id="c",
        measurement_type="aesthetic_score_diff",
        output_a_present=True,
        output_b_present=True,
        value=0.25,
    )
    assert row["value"] == 0.25
    assert row["window_id"] is None
    assert row["model"] is None
    pa.Table.from_pylist([row], schema=MEASUREMENT_SCHEMA)  # must not raise


def test_comparable_requires_value() -> None:
    """A comparable measurement with no value is rejected."""
    with pytest.raises(ValueError, match="value"):
        make_measurement(
            video_key="v",
            clip_id="c",
            measurement_type="aesthetic_score_diff",
            output_a_present=True,
            output_b_present=True,
            value=None,
        )


def test_non_comparable_rejects_value() -> None:
    """A non-comparable measurement carrying a value is rejected."""
    with pytest.raises(ValueError, match="value"):
        make_measurement(
            video_key="v",
            clip_id="c",
            measurement_type="aesthetic_score_diff",
            output_a_present=True,
            output_b_present=False,
            value=0.5,
        )


def test_one_sided_has_null_value() -> None:
    """A one-sided measurement has a null value and records the absent side."""
    row = make_measurement(
        video_key="v",
        clip_id="c",
        measurement_type="aesthetic_score_diff",
        output_a_present=True,
        output_b_present=False,
    )
    assert row["value"] is None
    assert row["output_b_present"] is False
    pa.Table.from_pylist([row], schema=MEASUREMENT_SCHEMA)


def test_corrupt_implies_present() -> None:
    """Marking a side corrupt without present is rejected."""
    with pytest.raises(ValueError, match="corrupt"):
        make_measurement(
            video_key="v",
            clip_id="c",
            measurement_type="aesthetic_score_diff",
            output_a_present=False,
            output_b_present=True,
            output_a_corrupt=True,
        )


def test_corrupt_side_makes_value_null() -> None:
    """A corrupt side makes the row non-comparable with a null value."""
    row = make_measurement(
        video_key="v",
        clip_id="c",
        measurement_type="aesthetic_score_diff",
        output_a_present=True,
        output_b_present=True,
        output_a_corrupt=True,
    )
    assert row["value"] is None
    pa.Table.from_pylist([row], schema=MEASUREMENT_SCHEMA)


def test_equality_value_round_trips() -> None:
    """Equality measurements store 1.0/0.0 as the value."""
    row = make_measurement(
        video_key="v",
        clip_id="c",
        measurement_type="span_uuid_equal",
        output_a_present=True,
        output_b_present=True,
        value=1.0,
    )
    assert row["value"] == 1.0


def test_window_and_model_dimensions() -> None:
    """Window and model dimensions are carried on the row."""
    row = make_measurement(
        video_key="v",
        clip_id="c",
        window_id="128_256",
        model="qwen",
        measurement_type="caption_similarity",
        output_a_present=True,
        output_b_present=True,
        value=0.97,
    )
    assert row["window_id"] == "128_256"
    assert row["model"] == "qwen"
    pa.Table.from_pylist([row], schema=MEASUREMENT_SCHEMA)


def test_empty_measurements_carries_schema() -> None:
    """empty_measurements returns an empty table that still carries the schema."""
    table = empty_measurements()
    assert table.num_rows == 0
    assert table.schema == MEASUREMENT_SCHEMA
