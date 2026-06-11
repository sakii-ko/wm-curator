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

"""Tests for Ray Data video_split pipeline config resolution."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cosmos_curator.core.utils.environment import MODEL_WEIGHTS_PREFIX
from cosmos_curator.pipelines.ray_data.video_split_config import (
    ConfigResolutionError,
    ResolvedVideoSplitConfig,
    UserVideoSplitConfig,
    list_video_split_presets,
    resolve_video_split_config,
    resolve_video_split_config_data,
    resolved_config_to_json,
    show_video_split_preset,
    user_video_split_schema_json,
)


def _minimal_raw(**overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "schema_version": 1,
        "kind": "video_split",
        "input": {"video_path": "/videos"},
        "output": {"clip_path": "/clips"},
    }
    raw.update(overrides)
    return raw


def test_resolve_minimal_config_fills_canonical_defaults() -> None:
    """A small user config resolves into the full Ray Data execution contract."""
    resolution = resolve_video_split_config_data(_minimal_raw())

    assert resolution.selected_presets == ()
    config = resolution.config
    assert config.schema_version == 1
    assert config.kind == "video_split"
    assert config.input.video_path == "/videos"
    assert config.input.limit == 0
    assert config.split.method == "transnetv2"
    assert config.split.transnetv2.threshold == 0.4
    assert config.caption.enabled is True
    assert config.caption.batch_size == 32
    assert config.output.metadata_format == "json"
    assert config.execution.model_weights_path == MODEL_WEIGHTS_PREFIX


def test_presets_expand_before_explicit_user_fields() -> None:
    """Preset fragments are visible in the resolved config and user fields win."""
    resolution = resolve_video_split_config_data(
        _minimal_raw(
            caption={"preset": "off", "enabled": True},
            execution={"preset": "progress"},
        )
    )

    assert resolution.selected_presets == ("caption.off", "execution.progress")
    assert resolution.config.caption.enabled is True
    assert resolution.config.execution.progress is True


def test_yaml_boolean_off_preset_resolves_to_caption_off() -> None:
    """Unquoted YAML ``off`` is parsed as False but should still select caption.off."""
    resolution = resolve_video_split_config_data(_minimal_raw(caption={"preset": False}))

    assert resolution.selected_presets == ("caption.off",)
    assert resolution.config.caption.enabled is False


def test_cli_overrides_apply_after_user_fields() -> None:
    """--set overrides are applied last and parsed as typed YAML scalars."""
    resolution = resolve_video_split_config_data(
        _minimal_raw(input={"video_path": "/videos", "limit": 10}),
        overrides=["input.limit=3", "caption.enabled=false"],
    )

    assert resolution.config.input.limit == 3
    assert resolution.config.caption.enabled is False


def test_cli_overrides_reject_preset_fields() -> None:
    """Preset selection is a user-config resolution step, not a resolved-config field."""
    with pytest.raises(ConfigResolutionError, match="preset fields"):
        resolve_video_split_config_data(_minimal_raw(), overrides=["caption.preset=off"])


def test_unknown_user_field_rejected_before_resolution() -> None:
    """Raw validation rejects typo fields instead of silently ignoring them."""
    with pytest.raises(ValidationError):
        resolve_video_split_config_data(_minimal_raw(caption={"batc_size": 4}))


def test_missing_required_resolved_field_rejected() -> None:
    """Defaults do not satisfy required semantic paths."""
    raw = {
        "schema_version": 1,
        "kind": "video_split",
        "input": {"video_path": "/videos"},
    }

    with pytest.raises(ValidationError):
        resolve_video_split_config_data(raw)


def test_invalid_output_format_rejected() -> None:
    """The v1 Ray Data output writer only supports JSON metadata."""
    with pytest.raises(ValidationError):
        resolve_video_split_config_data(_minimal_raw(output={"clip_path": "/clips", "metadata_format": "lance"}))


def test_fixed_stride_preset_resolves_to_canonical_method() -> None:
    """User configs use underscore method names; legacy CLI spelling stays outside the schema."""
    resolution = resolve_video_split_config_data(_minimal_raw(split={"preset": "fixed_stride_10s"}))

    assert resolution.config.split.method == "fixed_stride"
    assert resolution.config.split.fixed_stride.duration_s == 10.0
    assert resolution.config.split.fixed_stride.stride_s == 10.0


def test_rendered_config_round_trips_as_resolved_schema() -> None:
    """Rendered JSON is a canonical resolved config, not the raw user config."""
    config = resolve_video_split_config_data(_minimal_raw(caption={"preset": "balanced"})).config
    rendered = resolved_config_to_json(config)
    payload = json.loads(rendered)

    assert "preset" not in payload["caption"]
    assert ResolvedVideoSplitConfig.model_validate_json(rendered) == config


def test_user_schema_includes_top_level_contract() -> None:
    """JSON Schema is generated from the user-facing Pydantic model."""
    schema = json.loads(user_video_split_schema_json())

    assert schema["title"] == "UserVideoSplitConfig"
    assert set(schema["required"]) == {"schema_version", "kind"}
    assert UserVideoSplitConfig.model_json_schema() == schema


def test_presets_can_be_listed_and_shown_by_short_name() -> None:
    """Preset inspection exposes packaged fragments."""
    presets = list_video_split_presets()
    names = {preset["qualified_name"] for preset in presets}

    assert "caption.balanced" in names
    preset = show_video_split_preset("balanced")
    assert preset["qualified_name"] == "caption.balanced"
    assert preset["fragment"]["model"] == "qwen"


def test_unknown_preset_rejected() -> None:
    """Unknown preset names fail during resolution."""
    with pytest.raises(ConfigResolutionError, match="Unknown caption preset"):
        resolve_video_split_config_data(_minimal_raw(caption={"preset": "aggressive"}))


def test_loads_yaml_config_file(tmp_path: Path) -> None:
    """YAML files are first-class inputs for the config path."""
    config_path = tmp_path / "split.yaml"
    config_path.write_text(
        """schema_version: 1
kind: video_split
input:
  video_path: /videos
output:
  clip_path: /clips
""",
        encoding="utf-8",
    )

    resolution = resolve_video_split_config(config_path)

    assert resolution.config.input.video_path == "/videos"
    assert resolution.config.output.clip_path == "/clips"
