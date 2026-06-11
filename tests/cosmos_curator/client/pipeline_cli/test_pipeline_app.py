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

"""Tests for the config-driven ``cosmos-curator pipeline`` CLI."""

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator

runner = CliRunner()


def _write_config(path: Path, *, extra: str = "") -> Path:
    path.write_text(
        f"""schema_version: 1
kind: video_split
input:
  video_path: /videos
output:
  clip_path: /clips
{extra}
""",
        encoding="utf-8",
    )
    return path


def test_pipeline_validate_reports_valid_config(tmp_path: Path) -> None:
    """Validate resolves defaults and reports success."""
    config_path = _write_config(tmp_path / "split.yaml")

    result = runner.invoke(cosmos_curator, ["pipeline", "validate", str(config_path), "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"ok": True, "selected_presets": []}


def test_pipeline_validate_reports_invalid_config_as_json(tmp_path: Path) -> None:
    """Validate --json returns structured errors for tools."""
    config_path = _write_config(tmp_path / "split.yaml", extra="caption:\n  bad_field: true")

    result = runner.invoke(cosmos_curator, ["pipeline", "validate", str(config_path), "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["ok"] is False
    assert payload["error"] == "invalid"
    assert "bad_field" in payload["message"]


def test_pipeline_render_outputs_resolved_config_with_overrides(tmp_path: Path) -> None:
    """Render prints canonical JSON after presets and --set overrides."""
    config_path = _write_config(tmp_path / "split.yaml", extra="caption:\n  preset: balanced")

    result = runner.invoke(
        cosmos_curator,
        ["pipeline", "render", str(config_path), "--set", "caption.enabled=false"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["caption"] == {
        "enabled": False,
        "backend": "ray_data_llm",
        "model": "qwen",
        "batch_size": 32,
    }
    assert "preset" not in payload["caption"]


def test_pipeline_schema_outputs_user_config_schema() -> None:
    """Schema prints JSON Schema for video_split."""
    result = runner.invoke(cosmos_curator, ["pipeline", "schema", "video_split", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["title"] == "UserVideoSplitConfig"
    assert "RawCaptionConfig" in payload["$defs"]


def test_pipeline_template_outputs_base_yaml_by_default() -> None:
    """Template prints the smallest editable YAML config by default."""
    result = runner.invoke(cosmos_curator, ["pipeline", "template", "video_split"])

    assert result.exit_code == 0
    payload = yaml.safe_load(result.stdout)
    assert payload == {
        "schema_version": 1,
        "kind": "video_split",
        "input": {"video_path": "/config/input"},
        "output": {"clip_path": "/config/output"},
    }


def test_pipeline_template_outputs_base_json_with_required_fields() -> None:
    """Template --json gives agents the template plus required author inputs."""
    result = runner.invoke(cosmos_curator, ["pipeline", "template", "video_split", "--profile", "base", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["kind"] == "video_split"
    assert payload["profile"] == "base"
    assert [field["path"] for field in payload["required_fields"]] == [
        "schema_version",
        "kind",
        "input.video_path",
        "output.clip_path",
    ]
    assert payload["config"] == {
        "schema_version": 1,
        "kind": "video_split",
        "input": {"video_path": "/config/input"},
        "output": {"clip_path": "/config/output"},
    }


def test_pipeline_template_outputs_smoke_yaml() -> None:
    """The smoke template keeps the first runtime check cheap."""
    result = runner.invoke(cosmos_curator, ["pipeline", "template", "video_split", "--profile", "smoke"])

    assert result.exit_code == 0
    payload = yaml.safe_load(result.stdout)
    assert payload == {
        "schema_version": 1,
        "kind": "video_split",
        "input": {"video_path": "/config/input"},
        "split": {"preset": "fixed_stride_10s"},
        "caption": {"enabled": False},
        "output": {"clip_path": "/config/output"},
    }


def test_pipeline_presets_list_and_show() -> None:
    """Preset inspection commands are JSON-friendly."""
    list_result = runner.invoke(cosmos_curator, ["pipeline", "presets", "list", "--json"])
    show_result = runner.invoke(cosmos_curator, ["pipeline", "presets", "show", "caption.balanced", "--json"])

    assert list_result.exit_code == 0
    assert show_result.exit_code == 0
    assert "caption.balanced" in {preset["qualified_name"] for preset in json.loads(list_result.stdout)["presets"]}
    assert json.loads(show_result.stdout)["fragment"]["batch_size"] == 32


def test_pipeline_run_is_not_host_cli_command(tmp_path: Path) -> None:
    """Pipeline execution is exposed through the runtime Pixi task, not the host config CLI."""
    config_path = _write_config(tmp_path / "split.yaml")

    result = runner.invoke(cosmos_curator, ["pipeline", "run", str(config_path)])

    assert result.exit_code != 0
