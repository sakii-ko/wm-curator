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

"""Tests for the run-only pipeline runtime entrypoint."""

import json
from pathlib import Path

import pytest
import typer

from cosmos_curator.client.pipeline_cli import pipeline_runtime
from cosmos_curator.pipelines.ray_data import splitting_pipeline
from cosmos_curator.pipelines.ray_data.video_split_config import ResolvedVideoSplitConfig


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


def test_pipeline_runtime_hands_resolved_config_to_ray_data_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime entrypoint resolves config and delegates to the Ray Data typed execution entrypoint."""
    captured: dict[str, ResolvedVideoSplitConfig] = {}

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        captured["config"] = config
        return 12

    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)
    config_path = _write_config(tmp_path / "split.yaml", extra="split:\n  preset: fixed_stride_10s")

    pipeline_runtime.main(config_path, set_overrides=None, json_output=True)

    assert json.loads(capsys.readouterr().out) == {"clips_written": 12}
    assert captured["config"].split.method == "fixed_stride"
    assert captured["config"].input.video_path == "/videos"


def test_pipeline_runtime_reports_runtime_errors_as_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime errors are machine-readable when --json is requested."""

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        del config
        msg = "runtime boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)
    config_path = _write_config(tmp_path / "split.yaml")

    with pytest.raises(typer.Exit) as exc_info:
        pipeline_runtime.main(config_path, set_overrides=None, json_output=True)

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "ok": False,
        "error": "runtime",
        "message": "runtime boom",
    }


def test_pipeline_runtime_reraises_runtime_errors_without_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Interactive runs keep normal tracebacks for runtime failures."""

    def fake_run_config(config: ResolvedVideoSplitConfig) -> int:
        del config
        msg = "runtime boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(splitting_pipeline, "run_config", fake_run_config)
    config_path = _write_config(tmp_path / "split.yaml")

    with pytest.raises(RuntimeError, match="runtime boom"):
        pipeline_runtime.main(config_path, set_overrides=None, json_output=False)
