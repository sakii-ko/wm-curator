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

"""Tests for the image run_pipeline entry point."""

import contextlib
import json
import pathlib
import sys
from typing import TYPE_CHECKING

import pytest

from cosmos_curator.pipelines.image import annotate_pipeline
from cosmos_curator.pipelines.image import run_pipeline as image_run_pipeline

if TYPE_CHECKING:
    import argparse


def test_config_mode_accepts_flat_yaml(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A flat YAML config should dispatch through the image NVCF-style entry point."""
    config_path = tmp_path / "annotate.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pipeline: annotate",
                f"input_image_path: {tmp_path / 'in'}",
                f"output_path: {tmp_path / 'out'}",
                "generate_captions: false",
                "limit: 10",
            ]
        )
    )
    captured: dict[str, argparse.Namespace] = {}

    monkeypatch.setattr(sys, "argv", ["run_pipeline", str(config_path)])
    monkeypatch.setattr(annotate_pipeline, "nvcf_run_annotate", lambda args: captured.setdefault("args", args))

    image_run_pipeline.cli()

    args = captured["args"]
    assert args.input_image_path == str(tmp_path / "in")
    assert args.output_path == str(tmp_path / "out")
    assert args.generate_captions is False
    assert args.limit == 10


def test_config_mode_accepts_nested_json_args(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An NVCF-shaped JSON config should dispatch with args flattened."""
    config_path = tmp_path / "annotate.json"
    config_path.write_text(
        json.dumps(
            {
                "pipeline": "annotate",
                "args": {
                    "input_image_path": str(tmp_path / "images"),
                    "output_path": str(tmp_path / "output"),
                    "generate_embeddings": False,
                },
            }
        )
    )
    captured: dict[str, argparse.Namespace] = {}

    monkeypatch.setattr(sys, "argv", ["run_pipeline", str(config_path)])
    monkeypatch.setattr(annotate_pipeline, "nvcf_run_annotate", lambda args: captured.setdefault("args", args))

    image_run_pipeline.cli()

    args = captured["args"]
    assert args.input_image_path == str(tmp_path / "images")
    assert args.output_path == str(tmp_path / "output")
    assert args.generate_embeddings is False


def test_config_mode_rejects_missing_pipeline(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config mode requires an explicit supported pipeline name."""
    config_path = tmp_path / "annotate.yaml"
    config_path.write_text(f"input_image_path: {tmp_path / 'in'}\noutput_path: {tmp_path / 'out'}\n")

    monkeypatch.setattr(sys, "argv", ["run_pipeline", str(config_path)])

    with pytest.raises(SystemExit, match="valid 'pipeline' key"):
        image_run_pipeline.cli()


def test_config_mode_rejects_invalid_pipeline(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config mode only supports the image annotate pipeline."""
    config_path = tmp_path / "split.yaml"
    config_path.write_text("pipeline: split\nargs: {}\n")

    monkeypatch.setattr(sys, "argv", ["run_pipeline", str(config_path)])

    with pytest.raises(SystemExit, match="Valid pipelines: annotate"):
        image_run_pipeline.cli()


def test_config_mode_rejects_extra_args(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config mode should not accept mixed CLI flags."""
    config_path = tmp_path / "annotate.yaml"
    config_path.write_text("pipeline: annotate\nargs: {}\n")

    monkeypatch.setattr(sys, "argv", ["run_pipeline", str(config_path), "--limit", "1"])

    with pytest.raises(SystemExit, match="Config mode takes no extra arguments"):
        image_run_pipeline.cli()


def test_cli_mode_still_dispatches_annotate_subcommand(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing annotate subcommand path should keep working."""
    captured: dict[str, argparse.Namespace] = {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline",
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(annotate_pipeline, "annotate", lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(image_run_pipeline, "profiling_scope", lambda _args: contextlib.nullcontext())

    image_run_pipeline.cli()

    args = captured["args"]
    assert args.command == "annotate"
    assert args.input_image_path == str(tmp_path / "in")
    assert args.output_path == str(tmp_path / "out")
