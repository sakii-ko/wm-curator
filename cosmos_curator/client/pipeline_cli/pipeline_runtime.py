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

"""Run-only entrypoint for config-backed pipelines inside runtime environments."""

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from typer import Argument, Option

from cosmos_curator.pipelines.ray_data.video_split_config import resolve_video_split_config


def main(
    config: Annotated[Path, Argument(help="Path to a JSON/YAML pipeline config.")],
    set_overrides: Annotated[
        list[str] | None,
        Option("--set", help="Small resolved-config override in dotted PATH=VALUE form."),
    ] = None,
    *,
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Run a pipeline from a JSON/YAML config."""
    try:
        resolution = resolve_video_split_config(config, overrides=set_overrides or [])
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail("invalid", exc, json_output=json_output)

    from cosmos_curator.pipelines.ray_data.splitting_pipeline import run_config  # noqa: PLC0415

    try:
        clips_written = run_config(resolution.config)
    except Exception as exc:
        if json_output:
            _fail("runtime", exc, json_output=True)
        raise

    if json_output:
        typer.echo(json.dumps({"clips_written": clips_written}, indent=2))
    else:
        typer.echo(f"Wrote {clips_written} clip(s)")


def _fail(code: str, exc: Exception, *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({"ok": False, "error": code, "message": str(exc)}, indent=2), err=True)
    else:
        typer.echo(str(exc), err=True)
    raise typer.Exit(2)


if __name__ == "__main__":
    typer.run(main)
