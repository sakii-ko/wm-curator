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

"""Config-driven pipeline commands."""

import json
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from typer import Argument, Option

pipeline_app = typer.Typer(
    help="Pipeline config tooling.",
    no_args_is_help=True,
)
presets_app = typer.Typer(
    help="Inspect packaged pipeline presets.",
    no_args_is_help=True,
)
pipeline_app.add_typer(presets_app, name="presets")

PipelineName = Literal["video_split"]
TemplateProfile = Literal["base", "smoke"]


@pipeline_app.command(no_args_is_help=True)
def template(
    *,
    kind: Annotated[PipelineName, Argument(help="Pipeline kind template to print.")],
    profile: Annotated[
        TemplateProfile,
        Option("--profile", help="Template profile to print. Use smoke for cheap first-run runtime checks."),
    ] = "base",
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Print an editable config template for a supported pipeline kind."""
    from cosmos_curator.pipelines.ray_data.video_split_config import (  # noqa: PLC0415
        user_config_to_yaml,
        video_split_config_template,
        video_split_template_payload,
    )

    del kind
    if json_output:
        typer.echo(json.dumps(video_split_template_payload(profile), indent=2))
    else:
        sys.stdout.write(user_config_to_yaml(video_split_config_template(profile)))


@pipeline_app.command(no_args_is_help=True)
def validate(
    *,
    config: Annotated[Path, Argument(help="Path to a JSON/YAML pipeline config.")],
    set_overrides: Annotated[
        list[str] | None,
        Option("--set", help="Small resolved-config override in dotted PATH=VALUE form."),
    ] = None,
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Validate a config file after defaults, presets, and overrides resolve."""
    from pydantic import ValidationError  # noqa: PLC0415

    from cosmos_curator.pipelines.ray_data.video_split_config import resolve_video_split_config  # noqa: PLC0415

    try:
        resolution = resolve_video_split_config(config, overrides=set_overrides or [])
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail("invalid", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps({"ok": True, "selected_presets": resolution.selected_presets}, indent=2))
    else:
        typer.echo("valid")


@pipeline_app.command(no_args_is_help=True)
def render(
    *,
    config: Annotated[Path, Argument(help="Path to a JSON/YAML pipeline config.")],
    set_overrides: Annotated[
        list[str] | None,
        Option("--set", help="Small resolved-config override in dotted PATH=VALUE form."),
    ] = None,
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Render the canonical resolved config used for execution."""
    from pydantic import ValidationError  # noqa: PLC0415

    from cosmos_curator.pipelines.ray_data.video_split_config import (  # noqa: PLC0415
        resolve_video_split_config,
        resolved_config_to_json,
    )

    try:
        resolution = resolve_video_split_config(config, overrides=set_overrides or [])
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail("render_failed", exc, json_output=json_output)
    sys.stdout.write(resolved_config_to_json(resolution.config))


@pipeline_app.command(no_args_is_help=True)
def schema(
    *,
    kind: Annotated[PipelineName, Argument(help="Pipeline kind schema to print.")],
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Print JSON Schema for a supported pipeline config."""
    from cosmos_curator.pipelines.ray_data.video_split_config import user_video_split_schema_json  # noqa: PLC0415

    del kind
    del json_output
    sys.stdout.write(user_video_split_schema_json())


@presets_app.command("list")
def list_presets(
    *,
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """List packaged video_split presets."""
    from cosmos_curator.pipelines.ray_data.video_split_config import list_video_split_presets  # noqa: PLC0415

    presets = list_video_split_presets()
    if json_output:
        typer.echo(json.dumps({"presets": presets}, indent=2))
        return

    for preset in presets:
        typer.echo(f"{preset['qualified_name']}")


@presets_app.command("show", no_args_is_help=True)
def show_preset(
    *,
    name: Annotated[str, Argument(help="Preset name, e.g. caption.balanced or balanced.")],
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Show one packaged video_split preset."""
    from cosmos_curator.pipelines.ray_data.video_split_config import (  # noqa: PLC0415
        ConfigResolutionError,
        show_video_split_preset,
    )

    try:
        preset = show_video_split_preset(name)
    except ConfigResolutionError as exc:
        _fail("unknown_preset", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps(preset, indent=2))
    else:
        typer.echo(json.dumps(preset["fragment"], indent=2))


def _fail(code: str, exc: Exception, *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({"ok": False, "error": code, "message": str(exc)}, indent=2), err=True)
    else:
        typer.echo(str(exc), err=True)
    raise typer.Exit(2)


if __name__ == "__main__":
    pipeline_app()
