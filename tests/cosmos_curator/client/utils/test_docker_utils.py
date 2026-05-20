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
"""Tests for Dockerfile generation utilities."""

from itertools import pairwise
from pathlib import Path

import pytest

from cosmos_curator.client.utils import docker_utils

REPO_ROOT = Path(__file__).resolve().parents[4]
DOCKERFILE_TEMPLATE_PATH = REPO_ROOT / "package" / "cosmos_curator" / "default.dockerfile.jinja2"


def _render_dockerfile(
    tmp_path: Path,
    *,
    slim: bool,
    redistributable: bool,
) -> str:
    dockerfile_path = docker_utils.generate_dockerfile(
        dockerfile_template_path=DOCKERFILE_TEMPLATE_PATH,
        conda_env_names=["default"],
        dockerfile_output_path=tmp_path / f"Dockerfile-slim-{slim}-redistributable-{redistributable}",
        slim=slim,
        redistributable=redistributable,
    )
    return dockerfile_path.read_text()


def _empty_continuation_lines(contents: str) -> list[int]:
    lines = contents.splitlines()
    return [
        line_number
        for line_number, (previous_line, line) in enumerate(pairwise(lines), start=2)
        if previous_line.rstrip().endswith("\\") and not line.strip()
    ]


@pytest.mark.parametrize("slim", [False, True])
@pytest.mark.parametrize("redistributable", [False, True])
def test_generated_dockerfile_has_no_empty_continuation_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    slim: bool,
    redistributable: bool,
) -> None:
    """Dockerfile continuations must not include blank rendered template lines."""
    monkeypatch.chdir(REPO_ROOT)

    contents = _render_dockerfile(tmp_path, slim=slim, redistributable=redistributable)
    pkg_config_arg = contents.find("ARG PKG_CONFIG_PATH")
    pkg_config_env = contents.find('PKG_CONFIG_PATH="/opt/ffmpeg/lib/pkgconfig:${PKG_CONFIG_PATH:-}"')

    assert _empty_continuation_lines(contents) == []
    assert pkg_config_arg != -1
    assert pkg_config_env != -1
    assert pkg_config_arg < pkg_config_env
