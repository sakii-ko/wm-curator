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
"""Tests for split pipeline CLI argument wiring."""

import argparse

from cosmos_curator.pipelines.video.splitting_pipeline import _setup_parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    return parser


def test_caption_quality_flags_default_enabled() -> None:
    """Caption quality flags should default to enabled."""
    args = _parser().parse_args([])

    assert args.caption_quality_flags_enabled is True


def test_no_caption_quality_flags_disables_flags() -> None:
    """The disable flag should set caption_quality_flags_enabled to False."""
    args = _parser().parse_args(["--no-caption-quality-flags"])

    assert args.caption_quality_flags_enabled is False
