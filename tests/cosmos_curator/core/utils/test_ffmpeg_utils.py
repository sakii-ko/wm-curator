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
"""Tests for FFmpeg runtime checks."""

import subprocess
from unittest.mock import MagicMock

import pytest

from cosmos_curator.core.utils.ffmpeg_utils import assert_ffmpeg_supports_h264


def test_assert_ffmpeg_supports_h264_passes_when_decoder_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """An FFmpeg decoder listing with h264 should satisfy the preflight."""
    run_stub = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=["ffmpeg", "-hide_banner", "-decoders"],
            returncode=0,
            stdout=" VFS..D h264                 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10\n",
            stderr="",
        )
    )
    monkeypatch.setattr("cosmos_curator.core.utils.ffmpeg_utils.subprocess.run", run_stub)

    assert_ffmpeg_supports_h264()


def test_assert_ffmpeg_supports_h264_raises_when_decoder_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An FFmpeg decoder listing without h264 should fail clearly."""
    run_mock = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=["ffmpeg", "-hide_banner", "-decoders"],
            returncode=0,
            stdout=" V....D rawvideo             raw video\n VFS..D av1                  Alliance for Open Media AV1\n",
            stderr="",
        )
    )
    monkeypatch.setattr("cosmos_curator.core.utils.ffmpeg_utils.subprocess.run", run_mock)

    with pytest.raises(RuntimeError, match=r"does not expose an H\.264 decoder"):
        assert_ffmpeg_supports_h264()


def test_assert_ffmpeg_supports_h264_raises_when_ffmpeg_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing FFmpeg binary should fail before pipeline startup continues."""
    run_mock = MagicMock(side_effect=FileNotFoundError("ffmpeg"))
    monkeypatch.setattr("cosmos_curator.core.utils.ffmpeg_utils.subprocess.run", run_mock)

    with pytest.raises(RuntimeError, match="Failed to query FFmpeg decoders"):
        assert_ffmpeg_supports_h264()
