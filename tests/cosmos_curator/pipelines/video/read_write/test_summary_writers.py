# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for write_split_summary num_remuxed_videos metric."""

import pathlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from cosmos_curator.pipelines.video.read_write import summary_writers
from cosmos_curator.pipelines.video.read_write.summary_writers import write_split_summary
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask, Video


def _make_video(*, was_remuxed: bool, clip_chunk_index: int) -> Video:
    v = Video(input_video=pathlib.Path("test.ts"))
    v.was_remuxed = was_remuxed
    v.clip_chunk_index = clip_chunk_index
    return v


def test_num_remuxed_videos_no_double_count() -> None:
    """clip_chunk_index == 0 guard prevents double-counting chunked videos.

    Two Video objects represent the same source video split into two chunks.
    Both have was_remuxed=True, but only the chunk-0 object should be counted,
    so num_remuxed_videos must be 1, not 2.
    """
    chunk0 = _make_video(was_remuxed=True, clip_chunk_index=0)
    chunk1 = _make_video(was_remuxed=True, clip_chunk_index=1)

    task0 = SplitPipeTask(session_id="s", video=chunk0)
    task1 = SplitPipeTask(session_id="s", video=chunk1)

    with patch("cosmos_curator.pipelines.video.read_write.summary_writers._write_split_result_summary") as mock_write:
        write_split_summary(
            input_path="/in",
            input_videos_relative=["test.ts"],
            num_input_videos_selected=1,
            output_path="/out",
            output_s3_profile_name="default",
            output_tasks=[task0, task1],
            embedding_algorithm="internvideo2",
            limit=0,
        )

    mock_write.assert_called_once()
    _, kwargs = mock_write.call_args
    assert kwargs["num_remuxed_videos"] == 1, (
        f"Expected 1 remuxed video (chunk-0 only), got {kwargs['num_remuxed_videos']}"
    )


def test_split_result_summary_uses_caption_windows_for_token_averages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token averages are per caption window, not per clip with any caption."""
    captured_summary: dict[str, Any] = {}
    info_calls: list[tuple[str, tuple[object, ...]]] = []

    def fake_write_json(data: dict[str, Any], *_args: object, **_kwargs: object) -> None:
        captured_summary.update(data)

    def fake_info(message: str, *args: object) -> None:
        info_calls.append((message, args))

    monkeypatch.setattr(
        summary_writers,
        "logger",
        SimpleNamespace(info=fake_info),
    )
    monkeypatch.setattr(summary_writers.storage_utils, "get_storage_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(summary_writers, "get_files_relative", lambda *_args, **_kwargs: ["video.mp4"])
    monkeypatch.setattr(summary_writers, "write_json", fake_write_json)
    monkeypatch.setattr(
        summary_writers,
        "_read_all_video_metadata_parallel",
        lambda *_args, **_kwargs: {
            "video.mp4": summary_writers.ProcessedVideoMetadata(
                video_metadata={
                    "video_uuid": "video-id",
                    "num_clip_chunks": 1,
                    "num_total_clips": 1,
                    "duration": 5.0,
                },
                clip_chunks=[
                    {
                        "num_clips_passed": 1,
                        "num_clips_transcoded": 1,
                        "num_clips_with_caption": 1,
                        "num_caption_windows": 2,
                        "total_prompt_tokens": 30,
                        "total_output_tokens": 12,
                        "clips": ["clip-id"],
                        "filtered_clips": [],
                    }
                ],
            )
        },
    )

    summary_writers._write_split_result_summary(
        input_path="/input",
        input_videos_relative=["video.mp4"],
        num_input_videos_selected=1,
        output_path="/output",
        output_s3_profile_name="default",
        embedding_algorithm="internvideo2",
        limit=0,
        pipeline_run_time=1.0,
        write_all_caption_json=False,
    )

    assert captured_summary["total_num_clips_with_caption"] == 1
    assert captured_summary["total_num_caption_windows"] == 2
    assert captured_summary["video.mp4"]["num_caption_windows"] == 2

    throughput_args = next(args for message, args in info_calls if "Captioning throughput" in message)
    assert throughput_args[2] == 2
    assert throughput_args[3] == 15
    assert throughput_args[4] == 6
