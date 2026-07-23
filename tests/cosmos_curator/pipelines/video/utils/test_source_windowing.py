# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for decoding caption windows from source path/span references."""

import pathlib
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import numpy as np
import pytest
import torch

from cosmos_curator.pipelines.common.model_constraints import PreprocessMode
from cosmos_curator.pipelines.video.utils import windowing_utils
from cosmos_curator.pipelines.video.utils.data_model import Clip, Video, WindowConfig
from cosmos_curator.pipelines.video.utils.windowing_types import WindowFrameInfo


def test_source_frame_bounds_uses_container_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Indexed containers should map spans using their actual display timestamps."""
    sensor = SimpleNamespace(
        start_ns=1_000_000_000,
        timestamps_ns=np.array(
            [1_000_000_000, 1_500_000_000, 2_000_000_000, 2_500_000_000, 3_000_000_000],
            dtype=np.int64,
        ),
    )
    monkeypatch.setattr(windowing_utils, "CameraSensor", lambda *_args, **_kwargs: sensor)

    assert windowing_utils._source_frame_bounds(pathlib.Path("source.mkv"), (0.5, 1.75), 0) == (1, 4)


def test_source_frame_bounds_falls_back_for_unindexed_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """Containers without a header index should use the narrow CFR fallback."""

    def fail_sensor(*_args: object, **_kwargs: object) -> None:
        message = "container has no header index"
        raise ValueError(message)

    monkeypatch.setattr(windowing_utils, "CameraSensor", fail_sensor)
    monkeypatch.setattr(windowing_utils, "get_avg_frame_rate", lambda *_args, **_kwargs: 25.0)
    monkeypatch.setattr(windowing_utils, "get_frame_count", lambda *_args, **_kwargs: 100)

    assert windowing_utils._source_frame_bounds(pathlib.Path("source.ts"), (1.0, 3.5), 0) == (25, 88)


def test_split_source_video_uses_absolute_decode_ranges_and_relative_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decode ranges stay source-absolute while emitted window indices are clip-relative."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(windowing_utils, "_source_frame_bounds", lambda *_args, **_kwargs: (10, 20))

    def fake_fetch_video(
        _path: str,
        *,
        window_range: list[object],
        **kwargs: object,
    ) -> tuple[torch.Tensor, list[int]]:
        captured["window_range"] = window_range
        captured["kwargs"] = kwargs
        return torch.arange(18, dtype=torch.float32).reshape(6, 3, 1, 1), [2, 2, 2]

    monkeypatch.setattr(windowing_utils, "fetch_video", fake_fetch_video, raising=False)

    mp4, frames, windows = windowing_utils.split_source_video_into_windows(
        pathlib.Path("source.mkv"),
        (2.0, 4.0),
        window_size=4,
        remainder_threshold=2,
        sampling_fps=2.0,
        preprocess_mode=PreprocessMode.CURATOR,
        return_video_frames=True,
        stream_index=2,
        rotation_degrees_clockwise=90,
    )

    assert [(window.start, window.end) for window in windows] == [(0, 3), (4, 7), (8, 9)]
    source_windows = cast("list[WindowFrameInfo]", captured["window_range"])
    kwargs = cast("dict[str, object]", captured["kwargs"])
    assert [(window.start, window.end) for window in source_windows] == [(10, 13), (14, 17), (18, 19)]
    assert kwargs["stream_index"] == 2
    assert mp4 == [None, None, None]
    assert [tuple(frame.shape) for frame in frames if frame is not None] == [(2, 3, 1, 1)] * 3


def test_make_windows_for_video_decodes_all_source_clips_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Multiple source spans should share one open/decode pass."""
    source = tmp_path / "source.mkv"
    source.touch()
    clips = [
        Clip(uuid=uuid4(), source_video=str(source), span=(0.0, 1.0)),
        Clip(uuid=uuid4(), source_video=str(source), span=(1.0, 2.0)),
    ]
    video = Video(input_video=source, clips=clips)
    bounds = {(0.0, 1.0): (10, 14), (1.0, 2.0): (20, 24)}
    monkeypatch.setattr(
        windowing_utils,
        "_source_frame_bounds",
        lambda _path, span, _stream_index: bounds[span],
    )
    captured: dict[str, object] = {"calls": 0}

    def fake_read_video(
        _path: str,
        _sampling_fps: float,
        _num_frames_to_use: int,
        source_windows: list[WindowFrameInfo],
        *,
        stream_index: int,
    ) -> tuple[torch.Tensor, list[int]]:
        captured["calls"] = cast("int", captured["calls"]) + 1
        captured["source_windows"] = source_windows
        captured["stream_index"] = stream_index
        return torch.arange(12, dtype=torch.uint8).reshape(4, 3, 1, 1), [2, 2]

    monkeypatch.setattr(windowing_utils, "read_video_cpu", fake_read_video, raising=False)
    monkeypatch.setattr(
        windowing_utils,
        "preprocess_video_frames",
        lambda frames, **_kwargs: frames,
        raising=False,
    )

    windows, frames = windowing_utils.make_windows_for_video(
        video,
        WindowConfig(window_size=4, remainder_threshold=2, sampling_fps=2.0),
        num_decode_threads=1,
        preprocess_mode=PreprocessMode.MODEL,
        stream_index=2,
    )

    assert captured["calls"] == 1
    assert captured["stream_index"] == 2
    source_windows = cast("list[WindowFrameInfo]", captured["source_windows"])
    assert [(window.start, window.end) for window in source_windows] == [(10, 13), (20, 23)]
    assert [(window.start_frame, window.end_frame) for window in windows] == [(0, 3), (0, 3)]
    assert [tuple(frame.shape) for frame in frames] == [(2, 3, 1, 1), (2, 3, 1, 1)]
    assert [len(clip.windows) for clip in clips] == [1, 1]
