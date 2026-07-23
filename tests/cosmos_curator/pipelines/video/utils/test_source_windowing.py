# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for decoding caption windows from source path/span references."""

import pathlib
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from cosmos_curator.pipelines.common.model_constraints import PreprocessMode
from cosmos_curator.pipelines.video.utils import windowing_utils
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
