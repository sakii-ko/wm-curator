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
"""CPU tests for the shared geometry/normal decode grid."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.pipelines.video.annotation import decode_grid
from cosmos_curator.pipelines.video.annotation.decode_grid import (
    AnnotationGrid,
    annotation_grid_configuration_matches,
    decode_annotation_clip,
    make_raster_transform,
)


def _write_video(path: Path, *, fps: int, frame_count: int) -> None:
    with av.open(str(path), mode="w", format="mpegts") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 24
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, fps)
        for index in range(frame_count):
            array = np.empty((stream.height, stream.width, 3), dtype=np.uint8)
            array[..., 0] = index * 10
            array[..., 1] = np.arange(stream.width, dtype=np.uint8) * 10
            array[..., 2] = np.arange(stream.height, dtype=np.uint8)[:, None] * 15
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_default_raster_uses_center_crop_pixel_center_transform() -> None:
    """The old exercised 832x480 recipe should retain its exact crop math."""
    grid = AnnotationGrid()
    raster = make_raster_transform(
        1920,
        1080,
        rotation_degrees_clockwise=0,
        grid=grid,
    )

    assert (raster.crop_left, raster.crop_top, raster.crop_width, raster.crop_height) == (
        24,
        0,
        1872,
        1080,
    )
    assert raster.grid_to_oriented_source == (
        (2.25, 0.0, 24.625),
        (0.0, 2.25, 0.625),
        (0.0, 0.0, 1.0),
    )
    assert raster.grid_to_source == raster.grid_to_oriented_source


@pytest.mark.parametrize(
    ("rotation", "expected_size", "expected_matrix"),
    [
        (0, (6, 4), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        (90, (4, 6), ((0.0, 1.0, 0.0), (-1.0, 0.0, 3.0), (0.0, 0.0, 1.0))),
        (180, (6, 4), ((-1.0, 0.0, 5.0), (0.0, -1.0, 3.0), (0.0, 0.0, 1.0))),
        (270, (4, 6), ((0.0, -1.0, 5.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
    ],
)
def test_rotation_is_part_of_the_grid_to_source_transform(
    rotation: int,
    expected_size: tuple[int, int],
    expected_matrix: tuple[tuple[float, float, float], ...],
) -> None:
    """The persisted transform should include display rotation exactly once."""
    raster = make_raster_transform(
        6,
        4,
        rotation_degrees_clockwise=rotation,
        grid=AnnotationGrid(width=expected_size[0], height=expected_size[1]),
    )

    assert (raster.oriented_width, raster.oriented_height) == expected_size
    assert raster.grid_to_source == expected_matrix


def test_low_fps_source_keeps_regular_grid_and_records_repeated_source_pts(
    tmp_path: Path,
) -> None:
    """Supersampling should repeat source frames, never primary grid timestamps."""
    source = tmp_path / "ten-fps.ts"
    _write_video(source, fps=10, frame_count=10)
    grid = AnnotationGrid(sample_fps=15.0, width=12, height=8)

    decoded = decode_annotation_clip(
        source,
        None,
        stream_index=0,
        rotation_degrees_clockwise=90,
        grid=grid,
        min_frames=1,
        max_frames=20,
    )

    assert decoded.frames.shape == (15, 8, 12, 3)
    _, _, expected_timestamps = make_ts_grid(
        0,
        exclusive_end_ns=1_000_000_000,
        sample_rate_hz=15.0,
    )
    np.testing.assert_array_equal(decoded.timestamps_ns, expected_timestamps)
    assert np.all(np.diff(decoded.timestamps_ns) > 0)
    assert np.all(np.diff(decoded.source_timestamps_ns) >= 0)
    assert len(np.unique(decoded.source_timestamps_ns)) == 10
    assert len(decoded.source_timestamps_ns) == 15
    for source_timestamp in np.unique(decoded.source_timestamps_ns):
        selected = decoded.frames[decoded.source_timestamps_ns == source_timestamp]
        np.testing.assert_array_equal(selected, np.broadcast_to(selected[0], selected.shape))
    first_frame = decoded.frames[0]
    assert float(first_frame[-1, :, 1].mean()) > float(first_frame[0, :, 1].mean())
    assert float(first_frame[:, -1, 2].mean()) < float(first_frame[:, 0, 2].mean())
    assert decoded.raster.rotation_degrees_clockwise == 90
    assert (decoded.raster.oriented_width, decoded.raster.oriented_height) == (16, 24)
    assert decoded.decoder_backend in {"pyav_seek_grid", "pyav_sequential_grid"}


def test_grid_resume_signature_ignores_raster_details_but_not_configuration() -> None:
    """Resume compatibility should compare semantic knobs without a grid hash."""
    grid = AnnotationGrid(sample_fps=15.0, width=832, height=480)
    metadata = grid.metadata(
        make_raster_transform(
            3840,
            2160,
            rotation_degrees_clockwise=0,
            grid=grid,
        )
    )

    assert annotation_grid_configuration_matches(metadata, grid)
    assert not annotation_grid_configuration_matches(
        metadata,
        AnnotationGrid(sample_fps=15.0, width=640, height=360),
    )
    assert not annotation_grid_configuration_matches({}, grid)


def test_packet_pts_mismatch_retries_with_decoded_frame_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A packet/frame PTS mismatch should transparently use exact frame PTS."""
    packet_timeline = decode_grid._VideoTimeline(
        pts_stream=np.asarray([111], dtype=np.int64),
        timestamps_ns=np.asarray([0], dtype=np.int64),
        source_width=2,
        source_height=2,
        nominal_fps=1.0,
        duration_ns=1_000_000_000,
    )
    frame_timeline = decode_grid._VideoTimeline(
        pts_stream=np.asarray([222], dtype=np.int64),
        timestamps_ns=np.asarray([0], dtype=np.int64),
        source_width=2,
        source_height=2,
        nominal_fps=1.0,
        duration_ns=1_000_000_000,
    )
    calls: list[tuple[int, bool]] = []

    def packet_probe(_source: Path, *, stream_index: int) -> decode_grid._VideoTimeline:
        assert stream_index == 0
        return packet_timeline

    def frame_probe(_source: Path, *, stream_index: int) -> decode_grid._VideoTimeline:
        assert stream_index == 0
        return frame_timeline

    def decode_once(
        _source: Path,
        *,
        selected_pts_stream: np.ndarray,
        seek: bool,
        **_kwargs: object,
    ) -> np.ndarray:
        pts = int(selected_pts_stream[0])
        calls.append((pts, seek))
        if pts == 111:
            message = "packet PTS is not a frame PTS"
            raise decode_grid._SelectedFrameDecodeError(message)
        return np.zeros((1, 2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(decode_grid, "_probe_packet_timeline", packet_probe)
    monkeypatch.setattr(decode_grid, "_probe_decoded_timeline", frame_probe)
    monkeypatch.setattr(decode_grid, "_decode_selected_frames_once", decode_once)

    decoded = decode_annotation_clip(
        Path("not-opened.mp4"),
        None,
        stream_index=0,
        rotation_degrees_clockwise=0,
        grid=AnnotationGrid(sample_fps=1.0, width=2, height=2),
        min_frames=1,
        max_frames=1,
    )

    assert calls == [(111, True), (111, False), (222, True)]
    assert decoded.decoder_backend == "pyav_seek_grid"
    assert decoded.source_span == (0.0, 1.0)


def test_timeline_prefers_the_last_observed_frame_duration() -> None:
    """Container padding must not extend a known final display-frame duration."""
    timeline = decode_grid._make_timeline(
        [(0, 2), (10, 4)],
        time_base=Fraction(1, 10),
        source_width=2,
        source_height=2,
        nominal_fps=None,
        stream_start_pts=0,
        stream_duration_pts=30,
        mismatch_error=ValueError,
    )

    assert timeline.duration_ns == 1_400_000_000


def test_irregular_timeline_uses_nearest_frame_and_left_tie_break() -> None:
    """VFR sampling should stay deterministic at gaps and exact midpoints."""
    source_timestamps_ns = np.asarray(
        [0, 40_000_000, 120_000_000, 160_000_000, 200_000_000],
        dtype=np.int64,
    )
    target_timestamps_ns = np.asarray(
        [0, 20_000_000, 60_000_000, 80_000_000, 100_000_000, 140_000_000],
        dtype=np.int64,
    )

    selected = decode_grid._select_nearest_source_indices(
        source_timestamps_ns,
        target_timestamps_ns,
        start_ns=0,
        stop_ns=200_000_000,
    )

    np.testing.assert_array_equal(selected, np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64))
