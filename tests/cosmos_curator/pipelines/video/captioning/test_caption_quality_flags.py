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
"""Tests for heuristic caption quality flag annotations."""

import pytest

from cosmos_curator.pipelines.video.captioning.caption_quality_flags import apply_caption_quality_flags
from cosmos_curator.pipelines.video.utils.data_model import Window


def _window(
    caption: str | None,
    *,
    status: str | None = "success",
    model_variant: str = "qwen",
    start_frame: int = 0,
    end_frame: int = 10,
) -> Window:
    captions = {model_variant: caption} if caption is not None else {}
    return Window(start_frame=start_frame, end_frame=end_frame, caption=captions, caption_status=status)


def test_normalization_trims_whitespace_case_and_trailing_punctuation() -> None:
    """Whitespace, case, and trailing punctuation should normalize before checks."""
    window = _window("  A   Car   Drives   Slowly!!!  ")
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_length_outlier is False
    assert window.flag_repetition is False
    assert window.flag_near_duplicate is False


def test_truncated_status_is_evaluable() -> None:
    """Truncated captions should still receive quality flags."""
    window = _window("alpha beta gamma delta", status="truncated")
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_length_outlier is False
    assert window.flag_repetition is False
    assert window.flag_near_duplicate is False


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("car driving", True),
        ("one two three four", False),
        (" ".join(f"word{i}" for i in range(1025)), True),
        (" ".join(f"word{i}" for i in range(1024)), False),
    ],
    ids=["below_floor", "at_floor", "above_ceiling", "at_ceiling"],
)
def test_length_floor_and_ceiling(caption: str, *, expected: bool) -> None:
    """Length flag should trigger only outside the word floor and ceiling."""
    window = _window(caption)
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_length_outlier is expected


def test_repeated_word_dominance_sets_repetition_flag() -> None:
    """One word dominating a caption should set the repetition flag."""
    window = _window("road road road road car sky")
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_repetition is True


def test_repeated_trigram_sets_repetition_flag() -> None:
    """A repeated trigram loop should set the repetition flag."""
    window = _window("a man walks a man walks a man walks")
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_repetition is True


def test_low_unique_word_ratio_sets_repetition_flag() -> None:
    """Low unique-word ratio should set the repetition flag."""
    window = _window("red red red red blue blue blue blue green")
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_repetition is True


def test_adjacent_near_duplicate_exact_match_flags_both_windows() -> None:
    """Exact adjacent duplicate captions should flag both windows."""
    first = _window("a white sedan turns through the intersection", start_frame=0, end_frame=10)
    second = _window("a white sedan turns through the intersection", start_frame=10, end_frame=20)
    apply_caption_quality_flags([[first, second]], "qwen")

    assert first.flag_near_duplicate is True
    assert second.flag_near_duplicate is True


def test_adjacent_near_duplicate_jaccard_behavior() -> None:
    """Adjacent captions at or above the Jaccard threshold should flag both windows."""
    first = _window("alpha beta gamma delta epsilon zeta eta theta iota kappa", start_frame=0, end_frame=10)
    near = _window("kappa iota theta eta zeta epsilon delta gamma beta alpha", start_frame=10, end_frame=20)
    below = _window("alpha beta gamma delta epsilon zeta eta theta iota lambda", start_frame=20, end_frame=30)
    apply_caption_quality_flags([[first, near, below]], "qwen")

    assert first.flag_near_duplicate is True
    assert near.flag_near_duplicate is True
    assert below.flag_near_duplicate is False


def test_near_duplicate_uses_sorted_adjacency_not_incidental_list_order() -> None:
    """Near-duplicate comparisons should use temporal order, not input list order."""
    first = _window("alpha beta gamma delta", start_frame=0, end_frame=10)
    middle = _window("one two three four", start_frame=10, end_frame=20)
    third = _window("alpha beta gamma delta", start_frame=20, end_frame=30)
    apply_caption_quality_flags([[first, third, middle]], "qwen")

    assert first.flag_near_duplicate is False
    assert middle.flag_near_duplicate is False
    assert third.flag_near_duplicate is False


def test_near_duplicate_tie_breaks_same_frames_by_original_index() -> None:
    """Windows with identical frame bounds should use original index as tie-breaker."""
    first = _window("alpha beta gamma delta", start_frame=0, end_frame=10)
    second = _window("alpha beta gamma delta", start_frame=0, end_frame=10)
    third = _window("one two three four", start_frame=0, end_frame=10)
    apply_caption_quality_flags([[first, second, third]], "qwen")

    assert first.flag_near_duplicate is True
    assert second.flag_near_duplicate is True
    assert third.flag_near_duplicate is False


def test_evaluable_window_adjacent_to_non_evaluable_remains_false() -> None:
    """Comparison with a non-evaluable neighbor does not flag the evaluable side."""
    first = _window("alpha beta gamma delta", start_frame=0, end_frame=10)
    middle = _window(None, status="skipped", start_frame=10, end_frame=20)
    third = _window("alpha beta gamma delta", start_frame=20, end_frame=30)
    apply_caption_quality_flags([[first, middle, third]], "qwen")

    assert first.flag_near_duplicate is False
    assert middle.flag_near_duplicate is None
    assert third.flag_near_duplicate is False


def test_adjacent_only_within_clip_not_across_groups() -> None:
    """Identical captions across clip boundaries do not flag as near-duplicate."""
    boundary_text = "alpha beta gamma delta epsilon"
    clip1_w1 = _window("one two three four", start_frame=0, end_frame=10)
    clip1_w2 = _window(boundary_text, start_frame=10, end_frame=20)
    clip2_w1 = _window(boundary_text, start_frame=0, end_frame=10)
    clip2_w2 = _window("five six seven eight", start_frame=10, end_frame=20)

    apply_caption_quality_flags([[clip1_w1, clip1_w2], [clip2_w1, clip2_w2]], "qwen")

    assert clip1_w2.flag_near_duplicate is False
    assert clip2_w1.flag_near_duplicate is False


def test_missing_model_key_leaves_flags_unset() -> None:
    """Missing active model key should leave all flags unset without fallback."""
    window = _window("a valid caption text", model_variant="other")
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_length_outlier is None
    assert window.flag_repetition is None
    assert window.flag_near_duplicate is None


@pytest.mark.parametrize("caption", ["", "   ", "!!!"])
def test_empty_normalized_caption_leaves_flags_unset(caption: str) -> None:
    """Empty normalized caption text should leave all flags unset."""
    window = _window(caption)
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_length_outlier is None
    assert window.flag_repetition is None
    assert window.flag_near_duplicate is None


@pytest.mark.parametrize("status", [None, "blocked", "error", "skipped"])
def test_non_evaluable_status_leaves_flags_unset(status: str | None) -> None:
    """Non-evaluable statuses should leave all flags unset."""
    window = _window("a valid caption text", status=status)
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_length_outlier is None
    assert window.flag_repetition is None
    assert window.flag_near_duplicate is None


def test_active_model_key_isolation() -> None:
    """Only the active caption key should be evaluated."""
    window = Window(
        start_frame=0,
        end_frame=10,
        caption={"qwen": "car", "other": "a complete useful caption"},
        enhanced_caption={"qwen": "a complete useful enhanced caption"},
        caption_status="success",
    )
    apply_caption_quality_flags([[window]], "qwen")

    assert window.flag_length_outlier is True
    assert window.flag_repetition is False
    assert window.flag_near_duplicate is False
