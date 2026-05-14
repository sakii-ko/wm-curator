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

"""Focused tests for vLLM async builder-level config validation."""

import attrs
import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.pipelines.video.captioning.captioning_builders import (
    CaptioningConfig,
    VllmAsyncCaptionConfig,
    build_captioning_stages,
)
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage
from cosmos_curator.pipelines.video.captioning.vllm_async_stage import VllmAsyncCaptionStage
from cosmos_curator.pipelines.video.captioning.vllm_caption_stage import VllmPrepStage
from cosmos_curator.pipelines.video.preview.preview_stages import PreviewStage
from cosmos_curator.pipelines.video.utils.data_model import VllmAsyncConfig, WindowConfig


def test_vllm_async_caption_config_default_uses_zero_for_auto_mode() -> None:
    """Default config uses ``0`` as the auto-derive sentinel."""
    cfg = VllmAsyncCaptionConfig()
    assert cfg.max_concurrent_requests == 0


def test_vllm_async_caption_config_accepts_explicit_positive() -> None:
    """Positive ``int`` overrides auto-derive."""
    cfg = VllmAsyncCaptionConfig(max_concurrent_requests=128)
    assert cfg.max_concurrent_requests == 128


def test_vllm_async_caption_config_rejects_negative() -> None:
    """Validator ``ge(0)``: 0 is valid (auto-derive), negatives are invalid."""
    with pytest.raises(ValueError, match="must be"):
        VllmAsyncCaptionConfig(max_concurrent_requests=-1)


def test_num_workers_per_node_default_is_zero() -> None:
    """Default num_workers_per_node is 0 (auto-derive sentinel: Xenna autoscale)."""
    cfg = VllmAsyncCaptionConfig()
    assert cfg.num_workers_per_node == 0


def _make_config(
    data_parallel_size: int = 1,
    num_gpus: int = 1,
    num_workers_per_node: int = 0,
    *,
    generate_previews: bool = False,
) -> CaptioningConfig:
    serve_config = VllmAsyncConfig(
        model_variant="qwen",
        num_gpus=num_gpus,
        data_parallel_size=data_parallel_size,
    )
    backend = VllmAsyncCaptionConfig(
        serve_config=serve_config,
        num_workers_per_node=num_workers_per_node,
    )
    return CaptioningConfig(
        backend=backend,
        window_config=WindowConfig(),
        generate_previews=generate_previews,
    )


class TestBuildersNumWorkersPerNode:
    """Verify builders set num_workers_per_node based on mode and config."""

    def test_explicit_workers(self) -> None:
        """Explicit positive value -> exact worker count."""
        cfg = _make_config(num_workers_per_node=7)
        stages = build_captioning_stages(cfg)
        caption_stage = stages[-1]
        assert isinstance(caption_stage, CuratorStageSpec)
        assert caption_stage.num_workers_per_node == 7

    def test_dp_mode_ignores_num_workers(self) -> None:
        """DP mode (dp=7): always 1 worker, num_workers_per_node ignored."""
        cfg = _make_config(data_parallel_size=7, num_workers_per_node=5)
        stages = build_captioning_stages(cfg)
        caption_stage = stages[-1]
        assert isinstance(caption_stage, CuratorStageSpec)
        assert caption_stage.num_workers_per_node == 1

    def test_prep_stage_receives_windowing_fields(self) -> None:
        """``VllmPrepStage`` exposes the builder's windowing config and mp4 flag.

        The captioning builder routes ``CaptioningConfig.window_config`` into
        ``stage._window_config`` and ``CaptioningConfig.keep_mp4`` (combined
        with ``generate_previews``) into ``stage._keep_mp4``.
        """
        cfg = _make_config()
        stages = build_captioning_stages(cfg)
        prep_spec = stages[0]
        assert isinstance(prep_spec, CuratorStageSpec)
        stage = prep_spec.stage
        assert isinstance(stage, VllmPrepStage)
        assert stage._window_config.window_size == 256
        assert stage._window_config.remainder_threshold == 128
        assert stage._keep_mp4 is False

    def test_build_stages_vllm_async_prep_is_first(self) -> None:
        """``vllm_async`` ``build_stages`` produces Prep + Caption (2 stages) with ``VllmPrepStage``."""
        cfg = _make_config()
        stages = build_captioning_stages(cfg)
        assert len(stages) == 2
        assert isinstance(stages[0], CuratorStageSpec)
        assert isinstance(stages[0].stage, VllmPrepStage)
        assert isinstance(stages[1], CuratorStageSpec)
        assert isinstance(stages[1].stage, VllmAsyncCaptionStage)

    def test_build_stages_vllm_async_no_api_prep_stage(self) -> None:
        """vllm_async pipeline should NOT include ApiPrepStage."""
        cfg = _make_config()
        stages = build_captioning_stages(cfg)
        for s in stages:
            stage_obj = s.stage if isinstance(s, CuratorStageSpec) else s
            assert not isinstance(stage_obj, ApiPrepStage)

    def test_build_stages_vllm_async_with_previews(self) -> None:
        """Enabling previews produces Prep + Preview + Caption (3 stages)."""
        cfg = _make_config(generate_previews=True)
        stages = build_captioning_stages(cfg)
        assert len(stages) == 3
        assert isinstance(stages[0], CuratorStageSpec)
        assert isinstance(stages[0].stage, VllmPrepStage)
        assert isinstance(stages[1], CuratorStageSpec)
        assert isinstance(stages[1].stage, PreviewStage)
        assert isinstance(stages[2], CuratorStageSpec)
        assert isinstance(stages[2].stage, VllmAsyncCaptionStage)

    def test_prep_config_keep_mp4_from_generate_previews(self) -> None:
        """``generate_previews=True`` forces ``VllmPrepStage._keep_mp4 = True``.

        Preview runs BEFORE captioning, so the prep stage owns keeping
        bytes available for it -- see ``_build_vllm_async_prep_stage``
        wiring ``keep_mp4=config.generate_previews or config.keep_mp4``.
        """
        cfg = _make_config(generate_previews=True)
        stages = build_captioning_stages(cfg)
        prep_spec = stages[0]
        assert isinstance(prep_spec, CuratorStageSpec)
        stage = prep_spec.stage
        assert isinstance(stage, VllmPrepStage)
        assert stage._keep_mp4 is True

    def test_caption_stage_keep_mp4_propagates_from_config(self) -> None:
        """``CaptioningConfig.keep_mp4`` flows directly to ``VllmAsyncCaptionStage._keep_mp4``.

        Mirrors sync ``_build_vllm_caption_stage``: caption stage drops
        bytes by default but preserves them when a downstream consumer
        (e.g. metadata writer with ``--generate-cosmos-predict-dataset``)
        opts in via ``keep_mp4=True``.
        """
        cfg = _make_config()
        cfg = attrs.evolve(cfg, keep_mp4=True)  # type: ignore[arg-type]
        stages = build_captioning_stages(cfg)
        caption_spec = stages[-1]
        assert isinstance(caption_spec, CuratorStageSpec)
        stage = caption_spec.stage
        assert isinstance(stage, VllmAsyncCaptionStage)
        assert stage._keep_mp4 is True

    def test_caption_stage_keep_mp4_independent_of_generate_previews(self) -> None:
        """``generate_previews=True`` alone must NOT keep mp4 bytes through the caption stage.

        Preview consumes bytes BEFORE captioning runs, so passing
        ``generate_previews=True`` only forces the PREP stage to keep
        bytes -- the caption stage should still drop them unless
        ``config.keep_mp4`` is set explicitly upstream
        (``splitting_pipeline.py:628`` already folds the relevant flags
        into ``config.keep_mp4`` when needed).
        """
        cfg = _make_config(generate_previews=True)
        stages = build_captioning_stages(cfg)
        caption_spec = stages[-1]
        assert isinstance(caption_spec, CuratorStageSpec)
        stage = caption_spec.stage
        assert isinstance(stage, VllmAsyncCaptionStage)
        assert stage._keep_mp4 is False
