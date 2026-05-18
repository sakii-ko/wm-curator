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

"""Translate ``--event-caption-*`` CLI flags into a per-window inner stage.

Per-event captioning (:class:`PerEventCaptionStage`) consumes a
:class:`SingleInferenceCaptionStage` for inference. The four valid inner
stages (``VllmCaptionStage``, ``GeminiCaptionStage``, ``OpenAICaptionStage``,
``VllmAsyncCaptionStage``) each have different constructors with different
defaults; this module centralises the translation so the splitting
pipeline driver and the standalone ``sam3_event_pipeline`` example don't
diverge on per-event-specific knobs (Gemini retry policy, vLLM sampling
fps, Files API fallback, etc.).

The function :func:`build_event_caption_inner_stage` returns the inner
stage. Driver code is responsible for wrapping it in any ``CuratorStageSpec``
needed for scheduling (e.g. forcing a single async-API worker) before
appending to the stage list.
"""

import argparse

from cosmos_curator.pipelines.video.captioning import (
    gemini_caption_stage,
    vllm_caption_stage,
)
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import (
    GeminiCaptionStage,
    GeminiRetryPolicy,
)
from cosmos_curator.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curator.pipelines.video.captioning.single_inference import SingleInferenceCaptionStage
from cosmos_curator.pipelines.video.captioning.vllm_caption_stage import VllmCaptionStage
from cosmos_curator.pipelines.video.utils.data_model import VllmAsyncConfig, VllmConfig, VllmSamplingConfig


def _build_qwen_inner(args: argparse.Namespace, *, verbose: bool, log_stats: bool) -> VllmCaptionStage:
    """Build the Qwen (sync vLLM) inner stage with per-event defaults."""
    sampling_config = VllmSamplingConfig(max_tokens=4096)
    vllm_config = VllmConfig(
        model_variant=args.event_caption_qwen_variant,
        fp8=args.event_caption_qwen_fp8,
        sampling_config=sampling_config,
    )
    caption_single_options = vllm_caption_stage.CaptionSingleOptions(
        sampling_fps=args.event_caption_qwen_sampling_fps,
        temperature=args.event_caption_qwen_temperature,
        top_p=args.event_caption_qwen_top_p,
        top_k=args.event_caption_qwen_top_k,
    )
    return VllmCaptionStage(
        vllm_config=vllm_config,
        verbose=verbose,
        log_stats=log_stats,
        # Per-event ignores caption_quality_flags and the inflight_batching path.
        caption_quality_flags_enabled=False,
        caption_single_options=caption_single_options,
    )


def _build_gemini_inner(args: argparse.Namespace, *, verbose: bool, log_stats: bool) -> GeminiCaptionStage:
    """Build the Gemini inner stage with per-event-specific defaults.

    Per-event Gemini differs from the per-window default in several
    behaviour-significant ways (recorded so future drift is easy to
    spot):

    * ``response_mime_type="application/json"`` — per-event consumes
      a structured ``events`` array.
    * ``media_resolution="high"`` — required to OCR ``#id`` overlay
      labels on the SAM3 annotated video.
    * ``thinking_budget`` from CLI — Flash needs some thinking budget
      to disambiguate close-call events.
    * ``video_fps`` from CLI — Gemini's default ~1 fps misses
      sub-second events.
    * ``EXPONENTIAL_JITTER`` retry — per-event clips can take minutes
      and we want to ride out 503 spikes from concurrent workers
      without retrying in lockstep.
    * ``enable_files_api_fallback=True`` — long clips routinely
      exceed the 20 MB inline limit.
    """
    caption_single_options = gemini_caption_stage.CaptionSingleOptions(
        response_mime_type="application/json",
        media_resolution=args.event_caption_gemini_media_resolution,
        thinking_budget=args.event_caption_gemini_thinking_budget,
        video_fps=args.event_caption_gemini_fps,
        # 8 attempts x exponential backoff (base 4 s, max 120 s) + jitter gives
        # a ~9 min worst-case per clip, which straddles typical Flash 503
        # spikes without being wasteful.
        retry_policy=GeminiRetryPolicy.EXPONENTIAL_JITTER,
        retry_max_delay_seconds=120.0,
        retry_jitter_seconds=5.0,
        enable_files_api_fallback=True,
    )
    return GeminiCaptionStage(
        model_variant="gemini",
        model_name=args.event_caption_gemini_model_name,
        max_output_tokens=args.event_caption_gemini_max_output_tokens,
        max_caption_retries=8,
        retry_delay_seconds=4.0,
        verbose=verbose,
        log_stats=log_stats,
        caption_single_options=caption_single_options,
    )


def _build_openai_inner(args: argparse.Namespace, *, verbose: bool, log_stats: bool) -> OpenAICaptionStage:
    """Build the OpenAI-compatible inner stage with per-event defaults."""
    return OpenAICaptionStage(
        model_name=args.event_caption_openai_model_name,
        model_variant="openai",
        # Per-event owns its prompt construction; OpenAICaptionStage's prompt
        # is unused by caption_single (which takes the prompt as a method arg).
        prompt_text="",
        max_output_tokens=args.event_caption_openai_max_output_tokens,
        max_caption_retries=args.event_caption_openai_max_retries,
        retry_delay_seconds=args.event_caption_openai_retry_delay_seconds,
        endpoint_key=args.event_caption_openai_endpoint_key,
        verbose=verbose,
        log_stats=log_stats,
    )


def _build_vllm_async_inner(
    args: argparse.Namespace,
    vllm_async_config: VllmAsyncConfig | None,
    *,
    verbose: bool,
    log_stats: bool,
) -> SingleInferenceCaptionStage:
    """Build the vllm_async inner stage. Imported lazily because it pulls in vLLM."""
    if vllm_async_config is None:
        msg = "vllm_async_config is required when event-caption-backend='vllm_async'"
        raise ValueError(msg)
    # Lazy import: vllm_async_stage imports cosmos_xenna's continuous-stage
    # plumbing and (in the unified env) vLLM itself; keep CPU-only test
    # collection of this module fast by importing only on demand.
    from cosmos_curator.pipelines.video.captioning import vllm_async_stage as vllm_async_module  # noqa: PLC0415
    from cosmos_curator.pipelines.video.captioning.vllm_async_stage import VllmAsyncCaptionStage  # noqa: PLC0415

    caption_single_options = vllm_async_module.CaptionSingleOptions(
        sampling_fps=args.event_caption_vllm_async_sampling_fps,
        max_tokens=args.event_caption_vllm_async_max_output_tokens,
    )
    return VllmAsyncCaptionStage(
        serve_config=vllm_async_config,
        model_name=vllm_async_config.model_variant,
        verbose=verbose,
        log_stats=log_stats,
        caption_single_options=caption_single_options,
    )


def build_event_caption_inner_stage(
    args: argparse.Namespace,
    *,
    vllm_async_config: VllmAsyncConfig | None = None,
    verbose: bool = False,
    log_stats: bool = False,
) -> SingleInferenceCaptionStage:
    """Construct the per-window stage that ``PerEventCaptionStage`` will delegate to.

    Args:
        args: Parsed argparse namespace populated by
            :func:`cosmos_curator.pipelines.video.captioning.per_event_cli_args.add_event_caption_args`.
        vllm_async_config: Required when ``args.event_caption_backend ==
            "vllm_async"``; ignored otherwise. Drivers build this via
            :func:`build_vllm_async_config` so the
            ``--event-caption-vllm-async-*`` flags flow through.
        verbose: Forwarded to the inner stage's ``verbose``.
        log_stats: Forwarded to the inner stage's ``log_stats``.

    Returns:
        A constructed inner stage. Each backend's class subclasses
        :class:`SingleInferenceCaptionStage`, so the returned object can
        be passed directly as ``inner=`` to ``PerEventCaptionStage``.

    Raises:
        ValueError: On unrecognised backend or missing required config
            (e.g. ``vllm_async`` without a config).

    """
    backend = args.event_caption_backend
    if backend == "qwen":
        return _build_qwen_inner(args, verbose=verbose, log_stats=log_stats)
    if backend == "gemini":
        return _build_gemini_inner(args, verbose=verbose, log_stats=log_stats)
    if backend == "openai":
        return _build_openai_inner(args, verbose=verbose, log_stats=log_stats)
    if backend == "vllm_async":
        return _build_vllm_async_inner(args, vllm_async_config, verbose=verbose, log_stats=log_stats)
    msg = f"Unsupported event-caption backend: {backend!r}"
    raise ValueError(msg)


__all__ = [
    "build_event_caption_inner_stage",
]
