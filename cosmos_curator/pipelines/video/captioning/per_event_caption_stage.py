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

"""Per-event VLM captioning stage.

Generates a structured ``events`` array referencing SAM3 object IDs for
each clip. The stage owns only the per-event-specific surface — SAM3
prompt construction and event JSON parsing — and delegates the actual
``(prompt, mp4 bytes) -> str`` inference to an inner
:class:`~cosmos_curator.pipelines.video.captioning.single_inference.SingleInferenceCaptionStage`
implementation.

Concretely the inner stage is one of the per-window caption stages:

* :class:`~cosmos_curator.pipelines.video.captioning.vllm_caption_stage.VllmCaptionStage`
  (local Qwen2.5-VL via sync vLLM)
* :class:`~cosmos_curator.pipelines.video.captioning.vllm_async_stage.VllmAsyncCaptionStage`
  (in-process AsyncLLM with TP/DP for Qwen3-VL families)
* :class:`~cosmos_curator.pipelines.video.captioning.gemini_caption_stage.GeminiCaptionStage`
  (remote Google Gemini API)
* :class:`~cosmos_curator.pipelines.video.captioning.openai_caption_stage.OpenAICaptionStage`
  (any OpenAI-compatible chat-completion endpoint)

The driver pipelines (:mod:`cosmos_curator.pipelines.video.splitting_pipeline`
and :mod:`cosmos_curator.pipelines.examples.sam3_event_pipeline`) translate
``--event-caption-*`` CLI flags into the appropriate per-window stage
constructor and pass the result as ``inner`` to this stage.
"""

import json
import re
from importlib import resources as importlib_resources
from typing import Any

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.pipelines.video.captioning.single_inference import SingleInferenceCaptionStage
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask

_DEFAULT_PROMPT_RESOURCE = "traffic_surveillance.md"
_DEFAULT_EVENT_PROMPT = (
    importlib_resources.files("cosmos_curator.pipelines.video.captioning.prompts")
    .joinpath(_DEFAULT_PROMPT_RESOURCE)
    .read_text(encoding="utf-8")
)

# Max length of the raw-response preview logged when extraction yields 0 events.
_RAW_PREVIEW_MAX_CHARS = 1500


def _build_instances_block(clip: Clip) -> str:
    """Serialize the clip's SAM3 instances as a compact JSON block for the VLM.

    The annotated video carries spatial grounding; this block is text-level
    grounding only. Each entry keeps four fields (``object_id``, ``class``,
    ``start_time_s``, ``end_time_s``) — ``num_frames`` is dropped to keep
    the prompt short.
    """
    instances = clip.sam3_instances or []
    payload: dict[str, Any] = {
        "instances": [
            {
                "object_id": entry.get("object_id"),
                "class": entry.get("prompt", "?"),
                "start_time_s": entry.get("start_time_s"),
                "end_time_s": entry.get("end_time_s"),
            }
            for entry in instances
            if isinstance(entry.get("object_id"), int)
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def _build_prompt(user_template: str, clip: Clip) -> str:
    """Append clip duration + instances JSON to the user prompt template.

    Duration is explicit so the VLM doesn't hallucinate out-of-range times.
    """
    instances_block = _build_instances_block(clip)
    duration_s = float(clip.duration) if clip.duration else 0.0
    return (
        f"{user_template.strip()}\n\n"
        f"======================================================================\n"
        f"CLIP DURATION: {duration_s:.2f} seconds.\n"
        f"All ``start_time`` / ``end_time`` values MUST lie within [0.0, {duration_s:.2f}].\n"
        f"======================================================================\n"
        f"TRACKED INSTANCES (id -> class -> visibility interval in seconds)\n"
        f"======================================================================\n"
        f"{instances_block}\n"
    )


def _extract_events_payload(text: str) -> list[Any]:
    """Pull the ``events`` array out of a model response.

    Accepts either ``{"events": [...], ...}`` (extra keys ignored) or a bare
    ``[...]`` array. Returns ``[]`` on unparseable input.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # ``strict=False`` tolerates literal newlines/tabs inside long caption
    # strings (Gemini 2.5 Flash occasionally forgets to escape them).
    try:
        parsed = json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if match is None:
            return []
        try:
            parsed = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        events = parsed.get("events", [])
        return events if isinstance(events, list) else []
    if isinstance(parsed, list):
        return parsed
    return []


class PerEventCaptionStage(CuratorStage):
    """Generate per-event captions referencing SAM3 object IDs.

    Runs after ``SAM3BBoxStage`` and consumes the clip's annotated
    ``tracked.mp4`` (the only video the VLM sees — the ``#id`` overlay
    is load-bearing for the prompt) plus the clip's SAM3 instances list
    to produce a structured ``events`` array.

    The stage holds an inner :class:`SingleInferenceCaptionStage`
    (typically one of the per-window caption stages) and delegates
    inference to it via
    :meth:`SingleInferenceCaptionStage.caption_single`. Resource
    requirements, conda env, and weight-download metadata are forwarded
    from the inner stage so Xenna sees this stage's effective footprint
    instead of a bespoke one.
    """

    def __init__(
        self,
        *,
        inner: SingleInferenceCaptionStage,
        prompt_text: str | None = None,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialise the stage.

        Args:
            inner: A per-window caption stage subclassing
                :class:`SingleInferenceCaptionStage`. The driver
                constructs this from ``--event-caption-*`` flags.
                Per-event uses it for inference; the inner stage's
                ``stage_setup_on_node`` / ``stage_setup`` / ``destroy``
                / ``resources`` / ``model`` / ``conda_env_name`` are
                forwarded directly.
            prompt_text: User prompt template. Falls back to the
                bundled ``traffic_surveillance.md`` default. The
                instances JSON block + clip duration are always
                appended at prompt-build time.
            verbose: Emit per-clip logs.
            log_stats: Record stage performance stats.

        """
        super().__init__()
        self._inner = inner
        self._prompt_template = prompt_text if prompt_text is not None else _DEFAULT_EVENT_PROMPT
        self._verbose = verbose
        self._log_stats = log_stats
        self._timer = StageTimer(self)

    @property
    def resources(self) -> CuratorStageResource:
        """Forward the inner stage's resource requirements (CPU / GPU)."""
        return self._inner.resources

    @property
    def conda_env_name(self) -> str | None:
        """Forward the inner stage's conda env so weight download / setup works."""
        return self._inner.conda_env_name

    @property
    def model(self) -> ModelInterface | None:  # type: ignore[override]
        """Expose the inner stage's ``ModelInterface`` so weights are auto-downloaded."""
        return self._inner.model

    def secondary_name(self) -> str:
        """Return the inner stage's secondary name, suffixed for log clarity."""
        return f"per_event:{self._inner.secondary_name()}"

    def stage_setup_on_node(self) -> None:
        """Forward to ``inner.stage_setup_on_node`` so per-node weight copy still runs.

        ``VllmCaptionStage.stage_setup_on_node`` does the per-node weight
        copy + engine pre-init that is load-bearing on multi-node Slurm
        runs; without this forward, the inner stage's per-node hook is
        silently skipped.
        """
        super().stage_setup_on_node()
        self._inner.stage_setup_on_node()

    def stage_setup(self) -> None:
        """Forward to ``inner.stage_setup`` — per-event keeps no per-backend state."""
        super().stage_setup()
        self._inner.stage_setup()

    def destroy(self) -> None:
        """Forward to ``inner.destroy`` so engine / client shutdown still runs."""
        self._inner.destroy()

    def _process_clip(self, clip: Clip) -> None:
        """Build per-event prompt, run one inference, and write events JSON."""
        if not clip.sam3_instances:
            if self._verbose:
                logger.debug(f"[PerEventCaptionStage] clip {clip.uuid}: no SAM3 instances; skipping")
            return

        # The annotated ``tracked.mp4`` is the only video the VLM sees — the
        # ``#id`` overlay is load-bearing for the prompt. Fail fast rather
        # than silently falling back to the raw clip if SAM3 skipped it.
        annotated = clip.sam3_annotated_video.resolve()
        if annotated is None:
            clip.errors["per_event_caption"] = "missing_annotated_video"
            logger.warning(f"[PerEventCaptionStage] clip {clip.uuid}: sam3_annotated_video missing; skipping")
            return
        mp4_bytes = bytes(annotated.tobytes())
        if self._verbose:
            logger.debug(
                f"[PerEventCaptionStage] clip {clip.uuid}: feeding annotated video to VLM ({len(mp4_bytes)} bytes)"
            )

        prompt = _build_prompt(self._prompt_template, clip)
        try:
            raw = self._inner.caption_single(prompt, mp4_bytes)
        except Exception as exc:  # noqa: BLE001
            clip.errors["per_event_caption"] = f"api_error: {exc!r}"
            logger.exception(f"[PerEventCaptionStage] clip {clip.uuid}: caption_single failed")
            return

        events = _extract_events_payload(raw)
        clip.sam3_events = events
        if not events:
            self._log_empty_events(clip, raw)
        elif self._verbose:
            inner_name = type(self._inner).__name__
            logger.info(f"[PerEventCaptionStage] clip {clip.uuid}: {len(events)} events (inner={inner_name})")

    @staticmethod
    def _looks_truncated(raw: str) -> bool:
        """Heuristic: response got chopped mid-JSON (usually MAX_TOKENS)."""
        stripped = raw.strip().rstrip("` \n")
        if not stripped:
            return False
        return stripped[-1] not in {"}", "]"}

    def _log_empty_events(self, clip: Clip, raw: str) -> None:
        """Log the raw model response when no events were parsed."""
        preview = raw.strip().replace("\n", " ")
        if len(preview) > _RAW_PREVIEW_MAX_CHARS:
            preview = preview[:_RAW_PREVIEW_MAX_CHARS] + "...[truncated]"
        truncated = self._looks_truncated(raw)
        diag_parts = [f"raw_len={len(raw)}"]
        # Surface Gemini-specific diagnostics (finish reason / usage) when
        # the inner stage is GeminiCaptionStage and exposes them.
        finish_reasons = getattr(self._inner, "last_finish_reasons", None)
        if finish_reasons:
            diag_parts.append(f"finish_reasons={finish_reasons}")
        usage = getattr(self._inner, "last_usage_metadata", None)
        if usage is not None:
            diag_parts.append(f"usage={usage!s}")
        diagnostics = ", ".join(diag_parts)
        hint = ""
        if truncated:
            hint = " [likely MAX_TOKENS truncation — raise the inner stage's max_output_tokens]"
        inner_name = type(self._inner).__name__
        logger.warning(
            f"[PerEventCaptionStage] clip {clip.uuid}: extracted 0 events from "
            f"{inner_name} response ({diagnostics}){hint}: {preview!r}"
        )
        clip.errors.setdefault(
            "per_event_caption",
            "truncated_response" if truncated else "empty_or_unparseable_response",
        )

    @nvtx.annotate("PerEventCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:  # type: ignore[override]
        """Generate per-event captions for every clip in every task."""
        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)
            with self._timer.time_process():
                for video in task.videos:
                    for clip in video.clips:
                        self._process_clip(clip)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
