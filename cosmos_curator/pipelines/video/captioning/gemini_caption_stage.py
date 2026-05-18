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
"""Remote API-backed caption preparation and captioning stages (Gemini)."""

import asyncio
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import nvtx  # type: ignore[import-untyped]
import tenacity
from google import genai
from google.api_core.exceptions import DeadlineExceeded
from google.genai import types as genai_types
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, PipelineTask
from cosmos_curator.core.utils.config.config import load_config
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.misc.memfd import buffer_as_memfd_path
from cosmos_curator.models.prompts import get_prompt
from cosmos_curator.pipelines.common.api_caption_utils import (
    gemini_error_result_from_exception,
    handle_gemini_client_exception,
    normalize_gemini_response_with_detail,
    should_retry_gemini_exception,
)
from cosmos_curator.pipelines.common.api_stage_async_utils import destroy_api_clients
from cosmos_curator.pipelines.video.captioning.single_inference import SingleInferenceCaptionStage
from cosmos_curator.pipelines.video.utils import windowing_utils
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionOutcome,
    CaptionResult,
    Clip,
    SplitPipeTask,
    Video,
    Window,
    WindowConfig,
    get_video_from_task,
)

TTask = TypeVar("TTask", bound=PipelineTask)


class GeminiRetryPolicy(enum.Enum):
    """Retry style used by Gemini caption requests.

    ``FIXED``: the legacy per-window default — N attempts with a fixed
    ``retry_delay_seconds`` between them, retrying on
    :func:`should_retry_gemini_exception`. Cheap and predictable for
    high-volume async windows where each individual request is short.

    ``EXPONENTIAL_JITTER``: exponential backoff (base
    ``retry_delay_seconds``, capped at ``retry_max_delay_seconds``) plus
    uniform jitter (``[0, retry_jitter_seconds)``). Used for one-shot
    callers (e.g. ``PerEventCaptionStage``) where a clip can take
    minutes and we want to ride out 5xx / 429 spikes from concurrent
    workers without retrying in lockstep.
    """

    FIXED = "fixed"
    EXPONENTIAL_JITTER = "exponential_jitter"


_GEMINI_TRANSIENT_HTTP_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
_GEMINI_TRANSIENT_EXCEPTION_NAMES: frozenset[str] = frozenset(
    {"APIError", "ConnectionError", "TimeoutError", "ReadTimeout"},
)


def _is_transient_gemini_exception(exc: BaseException) -> bool:
    """Return True for transient (retry-worthy) Gemini exceptions.

    Mirrors the per-event retry predicate: retries 5xx / 429 / transport
    hiccups but lets 4xx auth/bad-prompt errors fail fast.
    """
    from google.genai import errors as genai_errors  # noqa: PLC0415

    if isinstance(exc, genai_errors.ServerError):
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _GEMINI_TRANSIENT_HTTP_CODES:
        return True
    return type(exc).__name__ in _GEMINI_TRANSIENT_EXCEPTION_NAMES


@dataclass(frozen=True)
class _WindowCaptionTask:
    """One captioning work item bound to a specific window."""

    clip: Clip
    window: Window
    window_index: int


@dataclass(frozen=True)
class CaptionSingleOptions:
    """Per-call overrides applied only by ``GeminiCaptionStage.caption_single``.

    Groups the eight Gemini-specific knobs that one-shot consumers
    (``PerEventCaptionStage``) need to tweak for SAM3 traffic-event
    captions: structured-output coercion, media-resolution preservation
    of small overlays, 2.5-series thinking budget, video sampling fps,
    and an exponential-jitter retry policy designed for long-running
    clips that ride out concurrent 503 spikes.

    All fields default to the per-window-friendly values; the per-window
    batch path bypasses this struct entirely.
    """

    response_mime_type: str | None = None
    media_resolution: str | None = None
    thinking_budget: int | None = None
    video_fps: float | None = None
    retry_policy: GeminiRetryPolicy = GeminiRetryPolicy.FIXED
    retry_max_delay_seconds: float = 120.0
    retry_jitter_seconds: float = 5.0
    enable_files_api_fallback: bool = False


# Module-level singleton used as the constructor default. Frozen
# dataclasses are safe to share across instances; using a singleton
# satisfies ruff B008 (no mutable function-call defaults).
_DEFAULT_CAPTION_SINGLE_OPTIONS = CaptionSingleOptions()


class ApiPrepStage(CuratorStage):
    """Stage that prepares windows for remote API captioning."""

    def __init__(  # noqa: PLR0913
        self,
        window_config: WindowConfig,
        *,
        model_variant: str = "gemini",
        num_cpus_for_prepare: float = 1.0,
        use_filter_windows: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the API prep stage."""
        super().__init__()
        self._timer = StageTimer(self)
        self._window_config = window_config
        self._model_variant = model_variant
        self._num_cpus_for_prepare = num_cpus_for_prepare
        self._use_filter_windows = use_filter_windows
        self._verbose = verbose
        self._log_stats = log_stats

    def secondary_name(self) -> str:
        """Return the model variant for logging."""
        return self._model_variant

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=self._num_cpus_for_prepare)

    @property
    def conda_env_name(self) -> str:
        """Use the unified environment for window preparation."""
        return "unified"

    def _prep_windows(self, video: Video) -> None:
        """Create windows for the provided video."""
        num_video_decode_threads = max(1, int(self.resources.cpus) + 1)
        windows, _ = windowing_utils.make_windows_for_video(
            video,
            self._window_config,
            num_video_decode_threads,
            keep_mp4=True,
            return_frames=False,
        )
        if self._verbose:
            logger.debug(f"Prepared {len(windows)} windows for {video.input_video}")
        if self._use_filter_windows:
            for clip in video.clips:
                clip.filter_windows = clip.windows[:]
                clip.windows = []

    @nvtx.annotate("ApiPrepStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[TTask]) -> list[TTask]:
        """Prepare data for API captioning."""
        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)
            video = get_video_from_task(task)
            with self._timer.time_process():
                self._prep_windows(video)

            stage_perf = getattr(task, "stage_perf", None)
            if self._log_stats and stage_perf is not None:
                stage_name, stage_perf_stats = self._timer.log_stats()
                stage_perf[stage_name] = stage_perf_stats

        return tasks


_VALID_MEDIA_RESOLUTIONS: tuple[str, ...] = ("low", "medium", "high")


class GeminiCaptionStage(SingleInferenceCaptionStage):
    """Caption video windows using the Google Gemini API.

    The Gemini API key must be provided in the cosmos-curator config file under the `gemini` section.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_variant: str = "gemini",
        model_name: str = "models/gemini-2.5-pro",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        max_output_tokens: int = 4096,
        max_caption_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        max_video_size_bytes: int = 20 * 1024 * 1024,
        batch_size: int = 1,
        use_filter_windows: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
        caption_single_options: CaptionSingleOptions = _DEFAULT_CAPTION_SINGLE_OPTIONS,
    ) -> None:
        """Initialize the API caption stage.

        Args:
            model_variant: Identifier stored alongside generated captions.
            model_name: Gemini model name to invoke.
            prompt_variant: Prompt variant used to build caption instructions.
            prompt_text: Optional custom prompt text.
            max_output_tokens: Maximum output tokens requested from the API.
            max_caption_retries: Number of retries per window before giving up.
            retry_delay_seconds: Delay between retries (FIXED policy) or
                exponential-backoff multiplier (EXPONENTIAL_JITTER policy).
            max_video_size_bytes: Maximum inline video size supported by the
                API. ``process_data`` rejects oversize windows; ``caption_single``
                routes them through the Files API when
                ``caption_single_options.enable_files_api_fallback`` is True.
            batch_size: Stage batch size and async concurrency limit.
            use_filter_windows: If True, iterate clip.filter_windows instead of clip.windows.
            verbose: Emit verbose logging.
            log_stats: Whether to record stage performance statistics.
            caption_single_options: Per-call overrides honoured only by
                :meth:`caption_single` (Gemini-specific MIME type,
                media resolution, thinking budget, video fps, retry
                policy, Files API fallback). Defaults to the
                per-window-friendly empty struct so the
                ``process_data`` path is unaffected.

        """
        super().__init__()
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._model_name = model_name
        self._use_filter_windows = use_filter_windows
        self._prompt_variant = prompt_variant
        self._prompt_text = prompt_text
        self._prompt = get_prompt(prompt_variant, prompt_text, verbose=verbose)
        self._max_output_tokens = max_output_tokens
        self._max_caption_retries = max_caption_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._max_video_size_bytes = max_video_size_bytes
        self._batch_size = max(1, batch_size)
        self._verbose = verbose
        self._log_stats = log_stats
        # caption_single knobs (unpacked from the dataclass — helpers
        # below read the per-field private attrs unchanged).
        self._response_mime_type = caption_single_options.response_mime_type
        media_resolution = caption_single_options.media_resolution
        if media_resolution is not None and media_resolution.strip().lower() not in _VALID_MEDIA_RESOLUTIONS:
            msg = f"media_resolution must be one of {_VALID_MEDIA_RESOLUTIONS} or None; got {media_resolution!r}"
            raise ValueError(msg)
        self._media_resolution = media_resolution.strip().lower() if media_resolution is not None else None
        self._thinking_budget = caption_single_options.thinking_budget
        self._video_fps = caption_single_options.video_fps
        self._retry_policy = caption_single_options.retry_policy
        self._retry_max_delay_seconds = caption_single_options.retry_max_delay_seconds
        self._retry_jitter_seconds = caption_single_options.retry_jitter_seconds
        self._enable_files_api_fallback = caption_single_options.enable_files_api_fallback
        # Diagnostics populated by ``caption_single`` on each call.
        self._last_finish_reasons: list[str] = []
        self._last_usage_metadata: object | None = None
        config = load_config()
        if config.gemini is None or not config.gemini.api_key:
            msg = "Gemini API key missing from config file."
            raise RuntimeError(msg)
        self._api_key = config.gemini.api_key
        self._client: genai.Client | None = None
        self._async_client: Any | None = None
        self._runner: asyncio.Runner | None = None

    def secondary_name(self) -> str:
        """Return the model variant for logging."""
        return self._model_variant

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def stage_batch_size(self) -> int:
        """Return the batch size used for scheduling and async concurrency."""
        return self._batch_size

    def stage_setup(self) -> None:
        """Create the Gemini API client."""
        self._client = genai.Client(api_key=self._api_key)
        self._async_client = self._client.aio
        self._runner = asyncio.Runner()

    @staticmethod
    def _write_caption_result(window: Window, model_variant: str, result: CaptionResult) -> None:
        """Write a Gemini caption result onto a window."""
        if result.text is not None:
            window.caption[model_variant] = result.text
        window.caption_status = result.outcome.value
        window.caption_failure_reason = result.failure_reason if result.outcome == CaptionOutcome.ERROR else None

    async def _generate_caption_with_error_detail_async(self, window: Window) -> tuple[CaptionResult, str | None]:
        """Generate a caption result for a single window using the async client."""
        client = self._async_client
        if client is None:
            msg = "Gemini async client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)

        instruction = self._prompt.strip()
        mp4_data = window.mp4_bytes.resolve()
        if mp4_data is None:
            msg = "Window missing mp4 bytes; _validate_window must be called before _generate_caption."
            raise RuntimeError(msg)

        inline_data = genai_types.Blob(data=bytes(mp4_data), mime_type="video/mp4")
        content = genai_types.Content(
            parts=[
                genai_types.Part(inline_data=inline_data),
                genai_types.Part(text=instruction),
            ]
        )
        generate_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "contents": content,
            "config": genai_types.GenerateContentConfig(max_output_tokens=self._max_output_tokens),
        }

        async def _call() -> object:
            async for attempt in tenacity.AsyncRetrying(
                stop=tenacity.stop_after_attempt(self._max_caption_retries),
                wait=tenacity.wait_fixed(self._retry_delay_seconds),
                retry=tenacity.retry_if_exception(should_retry_gemini_exception),
                reraise=True,
            ):
                with attempt:
                    try:
                        return await client.models.generate_content(**generate_kwargs)
                    except Exception as exc:
                        new_exc = handle_gemini_client_exception(exc)
                        if new_exc is exc:
                            raise
                        raise new_exc from exc
            msg = "Gemini async retry loop exited without a result."
            raise RuntimeError(msg)

        try:
            response = await _call()
        except Exception as exc:  # noqa: BLE001
            return gemini_error_result_from_exception(exc, timeout_error_type=DeadlineExceeded)
        return normalize_gemini_response_with_detail(response)

    def _validate_window(self, window: Window) -> None:
        """Validate that the window contains data suitable for Gemini."""
        mp4_data = window.mp4_bytes.resolve()
        if mp4_data is None:
            msg = "Window missing mp4 bytes; enable keep_mp4 in the prep stage."
            raise RuntimeError(msg)
        if mp4_data.nbytes > self._max_video_size_bytes:
            size_mb = mp4_data.nbytes / (1024 * 1024)
            max_mb = self._max_video_size_bytes / (1024 * 1024)
            msg = f"Window MP4 ({size_mb:.2f} MB) exceeds Gemini inline limit ({max_mb:.2f} MB)."
            raise RuntimeError(msg)

    def _iter_window_tasks(self, task: SplitPipeTask) -> list[_WindowCaptionTask]:
        """Flatten a SplitPipeTask into caption work items."""
        window_tasks: list[_WindowCaptionTask] = []
        for clip in task.video.clips:
            window_source = clip.filter_windows if self._use_filter_windows else clip.windows
            for window_index, window in enumerate(window_source):
                window_tasks.append(_WindowCaptionTask(clip=clip, window=window, window_index=window_index))
        return window_tasks

    def _process_window_caption_result(
        self,
        window_task: _WindowCaptionTask,
        result: CaptionResult,
        error_detail: str | None,
    ) -> None:
        """Write one caption result and handle provider-specific logging."""
        clip = window_task.clip
        window = window_task.window
        window_index = window_task.window_index
        if result.outcome == CaptionOutcome.ERROR:
            clip.errors[f"{self._model_variant}_caption_{window_index}"] = error_detail or (
                f"Gemini captioning failed: {result.failure_reason}"
            )
            logger.warning(
                f"Gemini captioning failed for clip {clip.uuid} window {window_index}: "
                f"{clip.errors[f'{self._model_variant}_caption_{window_index}']}"
            )
        elif result.outcome == CaptionOutcome.BLOCKED:
            logger.warning(f"Gemini captioning blocked for clip {clip.uuid} window {window_index}")
        elif self._verbose and result.text is not None:
            logger.info(f"Gemini caption clip {clip.uuid} window {window_index}: {result.text}")
        self._write_caption_result(window, self._model_variant, result)

    async def _process_one_window_async(
        self,
        window_task: _WindowCaptionTask,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Caption one window asynchronously."""
        clip = window_task.clip
        window = window_task.window
        window_index = window_task.window_index
        try:
            self._validate_window(window)
            async with semaphore:
                result, error_detail = await self._generate_caption_with_error_detail_async(window)
        except Exception as exc:  # noqa: BLE001
            result = CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
            clip.errors[f"{self._model_variant}_caption_{window_index}"] = str(exc)
            if self._verbose:
                logger.exception(f"Gemini captioning failed for clip {clip.uuid} window {window_index}")
            else:
                logger.warning(f"Gemini captioning failed for clip {clip.uuid} window {window_index}: {exc}")
            self._write_caption_result(window, self._model_variant, result)
            return
        self._process_window_caption_result(window_task, result, error_detail)

    async def _process_task_async(
        self,
        task: SplitPipeTask,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Process one SplitPipeTask using concurrent async API requests."""
        task_timer = StageTimer(self)
        task_timer.reinit(self, task.get_major_size())
        with task_timer.time_process():
            await asyncio.gather(
                *(
                    self._process_one_window_async(window_task, semaphore)
                    for window_task in self._iter_window_tasks(task)
                )
            )
        if self._log_stats:
            stage_name, stage_perf_stats = task_timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

    async def _process_tasks_async(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Process a batch of tasks using the async request path."""
        semaphore = asyncio.Semaphore(max(1, self._batch_size))
        await asyncio.gather(*(self._process_task_async(task, semaphore) for task in tasks))
        return tasks

    @staticmethod
    def _extract_text_or_raise(response: object) -> str:
        """Extract plain text from a Gemini response or raise with diagnostics.

        ``response.text`` is the happy path. When that's empty (most often
        because thinking tokens consumed the full output budget), we walk
        ``response.candidates[i].content.parts[j].text`` to recover any
        partial output and surface ``finish_reason`` / ``block_reason`` /
        ``usage_metadata`` so callers can disambiguate "model said
        nothing" from "request was filtered" from "we ran out of tokens".
        """
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidates = getattr(response, "candidates", None) or []
        collected: list[str] = []
        finish_reasons: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None) or candidate
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    collected.append(part_text.strip())
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                finish_reasons.append(str(fr))
        result = "\n".join(collected).strip()
        if result:
            return result
        usage = getattr(response, "usage_metadata", None)
        block = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        msg = (
            f"Gemini response contained no text "
            f"(finish_reasons={finish_reasons or 'n/a'}, block_reason={block!s}, usage={usage!s})"
        )
        raise RuntimeError(msg)

    def _with_video_metadata(self, part: "genai_types.Part") -> "genai_types.Part":
        """Attach the configured ingest-fps to a built ``Part`` (no-op if unsupported)."""
        if self._video_fps is None:
            return part
        video_metadata_cls = getattr(genai_types, "VideoMetadata", None)
        if video_metadata_cls is None:
            return part
        try:
            video_metadata = video_metadata_cls(fps=self._video_fps)
        except (TypeError, ValueError):
            logger.warning(
                f"[GeminiCaptionStage] VideoMetadata(fps={self._video_fps}) not supported "
                "by installed google-genai; falling back to Gemini's default sampling fps."
            )
            return part
        kwargs: dict[str, Any] = {"video_metadata": video_metadata}
        if getattr(part, "inline_data", None) is not None:
            kwargs["inline_data"] = part.inline_data
        if getattr(part, "file_data", None) is not None:
            kwargs["file_data"] = part.file_data
        return genai_types.Part(**kwargs)

    def _build_inline_video_part(self, video_bytes: bytes) -> "genai_types.Part":
        """Build a video ``Part`` that embeds the mp4 bytes inline (≤inline limit)."""
        inline_data = genai_types.Blob(data=video_bytes, mime_type="video/mp4")
        return self._with_video_metadata(genai_types.Part(inline_data=inline_data))

    def _build_uploaded_video_part(self, video_bytes: bytes, source_label: str) -> "genai_types.Part":
        """Upload video via the Files API and reference the resulting URI.

        Used by ``caption_single`` for clips above the inline limit when
        ``enable_files_api_fallback=True``. Uploaded files are GC'd
        server-side after 48 h, so we don't delete.
        """
        client = self._client
        if client is None:
            msg = "Gemini sync client not initialised; call stage_setup first."
            raise RuntimeError(msg)
        # memfd-backed path so we don't touch the real filesystem.
        with buffer_as_memfd_path(video_bytes, name=f"gemini-caption-{source_label}") as mp4_path:
            uploaded = client.files.upload(file=str(mp4_path), config={"mime_type": "video/mp4"})
        file_uri = getattr(uploaded, "uri", None)
        if not isinstance(file_uri, str) or not file_uri:
            msg = f"Gemini Files API upload did not return a usable URI (got {uploaded!r})"
            raise RuntimeError(msg)
        file_data = genai_types.FileData(file_uri=file_uri, mime_type="video/mp4")
        return self._with_video_metadata(genai_types.Part(file_data=file_data))

    def _build_video_part(self, video_bytes: bytes, source_label: str = "single") -> "genai_types.Part":
        """Pick inline vs Files API based on ``max_video_size_bytes`` / fallback flag."""
        if len(video_bytes) <= self._max_video_size_bytes:
            return self._build_inline_video_part(video_bytes)
        if not self._enable_files_api_fallback:
            size_mb = len(video_bytes) / (1024 * 1024)
            max_mb = self._max_video_size_bytes / (1024 * 1024)
            msg = (
                f"Video ({size_mb:.2f} MB) exceeds Gemini inline limit "
                f"({max_mb:.2f} MB); enable_files_api_fallback=False."
            )
            raise RuntimeError(msg)
        logger.info(
            f"[GeminiCaptionStage] {source_label}: mp4 bytes {len(video_bytes)} exceed "
            f"inline limit {self._max_video_size_bytes}; uploading via Files API"
        )
        return self._build_uploaded_video_part(video_bytes, source_label)

    def _resolve_media_resolution(self) -> object | None:
        """Look up the configured ``MediaResolution`` enum (``None`` if not set / unsupported)."""
        if self._media_resolution is None:
            return None
        enum_cls = getattr(genai_types, "MediaResolution", None)
        if enum_cls is None:
            return None
        name_map = {
            "low": "MEDIA_RESOLUTION_LOW",
            "medium": "MEDIA_RESOLUTION_MEDIUM",
            "high": "MEDIA_RESOLUTION_HIGH",
        }
        attr_name = name_map.get(self._media_resolution)
        if attr_name is None:
            return None
        return getattr(enum_cls, attr_name, None)

    def _build_caption_single_config(self) -> "genai_types.GenerateContentConfig":
        """Build a ``GenerateContentConfig`` honouring the per-event-style knobs."""
        config_kwargs: dict[str, Any] = {"max_output_tokens": self._max_output_tokens}
        if self._response_mime_type is not None:
            config_kwargs["response_mime_type"] = self._response_mime_type
        media_resolution = self._resolve_media_resolution()
        if media_resolution is not None:
            config_kwargs["media_resolution"] = media_resolution
        if self._thinking_budget is not None:
            thinking_cfg_cls = getattr(genai_types, "ThinkingConfig", None)
            if thinking_cfg_cls is not None:
                try:
                    config_kwargs["thinking_config"] = thinking_cfg_cls(thinking_budget=self._thinking_budget)
                except (TypeError, ValueError):
                    # Older SDKs reject -1 (dynamic); fall back to disabled.
                    logger.warning(
                        f"[GeminiCaptionStage] thinking_budget={self._thinking_budget} not supported "
                        "by installed google-genai; falling back to 0 (thinking disabled)."
                    )
                    config_kwargs["thinking_config"] = thinking_cfg_cls(thinking_budget=0)
        return genai_types.GenerateContentConfig(**config_kwargs)

    def _build_caption_single_retry_decorator(self) -> Any:  # noqa: ANN401  # tenacity.retry has no public Protocol
        """Build a ``tenacity.retry`` decorator matching the configured policy."""

        def _log_retry(retry_state: "tenacity.RetryCallState") -> None:
            outcome = retry_state.outcome
            exc = outcome.exception() if outcome is not None else None
            next_wait = getattr(retry_state.next_action, "sleep", "?") if retry_state.next_action else "?"
            logger.warning(
                f"[GeminiCaptionStage] caption_single attempt "
                f"{retry_state.attempt_number}/{self._max_caption_retries} failed "
                f"({type(exc).__name__ if exc else 'unknown'}: {exc}); "
                f"sleeping {next_wait}s before retry"
            )

        if self._retry_policy is GeminiRetryPolicy.EXPONENTIAL_JITTER:
            wait_strategy: Any = tenacity.wait_exponential(
                multiplier=self._retry_delay_seconds,
                max=self._retry_max_delay_seconds,
            ) + tenacity.wait_random(0, self._retry_jitter_seconds)
        else:
            wait_strategy = tenacity.wait_fixed(self._retry_delay_seconds)
        return tenacity.retry(
            stop=tenacity.stop_after_attempt(self._max_caption_retries),
            wait=wait_strategy,
            retry=tenacity.retry_if_exception(_is_transient_gemini_exception),
            reraise=True,
            before_sleep=_log_retry,
        )

    def caption_single(self, prompt: str, video_bytes: bytes) -> str:
        """Implement :class:`SingleInferenceCaptionStage` for one-shot consumers.

        Builds a single-content request from ``(prompt, video_bytes)``,
        honouring the per-call knobs (``response_mime_type``,
        ``media_resolution``, ``thinking_budget``, ``video_fps``,
        ``enable_files_api_fallback``). Retries per
        ``retry_policy``. Returns the response text or raises
        ``RuntimeError`` on empty / blocked / unrecoverable responses.

        Side effect: stashes ``finish_reasons`` / ``usage_metadata`` from
        the last attempt onto ``self._last_*`` for downstream
        diagnostics.
        """
        client = self._client
        if client is None:
            msg = "Gemini sync client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)

        # Reset per-call diagnostics so the public accessors never report
        # stale data from a previous clip on failures / empty / blocked.
        self._last_finish_reasons = []
        self._last_usage_metadata = None

        config = self._build_caption_single_config()

        def _call() -> str:
            # Build the video Part inside the retry-wrapped closure so
            # ``_build_uploaded_video_part`` (Files API upload, ~20 MB+
            # transfer) participates in the same transient-error retry
            # policy as ``generate_content``. The inline (≤20 MB) path
            # short-circuits cheaply; the upload path tolerates 503/429
            # spikes without bypassing the retry decorator.
            content = genai_types.Content(
                parts=[
                    self._build_video_part(video_bytes),
                    genai_types.Part(text=prompt),
                ],
            )
            response = client.models.generate_content(
                model=self._model_name,
                contents=content,
                config=config,
            )
            text = self._extract_text_or_raise(response)
            self._last_finish_reasons = [
                str(getattr(c, "finish_reason", "")) for c in (getattr(response, "candidates", None) or [])
            ]
            self._last_usage_metadata = getattr(response, "usage_metadata", None)
            return text

        retry_decorator = self._build_caption_single_retry_decorator()
        wrapped = cast("Callable[[], str]", retry_decorator(_call))
        return wrapped()

    @property
    def last_finish_reasons(self) -> list[str]:
        """Return the ``finish_reason`` from the last ``caption_single`` attempt."""
        return list(self._last_finish_reasons)

    @property
    def last_usage_metadata(self) -> object | None:
        """Return the ``usage_metadata`` from the last ``caption_single`` attempt."""
        return self._last_usage_metadata

    def destroy(self) -> None:
        """Close the async runner and any provider clients."""
        destroy_api_clients(async_client=self._async_client, runner=self._runner, sync_client=self._client)
        self._async_client = None
        self._runner = None
        self._client = None

    @nvtx.annotate("GeminiCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Caption each window in the provided tasks using Gemini."""
        if self._runner is None:
            msg = "Gemini async runner not initialized; call stage_setup before processing data."
            raise RuntimeError(msg)
        return self._runner.run(self._process_tasks_async(tasks))
