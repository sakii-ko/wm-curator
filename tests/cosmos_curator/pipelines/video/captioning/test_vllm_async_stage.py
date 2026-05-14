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

"""Tests for vllm_async_stage."""

import asyncio
import gc
import uuid as uuid_lib
import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncPrepConfig
from cosmos_curator.pipelines.video.captioning.vllm_async_stage import (
    VllmAsyncCaptionStage,
    _resolve_mode,
    _VllmAsyncModel,
    _VllmAsyncStageMode,
)
from cosmos_curator.pipelines.video.utils.data_model import (
    CaptionOutcome,
    CaptionResult,
    Clip,
    TokenCounts,
    VllmAsyncConfig,
    VllmCaptionRequest,
    VllmConfig,
    Window,
)


class TestVllmAsyncModel:
    """Lightweight ModelInterface used to register weights for download."""

    def test_model_id_names_resolves_known_variant(self) -> None:
        """Known variant 'qwen' resolves to the Qwen HuggingFace model ID."""
        model = _VllmAsyncModel("qwen")
        assert model.model_id_names == ["Qwen/Qwen2.5-VL-7B-Instruct"]

    def test_unknown_variant_raises(self) -> None:
        """Unregistered variant raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            _VllmAsyncModel("custom-org/my-model")

    def test_conda_env_name_is_unified(self) -> None:
        """conda_env_name returns 'unified' (where vLLM is installed)."""
        assert _VllmAsyncModel("qwen").conda_env_name == "unified"

    def test_setup_is_noop(self) -> None:
        """setup() succeeds without side effects (engine loads weights)."""
        _VllmAsyncModel("qwen").setup()


class TestVllmAsyncConfigTypedDefaults:
    """Validate ``VllmAsyncConfig`` field defaults, types, and per-field validators."""

    def test_default_field_values(self) -> None:
        """Auto-sentinel defaults (0/'') for int/str fields; typed defaults for enum-string fields."""
        cfg = VllmAsyncConfig(model_variant="qwen")
        assert cfg.max_num_seqs == 0
        assert cfg.long_prefill_token_threshold == 0
        assert cfg.mm_processor_cache_type == ""
        assert cfg.mm_encoder_tp_mode == "data"
        assert cfg.extra_env_vars == ""
        assert cfg.distributed_executor_backend == "ray"
        assert cfg.kv_cache_dtype == "auto"
        assert cfg.fp8 is False
        assert cfg.disable_mmcache is False

    def test_num_gpus_below_one_rejected(self) -> None:
        """``num_gpus`` must be ``>= 1``; integer ``0`` is rejected.

        ``num_gpus`` is now ``int``-only (mirrors sync's ``VllmConfig``);
        the ``attrs.validators.ge(1)`` enforces vLLM's tensor-parallel
        size domain at config construction time so plugin layers can no
        longer silently truncate fractional inputs to ``1``.
        """
        with pytest.raises(ValueError, match=r"num_gpus"):
            VllmAsyncConfig(model_variant="qwen", num_gpus=0)

    def test_num_gpus_float_rejected(self) -> None:
        """Float ``num_gpus`` is rejected -- vLLM's ``tensor_parallel_size`` is integer-only."""
        with pytest.raises(TypeError):
            VllmAsyncConfig(model_variant="qwen", num_gpus=1.5)  # type: ignore[arg-type]

    def test_data_parallel_size_zero_rejected(self) -> None:
        """``data_parallel_size=0`` is rejected by ``attrs.validators.ge(1)``."""
        with pytest.raises(ValueError, match=r"data_parallel_size"):
            VllmAsyncConfig(model_variant="qwen", data_parallel_size=0)

    def test_async_scheduling_with_mp_executor_allowed(self) -> None:
        """``async_scheduling=True`` is allowed with the ``mp`` executor."""
        cfg = VllmAsyncConfig(
            model_variant="qwen",
            async_scheduling=True,
            distributed_executor_backend="mp",
        )
        assert cfg.async_scheduling is True
        assert cfg.distributed_executor_backend == "mp"

    def test_total_gpus_arithmetic(self) -> None:
        """``total_gpus == num_gpus * data_parallel_size`` (both int -> int product)."""
        cfg = VllmAsyncConfig(model_variant="qwen", num_gpus=2, data_parallel_size=4)
        assert cfg.total_gpus == 8
        assert isinstance(cfg.total_gpus, int)


class TestVllmAsyncConfigExtraEnvVarsValidation:
    """``extra_env_vars`` is a JSON-encoded ``dict[str, str]`` string."""

    def test_default_is_empty_string(self) -> None:
        """Empty default means "no extra env vars" -- validator skips parsing."""
        cfg = VllmAsyncConfig(model_variant="qwen")
        assert cfg.extra_env_vars == ""

    def test_accepts_valid_json_object(self) -> None:
        """A well-formed JSON dict[str, str] is preserved verbatim."""
        cfg = VllmAsyncConfig(
            model_variant="qwen",
            extra_env_vars='{"NCCL_DEBUG": "TRACE"}',
        )
        assert cfg.extra_env_vars == '{"NCCL_DEBUG": "TRACE"}'

    def test_rejects_malformed_json(self) -> None:
        """Non-JSON input is rejected with a clear error citing the field."""
        with pytest.raises(ValueError, match="must be valid JSON"):
            VllmAsyncConfig(model_variant="qwen", extra_env_vars="not json")

    def test_rejects_non_object_json(self) -> None:
        """JSON that parses but is not an object (e.g. an array) is rejected."""
        with pytest.raises(TypeError, match="JSON object"):
            VllmAsyncConfig(model_variant="qwen", extra_env_vars="[1, 2, 3]")

    def test_rejects_non_string_values(self) -> None:
        """Non-string values inside the dict are rejected (env vars must be strings)."""
        with pytest.raises(TypeError, match="values must be strings"):
            VllmAsyncConfig(model_variant="qwen", extra_env_vars='{"K": 1}')


class TestVllmAsyncConfigToVllmConfig:
    """Verifies the adapter to ``VllmConfig`` for plugin reuse."""

    def test_minimal_translation(self) -> None:
        """Adapter forwards model_variant + user knobs; pins async-only invariants."""
        async_cfg = VllmAsyncConfig(model_variant="qwen", fp8=True, disable_mmcache=True)
        sync_cfg = async_cfg.to_vllm_config()

        assert isinstance(sync_cfg, VllmConfig)
        assert sync_cfg.model_variant == "qwen"
        assert sync_cfg.use_image_input is False  # async is video-only
        assert sync_cfg.copy_weights_to is None  # async never uses sync's SSD-copy fast path
        assert sync_cfg.fp8 is True
        assert sync_cfg.disable_mmcache is True

    def test_defaults_propagate(self) -> None:
        """Default fp8/disable_mmcache forward to the sync VllmConfig."""
        sync_cfg = VllmAsyncConfig(model_variant="nemotron").to_vllm_config()
        assert sync_cfg.fp8 is False
        assert sync_cfg.disable_mmcache is False


class TestVllmAsyncConfigTotalGpus:
    """``total_gpus`` derives from num_gpus * data_parallel_size."""

    def test_single_gpu_no_dp(self) -> None:
        """Single GPU, no DP => total_gpus == num_gpus."""
        assert VllmAsyncConfig(model_variant="qwen").total_gpus == 1

    def test_dp_multiplies_num_gpus(self) -> None:
        """data_parallel_size > 1 multiplies num_gpus."""
        cfg = VllmAsyncConfig(model_variant="qwen", num_gpus=2, data_parallel_size=4)
        assert cfg.total_gpus == 8


class TestVllmAsyncPrepConfig:
    """Prep stage windowing / decoding configuration."""

    def test_defaults(self) -> None:
        """Default window_size, remainder_threshold, sample_fps, keep_mp4."""
        cfg = VllmAsyncPrepConfig(model_variant="qwen")
        assert cfg.window_size == 256
        assert cfg.remainder_threshold == 128
        assert cfg.sample_fps == pytest.approx(2.0)
        assert cfg.keep_mp4 is False


class TestResolveMode:
    """Mode-dependent parameter resolution (DP vs N-actors)."""

    def test_n_actors_single_gpu_uses_mp_backend(self) -> None:
        """N-actors mode with num_gpus=1 auto-selects 'mp' backend (enables async_scheduling)."""
        cfg = VllmAsyncConfig(model_variant="qwen", num_gpus=1, data_parallel_size=1)
        mode = _resolve_mode(cfg)
        assert mode.is_dp_mode is False
        assert mode.executor_backend == "mp"
        assert mode.gpus_per_actor == 1
        assert mode.stage_batch_size == 1

    def test_n_actors_multi_gpu_uses_configured_backend(self) -> None:
        """N-actors mode with num_gpus>1 uses the configured backend (default 'ray')."""
        cfg = VllmAsyncConfig(
            model_variant="qwen",
            num_gpus=2,
            data_parallel_size=1,
            distributed_executor_backend="ray",
        )
        mode = _resolve_mode(cfg)
        assert mode.is_dp_mode is False
        assert mode.executor_backend == "ray"

    def test_dp_mode_uses_total_gpus_and_dp_batch(self) -> None:
        """DP mode (data_parallel_size > 1) uses total_gpus and DP-batch sizing."""
        cfg = VllmAsyncConfig(model_variant="qwen", num_gpus=2, data_parallel_size=4)
        mode = _resolve_mode(cfg)
        assert mode.is_dp_mode is True
        assert mode.gpus_per_actor == 8
        # DP batch = max(DP_BATCH_MULTIPLIER * total, DP_BATCH_FLOOR)
        expected_batch = max(_VllmAsyncStageMode.DP_BATCH_MULTIPLIER * 8, _VllmAsyncStageMode.DP_BATCH_FLOOR)
        assert mode.stage_batch_size == expected_batch


def _make_stage(*, keep_mp4: bool) -> VllmAsyncCaptionStage:
    """Construct ``VllmAsyncCaptionStage`` with a minimal serve config."""
    cfg = VllmAsyncConfig(model_variant="qwen")
    return VllmAsyncCaptionStage(serve_config=cfg, model_name="test", keep_mp4=keep_mp4)


def _stub_normalize_vllm_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``_normalize_vllm_result`` so ``_scatter_one`` runs outside ``unified`` env.

    The real symbol is imported only when the ``unified`` pixi env loads
    vLLM; mocking it lets the focused ``keep_mp4`` test run in the default
    CPU env without dragging in heavy dependencies.
    """
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.captioning.vllm_async_stage._normalize_vllm_result",
        lambda result: CaptionResult(outcome=CaptionOutcome.SUCCESS, text=result.text),
        raising=False,
    )


def _make_window_with_mp4(payload: bytes = b"mp4-bytes") -> Window:
    """Build a ``Window`` populated with mp4 bytes via ``LazyData.coerce``."""
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    return Window(start_frame=0, end_frame=15, mp4_bytes=arr)  # type: ignore[arg-type]


def _fake_result(text: str = "caption") -> Any:  # noqa: ANN401  # SimpleNamespace mirrors VllmWindowResult shape
    """Return a ``SimpleNamespace`` shaped like ``VllmWindowResult``."""
    return SimpleNamespace(text=text, finish_reason="stop", token_counts=TokenCounts())


class TestVllmAsyncCaptionStageKeepMp4:
    """``_scatter_one`` honours ``keep_mp4`` -- mirrors sync ``VllmCaptionStage``."""

    def test_scatter_keeps_mp4_when_flag_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``keep_mp4=True`` preserves ``window.mp4_bytes`` for downstream stages."""
        _stub_normalize_vllm_result(monkeypatch)
        stage = _make_stage(keep_mp4=True)
        window = _make_window_with_mp4()

        stage._scatter_one(0, _fake_result(), [window], ["clip-uuid"])

        assert bool(window.mp4_bytes), "mp4_bytes must remain available when keep_mp4=True"

    def test_scatter_drops_mp4_when_flag_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``keep_mp4=False`` drops ``window.mp4_bytes`` after captioning."""
        _stub_normalize_vllm_result(monkeypatch)
        stage = _make_stage(keep_mp4=False)
        window = _make_window_with_mp4()

        stage._scatter_one(0, _fake_result(), [window], ["clip-uuid"])

        assert not bool(window.mp4_bytes), "mp4_bytes must be dropped when keep_mp4=False"

    def test_constructor_default_is_false(self) -> None:
        """Default ``keep_mp4`` is ``False`` -- matches sync default and historical behaviour."""
        cfg = VllmAsyncConfig(model_variant="qwen")
        stage = VllmAsyncCaptionStage(serve_config=cfg, model_name="test")
        assert stage._keep_mp4 is False


def _make_clip() -> Clip:
    """Build a minimal ``Clip`` -- only ``uuid`` / ``source_video`` / ``span`` are required."""
    return Clip(uuid=uuid_lib.uuid4(), source_video="/tmp/x.mp4", span=(0.0, 1.0))  # noqa: S108


class TestRecordWindowError:
    """``_record_window_error`` populates ``clip.errors`` for per-window caption failures.

    Restores the cross-stage ``clip.errors`` registry entry the refactor
    dropped (legacy ``_generate_and_assign`` / ``_stage2_refine_and_assign``
    wrote it on every failure).  Downstream consumers that enumerate
    ``clip.errors`` -- notably ``metadata_writer_stage`` which serializes
    the dict to the output JSON ``errors`` array -- otherwise silently
    see zero entries even when every window failed.
    """

    def test_stage1_failure_writes_clip_errors_with_caption_key(self) -> None:
        """``stage1`` failure -> ``clip.errors[f"{caption_key}_caption_{window_index}"] = str(exc)``."""
        stage = _make_stage(keep_mp4=False)
        clip = _make_clip()
        exc = RuntimeError("boom")

        stage._record_window_error(idx=0, phase="stage1", exc=exc, clips=[clip], window_indices=[3])

        assert clip.errors == {f"{stage._caption_key}_caption_3": str(exc)}

    @pytest.mark.parametrize("late_phase", ["stage2", "stage2_build"])
    def test_late_phase_failure_does_not_overwrite_stage1_entry(self, late_phase: str) -> None:
        """Non-``stage1`` phases get a distinct ``_<phase>`` suffix so both phases coexist."""
        stage = _make_stage(keep_mp4=False)
        clip = _make_clip()
        exc1, exc2 = RuntimeError("stage1 fail"), RuntimeError(f"{late_phase} fail")

        stage._record_window_error(0, "stage1", exc1, [clip], [0])
        stage._record_window_error(0, late_phase, exc2, [clip], [0])

        base = f"{stage._caption_key}_caption_0"
        assert clip.errors == {base: str(exc1), f"{base}_{late_phase}": str(exc2)}


def _make_continuous_stage() -> VllmAsyncCaptionStage:
    """Build a ``VllmAsyncCaptionStage`` and stub out collaborators that need vLLM."""
    cfg = VllmAsyncConfig(model_variant="qwen")
    return VllmAsyncCaptionStage(serve_config=cfg, model_name="test", max_concurrent_requests=4)


def _continuous_input(task_id: str) -> Any:  # noqa: ANN401  # mirrors ContinuousTaskInput shape
    """Construct a fake ``ContinuousTaskInput`` -- only ``task_id`` is read by the loop."""
    from cosmos_xenna.ray_utils.continuous_stage import ContinuousTaskInput  # noqa: PLC0415
    from cosmos_xenna.ray_utils.task_metadata import TimingInfo  # noqa: PLC0415

    return ContinuousTaskInput(task_id=task_id, data=[], timing=TimingInfo(), object_sizes=[])


class TestRunContinuousRefillCadence:
    """``run_continuous`` keeps the wait set rebuilt across iterations."""

    @pytest.mark.asyncio
    async def test_stop_event_short_circuits_with_empty_queue(self) -> None:
        """``stop_event`` already set on entry -> loop exits immediately, no get_task left running.

        The post-fix loop creates ``get_task`` on each iteration; if the
        initial ``stop_event`` check passes (``is_set()=True``) the
        function must NOT spawn a get_task -- otherwise we leak an
        ``asyncio.Task`` waiting on an empty input queue.
        """
        stage = _make_continuous_stage()
        stage._gpu_trace_attributes = {}

        input_q: asyncio.Queue[Any] = asyncio.Queue()
        output_q: asyncio.Queue[Any] = asyncio.Queue()
        stop = asyncio.Event()
        stop.set()

        await stage.run_continuous(input_q, output_q, stop)

        assert output_q.empty()

    @pytest.mark.asyncio
    async def test_stop_event_cancels_pending_get_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When stop_event fires while ``get_task`` is parked, the loop cancels it cleanly.

        Without the ``finally`` block + ``contextlib.suppress`` wrapper the
        pending ``input_queue.get()`` task would survive past loop exit and
        emit "Task was destroyed but it is pending" warnings at GC.
        """
        stage = _make_continuous_stage()
        stage._gpu_trace_attributes = {}

        input_q: asyncio.Queue[Any] = asyncio.Queue()
        output_q: asyncio.Queue[Any] = asyncio.Queue()
        stop = asyncio.Event()

        # Shorten the per-loop timeout so the test does not block on the
        # default 1s timeout used by the steady-state guard.
        monkeypatch.setattr(
            "cosmos_curator.pipelines.video.captioning.vllm_async_stage._INPUT_GET_TIMEOUT_S",
            0.05,
        )

        async def _fire_stop() -> None:
            await asyncio.sleep(0.05)
            stop.set()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(stage.run_continuous(input_q, output_q, stop))
            tg.create_task(_fire_stop())

        # Loop returned cleanly; output queue stays empty (no input was ever delivered).
        assert output_q.empty()

    @pytest.mark.asyncio
    async def test_register_task_called_when_input_arrives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Input arrival -> ``_register_task`` invoked with the resolved ``ContinuousTaskInput``.

        Verifies the Option-B refill semantics: the loop wakes on a fresh
        ``get_task`` result and threads the value through ``_register_task``
        before re-entering ``asyncio.wait``.  The stubbed ``_register_task``
        records every invocation so we can assert order + count without
        needing a live engine or vLLM dependency.
        """
        stage = _make_continuous_stage()
        stage._gpu_trace_attributes = {}

        registered: list[Any] = []

        def _record_register(
            trackers: dict[str, Any],
            task_input: Any,  # noqa: ANN401  -- mirrors helper signature
            semaphore: asyncio.Semaphore,
            output_queue: asyncio.Queue[Any],
        ) -> None:
            del trackers, semaphore, output_queue
            registered.append(task_input)

        monkeypatch.setattr(stage, "_register_task", _record_register)
        monkeypatch.setattr(
            "cosmos_curator.pipelines.video.captioning.vllm_async_stage._INPUT_GET_TIMEOUT_S",
            0.05,
        )

        input_q: asyncio.Queue[Any] = asyncio.Queue()
        output_q: asyncio.Queue[Any] = asyncio.Queue()
        stop = asyncio.Event()

        first = _continuous_input("task-1")
        await input_q.put(first)

        async def _stop_after_register() -> None:
            # Wait until ``_register_task`` records the first input,
            # then fire stop_event to end the loop cleanly.
            for _ in range(50):
                if registered:
                    break
                await asyncio.sleep(0.01)
            stop.set()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(stage.run_continuous(input_q, output_q, stop))
            tg.create_task(_stop_after_register())

        assert len(registered) == 1
        assert registered[0] is first

    @pytest.mark.asyncio
    async def test_burst_drain_absorbs_queued_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A burst of queued inputs is drained in a single loop tick via NOWAIT batching.

        Regression guard for ``_drain_input_queue``: once the first ``get``
        wakes the loop, subsequent already-queued inputs must be absorbed
        without waiting for additional loop iterations -- this is what
        keeps the semaphore saturated under high prep-stage throughput.
        """
        stage = _make_continuous_stage()
        stage._gpu_trace_attributes = {}

        registered: list[Any] = []

        def _record_register(
            trackers: dict[str, Any],
            task_input: Any,  # noqa: ANN401
            semaphore: asyncio.Semaphore,
            output_queue: asyncio.Queue[Any],
        ) -> None:
            del trackers, semaphore, output_queue
            registered.append(task_input)

        monkeypatch.setattr(stage, "_register_task", _record_register)
        monkeypatch.setattr(
            "cosmos_curator.pipelines.video.captioning.vllm_async_stage._INPUT_GET_TIMEOUT_S",
            0.05,
        )

        input_q: asyncio.Queue[Any] = asyncio.Queue()
        output_q: asyncio.Queue[Any] = asyncio.Queue()
        stop = asyncio.Event()

        bursts = [_continuous_input(f"task-{i}") for i in range(3)]
        for inp in bursts:
            await input_q.put(inp)

        async def _stop_after_drain() -> None:
            for _ in range(50):
                if len(registered) >= len(bursts):
                    break
                await asyncio.sleep(0.01)
            stop.set()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(stage.run_continuous(input_q, output_q, stop))
            tg.create_task(_stop_after_drain())

        assert [inp.task_id for inp in registered] == [inp.task_id for inp in bursts]


class _WeakrefDict(dict[str, Any]):
    """``dict`` subclass that supports ``weakref.ref`` -- builtin ``dict`` does not."""


def _make_payload(contents: dict[str, Any]) -> _WeakrefDict:
    """Build a weakref-able payload dict for alias-lifetime tests."""
    out = _WeakrefDict()
    out.update(contents)
    return out


def _make_window_with_payload(payload: dict[str, Any], input_key: str = "qwen") -> Window:
    """Build a ``Window`` whose ``model_input[input_key]`` is the given payload."""
    arr = np.zeros(4, dtype=np.uint8)
    return Window(start_frame=0, end_frame=15, mp4_bytes=arr, model_input={input_key: payload})  # type: ignore[arg-type]


class TestIterRequests:
    """``_iter_requests`` yields one ``VllmCaptionRequest`` per window with no list aliasing.

    These tests pin the no-intermediate-list contract: the iterator
    is the ONLY runtime path the stage uses to feed payload dicts to
    ``vllm_caption_async``, so the only stage-side aliases of each
    payload dict are ``window.model_input[input_key]`` and the in-
    flight ``request.inputs`` owned by the captioner.  The captioner
    is single-owner of the ``request.inputs`` alias and clears it at
    the per-window completion callback; ``window.model_input`` is
    cleared separately by ``_scatter_one``.  Aliases through any
    intermediate list (e.g. a pre-built ``model_inputs`` argument)
    are forbidden -- the alias-release behaviour is exercised
    end-to-end by ``TestPayloadAliasRelease``.
    """

    def test_yields_one_per_window_in_order(self) -> None:
        """N windows in -> N requests out, preserving input order."""
        stage = _make_stage(keep_mp4=False)
        windows = [
            _make_window_with_payload({"prompt": "a"}, stage._input_key),
            _make_window_with_payload({"prompt": "b"}, stage._input_key),
            _make_window_with_payload({"prompt": "c"}, stage._input_key),
        ]

        requests = list(stage._iter_requests(windows, ["u-a", "u-b", "u-c"], [None, None, None]))

        assert len(requests) == 3
        assert [r.inputs["prompt"] for r in requests] == ["a", "b", "c"]

    def test_request_inputs_is_same_object_as_window_payload(self) -> None:
        """``request.inputs`` is the SAME dict object as ``window.model_input[input_key]``."""
        stage = _make_stage(keep_mp4=False)
        payload: dict[str, Any] = {"prompt": "p"}
        window = _make_window_with_payload(payload, stage._input_key)

        request = next(stage._iter_requests([window], ["u-1"], [None]))

        assert request.inputs is payload

    def test_stage2_prompt_propagates_per_window(self) -> None:
        """Each request inherits its window's stage-2 prompt (``None`` skips refinement)."""
        stage = _make_stage(keep_mp4=False)
        windows = [
            _make_window_with_payload({"prompt": "a"}, stage._input_key),
            _make_window_with_payload({"prompt": "b"}, stage._input_key),
        ]
        stage2_prompts = ["refine-a", None]

        requests = list(stage._iter_requests(windows, ["u-a", "u-b"], stage2_prompts))

        assert requests[0].stage2_prompt == "refine-a"
        assert requests[1].stage2_prompt is None

    def test_mismatched_lengths_raise(self) -> None:
        """``zip(strict=True)`` ensures windows, clip_uuids and stage-2 prompts stay in lockstep."""
        stage = _make_stage(keep_mp4=False)
        windows = [_make_window_with_payload({"prompt": "a"}, stage._input_key)]
        too_many_prompts = [None, None]

        with pytest.raises(ValueError, match="zip"):
            list(stage._iter_requests(windows, ["u-1"], too_many_prompts))


class TestIterRequestsEmptyInputsSentinel:
    """``_iter_requests`` yields an empty-``inputs`` sentinel for windows missing their input key.

    Missing-input contract: a window without ``self._input_key`` in
    ``model_input`` MUST yield a request with ``inputs == {}`` rather
    than raise ``KeyError`` (which would crash the whole pipe task).
    The stage logs a structured warning at the point of detection so
    the captioner stays agnostic to which key was missing; positional
    alignment with neighbouring slots is preserved so the captioner
    scatters results into the correct windows.
    """

    def test_missing_input_yields_empty_inputs_sentinel(self) -> None:
        """Window without input key -> request with ``inputs == {}``."""
        stage = _make_stage(keep_mp4=False)
        window = _make_window_with_payload({}, input_key="other-key")
        window.model_input.pop("other-key", None)  # leave model_input empty
        assert stage._input_key not in window.model_input

        requests = list(stage._iter_requests([window], ["uuid-1"], [None]))

        assert len(requests) == 1
        assert requests[0].inputs == {}

    def test_present_input_yields_normal_request(self) -> None:
        """Window with input key -> request with non-empty ``inputs``."""
        stage = _make_stage(keep_mp4=False)
        window = _make_window_with_payload({"prompt": "p"}, stage._input_key)

        requests = list(stage._iter_requests([window], ["uuid-1"], [None]))

        assert requests[0].inputs == {"prompt": "p"}

    def test_mixed_windows_preserve_positional_alignment(self) -> None:
        """Mix of present + missing inputs keeps requests aligned with ``windows`` order."""
        stage = _make_stage(keep_mp4=False)
        ok_a = _make_window_with_payload({"prompt": "a"}, stage._input_key)
        missing = Window(start_frame=10, end_frame=20)  # no model_input at all
        ok_c = _make_window_with_payload({"prompt": "c"}, stage._input_key)
        stage2_prompts = ["s2-a", "s2-missing", None]

        requests = list(
            stage._iter_requests(
                [ok_a, missing, ok_c],
                ["uuid-a", "uuid-missing", "uuid-c"],
                stage2_prompts,
            )
        )

        assert len(requests) == 3
        assert requests[0].inputs == {"prompt": "a"}
        assert requests[0].stage2_prompt == "s2-a"
        assert requests[1].inputs == {}  # sentinel: empty dict
        assert requests[1].stage2_prompt == "s2-missing"
        assert requests[2].inputs == {"prompt": "c"}
        assert requests[2].stage2_prompt is None


@pytest.mark.env("unified")
class TestAsyncCaptionerEmptyInputsShortCircuits:
    """``_AsyncCaptioner.run`` short-circuits empty-``inputs`` sentinels.

    Sentinel contract: when ``request.inputs`` is falsy (empty dict),
    the captioner skips ``engine.generate``, emits
    ``VLLM_UNKNOWN_CAPTION`` for that index, and does NOT fire
    ``on_window_error`` -- the request builder that constructed the
    sentinel is expected to have already logged the underlying cause
    at the point of detection.
    """

    @pytest.mark.asyncio
    async def test_empty_inputs_emits_unknown_without_engine_call(self) -> None:
        """Sentinel request -> ``_emit_unknown`` only; ``engine.generate`` NOT called."""
        from cosmos_curator.models.vllm_interface import _AsyncCaptioner  # noqa: PLC0415
        from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION  # noqa: PLC0415

        engine_calls: list[str] = []

        class _FailIfCalledEngine:
            async def generate(self, **kwargs: Any) -> Any:  # noqa: ANN401  -- duck-typed engine fake
                engine_calls.append(kwargs["request_id"])
                msg = "engine.generate must not be called for sentinel slots"
                raise AssertionError(msg)
                yield

        errors: list[tuple[int, str, Exception]] = []

        captioner = _AsyncCaptioner(
            engine=_FailIfCalledEngine(),
            processor=SimpleNamespace(),
            plugin=SimpleNamespace(),
            semaphore=asyncio.Semaphore(1),
            sampling_params=SimpleNamespace(),
            max_retries=1,
            request_id_factory=lambda: "rid",
            on_window_error=lambda idx, phase, exc: errors.append((idx, phase, exc)),
        )

        sentinel = VllmCaptionRequest(request_id="r-empty", inputs={})

        results = await captioner.run([sentinel])

        assert engine_calls == []
        assert len(results) == 1
        assert results[0].text == VLLM_UNKNOWN_CAPTION
        # Stage already logged at iteration time -- captioner does NOT
        # fire on_window_error for the empty-inputs sentinel branch.
        assert errors == []


class TestGatherInputsContract:
    """``_gather_inputs`` returns ``(windows, clip_uuids)`` only -- no model_inputs list.

    A third ``model_inputs`` list would share identity with each
    ``window.model_input[input_key]`` and pin every payload dict for
    the lifetime of ``_caption_pipe_tasks``, violating the single-
    owner alias contract.  This test enforces the 2-tuple shape so the
    payload dicts have exactly one stage-side alias
    (``window.model_input``) until the captioner takes over.

    ``_get_windows_from_tasks`` is only imported in the ``unified``
    pixi env (it lives behind a conda gate in ``vllm_async_stage``).
    Mocking it lets this contract test run in the default CPU env
    without dragging in vLLM dependencies.
    """

    def test_returns_two_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_gather_inputs`` returns exactly two lists, matching the new contract shape."""
        sentinel_windows: list[Window] = []
        sentinel_uuids: list[str] = []
        monkeypatch.setattr(
            "cosmos_curator.pipelines.video.captioning.vllm_async_stage._get_windows_from_tasks",
            lambda _tasks: (sentinel_windows, sentinel_uuids),
            raising=False,
        )
        stage = _make_stage(keep_mp4=False)

        result = stage._gather_inputs([])

        assert isinstance(result, tuple)
        assert len(result) == 2
        windows, clip_uuids = result
        assert windows is sentinel_windows
        assert clip_uuids is sentinel_uuids


class TestPayloadAliasRelease:
    """Weakref-based release tests for the per-window payload dict.

    These exercise the alias kill contract at the request level
    without booting an ``AsyncLLM`` engine: the stage's
    ``_iter_requests`` produces requests whose ``inputs`` shares
    identity with the window's payload dict; once the request's
    ``inputs`` is cleared (mirroring what ``_AsyncCaptioner._handle_completed``
    does in production) and the window's outer key is popped
    (mirroring ``_scatter_one``), every alias is gone and the payload
    dict becomes weakref-releasable.  ``gc.collect()`` is called so a
    failure pins on a missed alias rather than GC timing.
    """

    def test_payload_survives_while_request_holds_inputs(self) -> None:
        """Both ``window.model_input`` AND ``request.inputs`` keep payload alive."""
        stage = _make_stage(keep_mp4=False)
        payload = _make_payload({"prompt": "p"})
        window = _make_window_with_payload(payload, stage._input_key)
        request = next(stage._iter_requests([window], ["u-1"], [None]))

        ref = weakref.ref(payload)
        del payload
        gc.collect()

        assert ref() is not None, "payload must survive while request.inputs holds it"
        assert request.inputs is ref()

    def test_payload_released_when_all_aliases_cleared(self) -> None:
        """Clearing window key + ``request.inputs`` releases the payload dict."""
        stage = _make_stage(keep_mp4=False)
        payload = _make_payload({"prompt": "p"})
        window = _make_window_with_payload(payload, stage._input_key)
        request = next(stage._iter_requests([window], ["u-1"], [None]))
        ref = weakref.ref(payload)
        del payload

        window.model_input.pop(stage._input_key, None)
        request.inputs = None  # type: ignore[assignment]  # mirrors _AsyncCaptioner._handle_completed
        del request
        gc.collect()

        assert ref() is None, "payload must be released once both aliases are cleared"

    def test_window_alone_does_not_release_payload(self) -> None:
        """Just popping the window key does NOT release; ``request.inputs`` still alive."""
        stage = _make_stage(keep_mp4=False)
        payload = _make_payload({"prompt": "p"})
        window = _make_window_with_payload(payload, stage._input_key)
        request = next(stage._iter_requests([window], ["u-1"], [None]))
        ref = weakref.ref(payload)
        del payload

        window.model_input.pop(stage._input_key, None)
        gc.collect()

        assert ref() is not None, "popping window alone must not release while request.inputs holds it"
        # Cleanup to keep the test isolated.
        request.inputs = None  # type: ignore[assignment]

    def test_request_inputs_alone_does_not_release_payload(self) -> None:
        """Just clearing ``request.inputs`` does NOT release; ``window.model_input`` still alive."""
        stage = _make_stage(keep_mp4=False)
        payload = _make_payload({"prompt": "p"})
        window = _make_window_with_payload(payload, stage._input_key)
        request = next(stage._iter_requests([window], ["u-1"], [None]))
        ref = weakref.ref(payload)
        del payload

        request.inputs = None  # type: ignore[assignment]
        del request
        gc.collect()

        assert ref() is not None, "clearing request alone must not release while window.model_input holds it"


@pytest.mark.env("unified")
class TestAsyncCaptionerHandleCompletedReleasesInputs:
    """``_AsyncCaptioner._handle_completed`` clears ``req.inputs`` at every terminal path.

    Verifies the production ``_handle_completed`` method releases the
    payload-dict alias on every exit branch (success, stage-2 refine,
    retry exhausted, fatal error).  Requires the ``unified`` env
    because ``_AsyncCaptioner`` lives in ``vllm_interface.py``, which
    imports vLLM at module level; local imports inside the test
    helpers keep this file CPU-importable for the rest of the tests.
    """

    @staticmethod
    def _build_captioner() -> Any:  # noqa: ANN401  -- returns _AsyncCaptioner, imported lazily
        from cosmos_curator.models.vllm_interface import _AsyncCaptioner  # noqa: PLC0415

        return _AsyncCaptioner(
            engine=SimpleNamespace(),
            processor=SimpleNamespace(),
            plugin=SimpleNamespace(),
            semaphore=asyncio.Semaphore(1),
            sampling_params=SimpleNamespace(),
            max_retries=1,
            request_id_factory=lambda: "rid",
        )

    @staticmethod
    async def _resolved_task(value: VllmCaptionRequest) -> VllmCaptionRequest:
        return value

    @staticmethod
    async def _failing_task(exc: Exception) -> VllmCaptionRequest:
        raise exc

    @pytest.mark.asyncio
    async def test_clears_inputs_on_terminal_success(self) -> None:
        """``stage2_prompt is None`` -> after ``_emit_final``, ``req.inputs is None``."""
        captioner = self._build_captioner()
        payload: dict[str, Any] = {"prompt": "p"}
        req = VllmCaptionRequest(request_id="r1", inputs=payload, stage2_prompt=None, caption="c")

        task = asyncio.create_task(self._resolved_task(req))
        await asyncio.sleep(0)  # let the task complete
        captioner._pending[task] = req
        captioner._phase[task] = "stage1"
        captioner._request_id_to_index[req.request_id] = 0

        captioner._handle_completed(task)

        assert req.inputs is None

    @pytest.mark.asyncio
    async def test_clears_inputs_on_per_window_failure(self) -> None:
        """Exception (non-EngineDeadError) -> sentinel + ``req.inputs is None``."""
        captioner = self._build_captioner()
        payload: dict[str, Any] = {"prompt": "p"}
        req = VllmCaptionRequest(request_id="r1", inputs=payload, stage2_prompt=None)

        task = asyncio.create_task(self._failing_task(RuntimeError("boom")))
        await asyncio.sleep(0)
        captioner._pending[task] = req
        captioner._phase[task] = "stage1"
        captioner._request_id_to_index[req.request_id] = 0

        captioner._handle_completed(task)

        assert req.inputs is None
        # Sentinel result was emitted.
        from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION  # noqa: PLC0415

        assert captioner._results[0].text == VLLM_UNKNOWN_CAPTION

    @pytest.mark.asyncio
    async def test_clears_stage1_inputs_after_stage2_spawn(self) -> None:
        """Stage-2 dispatch succeeds -> stage-1 ``req.inputs is None`` (refined owns its own)."""
        import contextlib  # noqa: PLC0415

        captioner = self._build_captioner()
        payload: dict[str, Any] = {"prompt": "p", "multi_modal_data": {"video": object()}}
        req = VllmCaptionRequest(request_id="r1", inputs=payload, stage2_prompt="refine", caption="c")

        refined_payload: dict[str, Any] = {"prompt": "p2", "multi_modal_data": payload["multi_modal_data"]}
        refined = VllmCaptionRequest(request_id="r2", inputs=refined_payload, stage2_prompt=None)
        captioner.plugin = SimpleNamespace(make_refined_llm_request=lambda _r, _p, _t: refined)

        task = asyncio.create_task(self._resolved_task(req))
        await asyncio.sleep(0)
        captioner._pending[task] = req
        captioner._phase[task] = "stage1"
        captioner._request_id_to_index[req.request_id] = 0

        captioner._handle_completed(task)

        assert req.inputs is None
        assert refined.inputs is refined_payload  # refined still owns its own payload
        # Stage-2 task is in flight; cancel it before teardown so asyncio
        # does not log "Task exception was never retrieved" at GC.
        for pending_task in list(captioner._pending):
            pending_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await pending_task

    @pytest.mark.asyncio
    async def test_clears_inputs_on_stage2_build_failure(self) -> None:
        """``make_refined_llm_request`` raises -> ``req.inputs is None`` + sentinel emitted."""
        captioner = self._build_captioner()
        payload: dict[str, Any] = {"prompt": "p"}
        req = VllmCaptionRequest(request_id="r1", inputs=payload, stage2_prompt="refine", caption="c")

        def _raise(_r: Any, _p: Any, _t: Any) -> None:  # noqa: ANN401
            msg = "stage-2 build failed"
            raise ValueError(msg)

        captioner.plugin = SimpleNamespace(make_refined_llm_request=_raise)

        task = asyncio.create_task(self._resolved_task(req))
        await asyncio.sleep(0)
        captioner._pending[task] = req
        captioner._phase[task] = "stage1"
        captioner._request_id_to_index[req.request_id] = 0

        captioner._handle_completed(task)

        assert req.inputs is None
