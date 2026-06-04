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
r"""vLLM plugins for the Cosmos3 omnimodel reasoner head.

The public ``nvidia/Cosmos3-Nano`` / ``nvidia/Cosmos3-Super`` repos are
Mixture-of-Transformers omnimodels (autoregressive tower + diffusion tower).
The autoregressive tower can be served standalone for text-output reasoning
by overriding the architecture string, which is registered by the
``vllm-cosmos3`` package (shipped via the ``unified`` pixi env).

Reference invocation from the model card::

    vllm serve nvidia/Cosmos3-Nano \
        --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
        --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling \
        --allowed-local-media-path / \
        --media-io-kwargs '{"video": {"num_frames": -1}}'

We mirror those flags in-process here so the captioner gets the same setup.
"""

from typing import Any

from vllm import LLM
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import AsyncEngineArgs

from cosmos_curator.models.vllm_qwen import (
    GPU_MEMORY_UTILIZATION,
    LIMIT_MM_PER_PROMPT_IMAGE,
    LIMIT_MM_PER_PROMPT_VIDEO,
    MAX_MODEL_LEN,
    TRUST_REMOTE_CODE,
    VllmQwen3VL,
)
from cosmos_curator.pipelines.video.utils.data_model import VllmAsyncConfig, VllmConfig

_OMNI_HF_OVERRIDES = {"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}
_OMNI_MEDIA_IO_KWARGS = {"video": {"num_frames": -1}}
_OMNI_ALLOWED_LOCAL_MEDIA_PATH = "/"
_OMNI_MM_ENCODER_TP_MODE = "data"


class VllmCosmos3NanoOmniVL(VllmQwen3VL):
    """Cosmos3-Nano omnimodel reasoner-head vLLM plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "cosmos3_nano"

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        """Instantiate the vLLM model with the Cosmos3 reasoner-head overrides."""
        limit_mm = LIMIT_MM_PER_PROMPT_IMAGE if config.use_image_input else LIMIT_MM_PER_PROMPT_VIDEO
        return LLM(
            model=str(cls.model_path(config)),
            hf_overrides=_OMNI_HF_OVERRIDES,
            limit_mm_per_prompt=limit_mm,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            pipeline_parallel_size=1,
            tensor_parallel_size=config.num_gpus,
            trust_remote_code=TRUST_REMOTE_CODE,
            mm_processor_cache_gb=0.0 if config.disable_mmcache else 4.0,
            mm_encoder_tp_mode=_OMNI_MM_ENCODER_TP_MODE,
            # async_scheduling is documented by the model card only for `vllm serve`
            # (the async/server path); keep it off for the offline LLM path.
            async_scheduling=False,
            allowed_local_media_path=_OMNI_ALLOWED_LOCAL_MEDIA_PATH,
            media_io_kwargs=_OMNI_MEDIA_IO_KWARGS,
            compilation_config={"cudagraph_mode": "piecewise"},
            performance_mode=config.performance_mode,
        )

    @classmethod
    def model_async(cls, config: VllmAsyncConfig) -> AsyncEngineArgs:
        """Build ``AsyncEngineArgs`` for the in-process ``AsyncLLM`` path."""
        extra_kwargs: dict[str, Any] = {}
        if config.gpu_memory_utilization is not None:
            extra_kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization
        return AsyncEngineArgs(
            model=str(cls.model_path(config.to_vllm_config())),
            hf_overrides=_OMNI_HF_OVERRIDES,
            served_model_name=[config.model_variant],
            tensor_parallel_size=config.num_gpus,
            data_parallel_size=config.data_parallel_size,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=TRUST_REMOTE_CODE,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT_VIDEO,  # type: ignore[arg-type]
            max_num_seqs=config.max_num_seqs if config.max_num_seqs > 0 else None,
            enforce_eager=config.enforce_eager,
            kv_cache_dtype=config.kv_cache_dtype,  # type: ignore[arg-type]
            mm_encoder_tp_mode=config.mm_encoder_tp_mode or _OMNI_MM_ENCODER_TP_MODE,  # type: ignore[arg-type]
            mm_processor_cache_type=config.mm_processor_cache_type or None,  # type: ignore[arg-type]
            # The model card recommends --async-scheduling for the server path; default it
            # on for omni while still honoring an explicit override from the config.
            async_scheduling=config.async_scheduling if config.async_scheduling is not None else True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            disable_chunked_mm_input=config.disable_chunked_mm_input,
            long_prefill_token_threshold=config.long_prefill_token_threshold,
            stream_interval=config.stream_interval,
            distributed_executor_backend=config.distributed_executor_backend,
            skip_mm_profiling=config.skip_mm_profiling,
            disable_log_stats=config.disable_log_stats,
            enable_log_requests=config.enable_log_requests,
            mm_processor_cache_gb=0.0 if config.disable_mmcache else 4.0,
            allowed_local_media_path=_OMNI_ALLOWED_LOCAL_MEDIA_PATH,
            media_io_kwargs=_OMNI_MEDIA_IO_KWARGS,
            compilation_config=CompilationConfig(cudagraph_mode="piecewise"),  # type: ignore[arg-type]
            enable_prefix_caching=True,
            use_tqdm_on_load=False,
            **extra_kwargs,
        )


class VllmCosmos3SuperOmniVL(VllmCosmos3NanoOmniVL):
    """Cosmos3-Super omnimodel reasoner-head vLLM plugin (TP=4 minimum on H100)."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "cosmos3_super"
