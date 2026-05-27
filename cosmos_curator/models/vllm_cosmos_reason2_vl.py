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
"""vLLM plugin for Cosmos-Reason2 vision-language model."""

from typing import Any

import torch
from transformers import AutoProcessor

from cosmos_curator.models.vllm_cosmos_reason1_vl import VllmCosmosReason1VL, make_message, make_prompt
from cosmos_curator.models.vllm_qwen import qwen3_video_size_kwargs
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig


class VllmCosmosReason2VL(VllmCosmosReason1VL):
    """Cosmos-Reason2 vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "cosmos_r2"

    @staticmethod
    def make_llm_input(
        prompt: str,
        frames: torch.Tensor,
        metadata: dict[str, Any],
        processor: AutoProcessor,
        config: VllmConfig,
    ) -> dict[str, Any]:
        """Make LLM inputs for Cosmos-Reason2 with Qwen3-style video sizing."""
        message = make_message(prompt, use_image=config.use_image_input)
        inputs = make_prompt(message, frames, metadata, processor, use_image=config.use_image_input)
        if config.video_max_pixels_per_frame is not None and not config.use_image_input:
            inputs["mm_processor_kwargs"] = qwen3_video_size_kwargs(
                int(frames.shape[0]),
                config.video_max_pixels_per_frame,
            )
        return inputs
