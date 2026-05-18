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

"""Abstract base class for one-shot caption inference stages.

Per-window captioning is window-batched / async-concurrent / continuous-mode
by design — each backend stage owns the windowing + scheduling that makes
the per-window throughput acceptable. Per-event captioning, by contrast, is
one inference per clip (the SAM3 prompt and event JSON shape are per-clip).

The :class:`SingleInferenceCaptionStage` ABC is the small seam that lets
``PerEventCaptionStage`` reuse a per-window stage's already-initialised
client / engine for a single inference. It combines the full
:class:`CuratorStage` lifecycle (resources / conda_env_name / model /
stage_setup_on_node / stage_setup / destroy) with two abstract methods
that one-shot consumers depend on: :meth:`caption_single` and
:meth:`secondary_name`. ``PerEventCaptionStage`` forwards every
lifecycle hook directly without isinstance / duck-typed fallbacks.

Implementors:

* :class:`OpenAICaptionStage`
* :class:`GeminiCaptionStage`
* :class:`VllmCaptionStage`
* :class:`VllmAsyncCaptionStage`
"""

import abc

from cosmos_curator.core.interfaces.stage_interface import CuratorStage


class SingleInferenceCaptionStage(CuratorStage, abc.ABC):
    """``CuratorStage`` that supports one-shot per-clip caption inference.

    Implementors must run :meth:`caption_single` only after their
    ``stage_setup`` has produced a usable client / engine, and must
    provide a stable :meth:`secondary_name` for log tagging.

    Implementors should not modify any per-window pipeline state (clip
    errors, window captions, etc.) from :meth:`caption_single`; it is
    intended for single-shot use from another stage, not as a
    substitute for ``process_data``.
    """

    @abc.abstractmethod
    def caption_single(self, prompt: str, video_bytes: bytes) -> str:
        """Run one inference and return raw response text.

        Args:
            prompt: User prompt string. Caller is responsible for any
                templating; the implementation may apply a model-specific
                chat template internally.
            video_bytes: Whole-clip MP4 bytes. Implementations decode and/or
                base64-encode as needed for their backend.

        Returns:
            Raw response text from the model.

        Raises:
            RuntimeError: When the backend returns no usable text.

        """

    @abc.abstractmethod
    def secondary_name(self) -> str:
        """Return a short, stable identifier used for log tagging.

        Used by ``PerEventCaptionStage`` to suffix its own
        ``secondary_name`` so multi-backend runs are easy to read.
        """
