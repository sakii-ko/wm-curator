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

"""CPU tests for the thin NormalCrafter model boundary."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from cosmos_curator.models import normalcrafter
from cosmos_curator.models.normalcrafter import (
    NORMALCRAFTER_MODEL_ID,
    NORMALCRAFTER_VAE_CHUNK_SIZE,
    NormalCrafterModel,
    NormalCrafterRawChunk,
)


class FakeRuntime:
    """Record inference and emit seven-frame raw chunks."""

    def __init__(self) -> None:
        """Initialize an unloaded fake output stream."""
        self.infer_called = False
        self.closed = False

    def infer(
        self,
        frames: np.ndarray,
    ) -> Iterator[NormalCrafterRawChunk]:
        """Yield deterministic raw normals with one invalid pixel."""
        self.infer_called = True
        for start in range(0, len(frames), NORMALCRAFTER_VAE_CHUNK_SIZE):
            stop = min(start + NORMALCRAFTER_VAE_CHUNK_SIZE, len(frames))
            values = np.empty((*frames[start:stop].shape[:3], 3), dtype=np.float32)
            values[...] = (3.0, 0.0, 4.0)
            values[:, 0, 0] = 0.0
            yield NormalCrafterRawChunk(frame_start=start, values=values)

    def close(self) -> None:
        """Record actor teardown."""
        self.closed = True


def test_model_is_lazy_and_canonicalizes_chunked_output(tmp_path: Path) -> None:
    """Setup owns heavy loading; infer preserves fixed recipe and chunk bounds."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    runtime = FakeRuntime()
    factory_calls: list[Path] = []

    def factory(path: Path) -> FakeRuntime:
        factory_calls.append(path)
        return runtime

    model = NormalCrafterModel(
        checkpoint_path=checkpoint,
        max_frames=20,
        runtime_factory=factory,
    )
    assert model.conda_env_name == "normalcrafter"
    assert model.model_id_names == []
    assert factory_calls == []

    model.setup()
    model.setup()
    assert factory_calls == [checkpoint]

    frames = np.zeros((15, 2, 3, 3), dtype=np.uint8)
    chunks = list(model.infer(frames))
    assert [(chunk.frame_start, chunk.frame_stop) for chunk in chunks] == [
        (0, 7),
        (7, 14),
        (14, 15),
    ]
    assert runtime.infer_called
    assert chunks[0].normal.dtype == np.float16
    assert chunks[0].valid.dtype == np.bool_
    assert not chunks[0].valid[:, 0, 0].any()
    assert np.count_nonzero(chunks[0].normal[:, 0, 0]) == 0
    np.testing.assert_allclose(
        chunks[0].normal[:, 1, 1],
        np.tile(
            np.asarray((-0.6, 0.0, 0.8), dtype=np.float16),
            (7, 1),
        ),
        atol=1.0e-3,
    )

    model.close()
    model.close()
    assert runtime.closed


def test_implicit_checkpoint_uses_cosmos_model_identity() -> None:
    """Only implicit checkpoint lookup participates in model-cache staging."""
    model = NormalCrafterModel(runtime_factory=lambda _path: FakeRuntime())
    assert model.model_id == NORMALCRAFTER_MODEL_ID
    assert model.model_id_names == [NORMALCRAFTER_MODEL_ID]


def test_fixed_temporal_recipe() -> None:
    """Window scheduling and tail blending match the released algorithm."""
    assert normalcrafter._temporal_windows(14) == ((0, 14),)
    assert normalcrafter._temporal_windows(25) == (
        (0, 14),
        (10, 24),
        (11, 25),
    )
    assert normalcrafter._padding_to_multiple(65, 66, 64) == (31, 32, 31, 31)


def test_window_latent_shape_uses_image_channels_not_unet_channels() -> None:
    """Four noise channels plus four image channels form the eight-channel UNet input."""
    assert normalcrafter._window_latent_shape(
        unet_in_channels=8,
        image_latent_channels=4,
        height=128,
        width=192,
        vae_scale_factor=8,
    ) == (1, 14, 4, 16, 24)
    with pytest.raises(ValueError, match="noise plus image"):
        normalcrafter._window_latent_shape(
            unet_in_channels=8,
            image_latent_channels=8,
            height=128,
            width=192,
            vae_scale_factor=8,
        )


def test_model_rejects_unbounded_full_clip_before_runtime() -> None:
    """The first runtime is explicit about its retained full-clip latent limit."""
    runtime = FakeRuntime()
    model = NormalCrafterModel(
        checkpoint_path="/unused",
        max_frames=14,
        runtime_factory=lambda _path: runtime,
    )
    model.setup()
    frames = np.zeros((15, 1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="explicit max_frames=14"):
        list(model.infer(frames))
    assert not runtime.infer_called
