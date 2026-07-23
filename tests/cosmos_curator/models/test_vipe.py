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
"""CPU-only contract tests for the lazy ViPE model adapter."""

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import cosmos_curator.models.vipe as vipe_module
from cosmos_curator.models.vipe import ViPEFrameResult, ViPEModel, ViPEModelConfig


class _FakeRuntime:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[tuple[int, ...], str, float]] = []

    def infer(
        self,
        frames: npt.NDArray[np.uint8],
        *,
        name: str,
        fps: float,
    ) -> Iterator[ViPEFrameResult]:
        self.calls.append((frames.shape, name, fps))
        return iter(())

    def close(self) -> None:
        self.closed = True


def test_model_defers_runtime_loading_until_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Constructing the model must not import or initialize ViPE."""
    vipe_was_imported = "vipe" in sys.modules
    runtime = _FakeRuntime()
    loads: list[ViPEModelConfig] = []

    def fake_load(config: ViPEModelConfig) -> _FakeRuntime:
        loads.append(config)
        return runtime

    monkeypatch.setattr(vipe_module, "_load_vipe_runtime", fake_load)
    config = ViPEModelConfig(
        slam_model_path=tmp_path / "slam",
        post_model_path=tmp_path / "post",
    )
    model = ViPEModel(config)

    assert loads == []
    assert ("vipe" in sys.modules) is vipe_was_imported
    assert model.conda_env_name == "vipe"
    assert model.model_id_names == []

    model.setup()
    model.setup()
    assert loads == [config]

    frames = np.zeros((8, 16, 24, 3), dtype=np.uint8)
    assert list(model.infer(frames, name="clip", fps=29.97)) == []
    assert runtime.calls == [((8, 16, 24, 3), "clip", 29.97)]

    model.close()
    assert runtime.closed


def test_model_requires_setup_before_inference(tmp_path: Path) -> None:
    """Inference should fail clearly before the actor lifecycle calls setup."""
    model = ViPEModel(
        ViPEModelConfig(
            slam_model_path=tmp_path / "slam",
            post_model_path=tmp_path / "post",
        )
    )

    with pytest.raises(RuntimeError, match=r"setup\(\) must be called"):
        model.infer(np.zeros((8, 16, 16, 3), dtype=np.uint8), name="clip", fps=30.0)


def test_production_runtime_close_removes_only_its_scratch_dir(tmp_path: Path) -> None:
    """Simple runtime cleanup should remove its unique temporary directory."""
    work_dir = tmp_path / "cosmos-curator-vipe-test"
    work_dir.mkdir()
    (work_dir / "artifact").touch()
    runtime = vipe_module._ProductionViPERuntime(
        pipeline=object(),
        device=object(),
        video_frame_type=object,
        video_stream_type=object,
        work_dir=work_dir,
    )

    runtime.close()
    runtime.close()

    assert not work_dir.exists()
    assert tmp_path.exists()
    with pytest.raises(RuntimeError, match="runtime is closed"):
        list(runtime.infer(np.zeros((8, 16, 16, 3), dtype=np.uint8), name="closed", fps=30.0))
