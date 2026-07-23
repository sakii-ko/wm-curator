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
"""Lazy, local-weight adapter for ViPE's DAv3 geometry pipeline."""

import math
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.interfaces.model_interface import ModelInterface

_FRAME_ARRAY_NDIM = 4
_RGB_CHANNELS = 3


@dataclass(frozen=True, slots=True)
class ViPEFrameResult:
    """One frame of ViPE output, copied to host memory."""

    raw_frame_idx: int
    metric_depth: npt.NDArray[np.float32]
    intrinsics: npt.NDArray[np.float32]
    camera_to_world: npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ViPEModelConfig:
    """Only the settings needed to construct ViPE's DAv3 inference pipeline."""

    slam_model_path: Path
    post_model_path: Path
    slam_weights_prefix: str | None = None
    post_weights_prefix: str | None = None
    torch_home: Path | None = None

    def __post_init__(self) -> None:
        """Normalize local paths without touching the filesystem."""
        object.__setattr__(self, "slam_model_path", Path(self.slam_model_path).expanduser())
        object.__setattr__(self, "post_model_path", Path(self.post_model_path).expanduser())
        if self.torch_home is not None:
            object.__setattr__(self, "torch_home", Path(self.torch_home).expanduser())


class _ViPERuntime(Protocol):
    def infer(
        self,
        frames: npt.NDArray[np.uint8],
        *,
        name: str,
        fps: float,
    ) -> Iterator[ViPEFrameResult]:
        """Yield one host-resident result at a time."""

    def close(self) -> None:
        """Release references to resident ViPE models."""


class ViPEModel(ModelInterface):
    """Resident ViPE model with imports and CUDA initialization deferred to setup."""

    def __init__(
        self,
        config: ViPEModelConfig,
        *,
        conda_env_name: str = "vipe",
    ) -> None:
        """Store local model locations without importing torch or ViPE."""
        if not isinstance(conda_env_name, str) or not conda_env_name.strip():
            msg = "conda_env_name must be a non-empty string"
            raise ValueError(msg)
        self.config = config
        self._conda_env_name = conda_env_name
        self._runtime: _ViPERuntime | None = None

    @property
    def conda_env_name(self) -> str:
        """Return the environment containing the vendored ViPE package."""
        return self._conda_env_name

    @property
    def model_id_names(self) -> list[str]:
        """Return no catalog IDs because ViPE weights are explicit local paths."""
        return []

    def setup(self) -> None:
        """Lazily construct one resident ViPE DAv3 pipeline."""
        if self._runtime is None:
            self._runtime = _load_vipe_runtime(self.config)

    def infer(
        self,
        frames: npt.NDArray[np.uint8],
        *,
        name: str,
        fps: float,
    ) -> Iterator[ViPEFrameResult]:
        """Run ViPE while yielding, rather than retaining, host output frames."""
        if self._runtime is None:
            msg = "ViPEModel.setup() must be called before infer()"
            raise RuntimeError(msg)
        return self._runtime.infer(frames, name=name, fps=fps)

    def close(self) -> None:
        """Drop the resident runtime."""
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None


def _load_vipe_runtime(config: ViPEModelConfig) -> _ViPERuntime:
    """Import ViPE only inside the target worker and construct its DAv3 pipeline."""
    slam_model_path = config.slam_model_path.resolve(strict=True)
    post_model_path = config.post_model_path.resolve(strict=True)
    if config.torch_home is not None:
        torch_home = config.torch_home.resolve()
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)

    import torch  # noqa: PLC0415 - model dependencies must stay actor-lazy

    if not torch.cuda.is_available():
        msg = "ViPE DAv3 requires a CUDA device"
        raise RuntimeError(msg)
    if config.torch_home is not None:
        torch.hub.set_dir(str(config.torch_home.resolve() / "hub"))

    from vipe.config import parse_typed_config  # type: ignore[import-not-found]  # noqa: PLC0415
    from vipe.pipeline import make_pipeline  # type: ignore[import-not-found]  # noqa: PLC0415
    from vipe.streams.base import VideoFrame, VideoStream  # type: ignore[import-not-found]  # noqa: PLC0415

    work_dir = Path(tempfile.mkdtemp(prefix="cosmos-curator-vipe-"))
    try:
        overrides = _runtime_overrides(
            config,
            slam_model_path=slam_model_path,
            post_model_path=post_model_path,
            work_dir=work_dir,
        )
        vipe_config = parse_typed_config("default", hydra_args=overrides)
        pipeline = make_pipeline(vipe_config.pipeline)
        pipeline.return_output_streams = True
    except BaseException:
        shutil.rmtree(work_dir)
        raise
    return _ProductionViPERuntime(
        pipeline=pipeline,
        device=torch.device("cuda", torch.cuda.current_device()),
        video_frame_type=VideoFrame,
        video_stream_type=VideoStream,
        work_dir=work_dir,
    )


def _runtime_overrides(
    config: ViPEModelConfig,
    *,
    slam_model_path: Path,
    post_model_path: Path,
    work_dir: Path,
) -> list[str]:
    """Translate the small adapter config to ViPE's supported Hydra overrides."""
    overrides = [
        "pipeline=dav3",
        "streams=raw_mp4_stream",
        f"streams.base_path={work_dir / 'unused.mp4'}",
        "pipeline.init.instance=null",
        "pipeline.init.async_prefetch=false",
        f"pipeline.slam.dav3_weights_path={slam_model_path}",
        f"pipeline.post.dav3_weights_path={post_model_path}",
        f"pipeline.output.path={work_dir}",
        "pipeline.output.save_artifacts=false",
        "pipeline.output.save_slam_map=false",
        "pipeline.output.save_viz=false",
        "pipeline.output.skip_exists=false",
    ]
    if config.slam_weights_prefix is not None:
        overrides.append(f"pipeline.slam.dav3_weights_prefix={config.slam_weights_prefix}")
    if config.post_weights_prefix is not None:
        overrides.append(f"pipeline.post.dav3_weights_prefix={config.post_weights_prefix}")
    return overrides


class _ProductionViPERuntime:
    """Thin bridge to the audited final ViPE cached stream."""

    def __init__(
        self,
        *,
        pipeline: Any,  # noqa: ANN401 - ViPE intentionally remains an optional runtime dependency
        device: Any,  # noqa: ANN401 - torch intentionally remains an optional runtime dependency
        video_frame_type: type[Any],
        video_stream_type: type[Any],
        work_dir: Path,
    ) -> None:
        self._pipeline = pipeline
        self._device = device
        self._video_frame_type = video_frame_type
        self._video_stream_type = video_stream_type
        self._work_dir = work_dir
        self._closed = False

    def infer(
        self,
        frames: npt.NDArray[np.uint8],
        *,
        name: str,
        fps: float,
    ) -> Iterator[ViPEFrameResult]:
        """Yield slim CPU results directly from ViPE's lazy final stream."""
        if self._closed:
            msg = "ViPE runtime is closed"
            raise RuntimeError(msg)
        stream = _make_numpy_video_stream(
            frames,
            name=name,
            fps=fps,
            device=self._device,
            video_frame_type=self._video_frame_type,
            video_stream_type=self._video_stream_type,
        )
        output: Any | None = None
        final_stream: Any | None = None
        try:
            output = self._pipeline.run(stream)
            if output.output_streams is None or len(output.output_streams) != 1:
                msg = "ViPE did not return exactly one final output stream"
                raise RuntimeError(msg)
            final_stream = output.output_streams[0]
            cached_data = getattr(final_stream, "data", None)
            frame_iterator = getattr(final_stream, "iterator", None)
            if not isinstance(cached_data, list) or cached_data or frame_iterator is None:
                msg = "ViPE final stream no longer matches the expected empty CachedVideoStream"
                raise RuntimeError(msg)
            for frame in frame_iterator:
                if frame.metric_depth is None or frame.pose is None or frame.intrinsics is None:
                    msg = f"ViPE frame {frame.raw_frame_idx} is missing depth, pose, or intrinsics"
                    raise ValueError(msg)
                result = ViPEFrameResult(
                    raw_frame_idx=int(frame.raw_frame_idx),
                    metric_depth=_owned_numpy(frame.metric_depth),
                    intrinsics=_owned_numpy(frame.intrinsics),
                    camera_to_world=_owned_numpy(frame.pose.matrix()),
                )
                del frame
                yield result
        finally:
            if final_stream is not None and hasattr(final_stream, "iterator"):
                final_stream.iterator = None
            del final_stream, output, stream

    def close(self) -> None:
        """Drop references to the pipeline and its resident model cache."""
        if self._closed:
            return
        self._closed = True
        self._pipeline = None
        shutil.rmtree(self._work_dir)


def _make_numpy_video_stream(  # noqa: PLR0913
    frames: npt.NDArray[np.uint8],
    *,
    name: str,
    fps: float,
    device: Any,  # noqa: ANN401 - torch intentionally remains an optional runtime dependency
    video_frame_type: type[Any],
    video_stream_type: type[Any],
) -> Any:  # noqa: ANN401 - ViPE intentionally remains an optional runtime dependency
    """Adapt decoded RGB frames without making another full-sequence tensor."""
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != _FRAME_ARRAY_NDIM
        or frames.shape[-1] != _RGB_CHANNELS
        or not frames.flags.c_contiguous
    ):
        msg = "ViPE frames must be contiguous uint8 [T,H,W,3] RGB"
        raise ValueError(msg)
    if not math.isfinite(fps) or fps <= 0:
        msg = "ViPE fps must be finite and positive"
        raise ValueError(msg)

    import torch  # noqa: PLC0415 - model dependencies must stay actor-lazy

    class NumpyVideoStream(video_stream_type):  # type: ignore[misc,valid-type]
        def frame_size(self) -> tuple[int, int]:
            return int(frames.shape[1]), int(frames.shape[2])

        def name(self) -> str:
            return name

        def fps(self) -> float:
            return fps

        def __len__(self) -> int:
            return int(frames.shape[0])

        def __iter__(self) -> Iterator[Any]:
            for index in range(len(self)):
                rgb = torch.from_numpy(frames[index]).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=False,
                )
                rgb.mul_(1.0 / 255.0)
                yield video_frame_type(raw_frame_idx=index, rgb=rgb)

    return NumpyVideoStream()


def _owned_numpy(value: object) -> npt.NDArray[np.float32]:
    """Copy one tensor-like value to an owned, contiguous CPU float32 array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "to"):
        value = value.to(device="cpu")
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32)).copy()
