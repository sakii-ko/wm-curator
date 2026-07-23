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

"""Clip Frame Extraction Stage."""

import io
import math
import pathlib
from functools import reduce
from typing import Final, Literal

import cv2
import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.sensors.sampling.compat import make_decoder_utils_compat_grid
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid, make_ts_grid
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.core.utils.data.ref_resolver import prefetch, resolve_as_ready
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask, Video
from cosmos_curator.pipelines.video.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
    decode_video_cpu,
    extract_frames,
    get_video_timestamps,
)


class ClipFrameExtractionStage(CuratorStage):
    """Stage for extracting frames from video clips.

    This class processes video clips through a series of steps including frame extraction,
    target frame rate selection, and frame extraction signature creation.
    """

    DEFAULT_DECODER_MODE: Final = "extract_frames"
    CAMERA_SENSOR_DECODER_MODE: Final = "camera_sensor"

    def __init__(  # noqa: PLR0913
        self,
        extraction_policies: tuple[FrameExtractionPolicy, ...] = (FrameExtractionPolicy.sequence,),
        target_fps: list[float | int] | None = None,
        target_res: tuple[int, int] | None = None,
        *,
        decoder_mode: Literal["extract_frames", "camera_sensor"] = DEFAULT_DECODER_MODE,
        num_cpus_per_worker: float = 3.0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the clip frame extraction stage.

        Args:
            extraction_policies: Frame extraction policies to use.
            target_fps: Target frames per second for extraction.
            target_res: Target resolution for extracted frames.
            decoder_mode: Backend used to decode per-clip frames.
            num_cpus_per_worker: Number of CPU cores to allocate per worker.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        if target_fps is None:
            target_fps = [2]
        if target_res is None:
            target_res = (-1, -1)
        self._timer = StageTimer(self)
        self._extraction_policies = extraction_policies
        self._target_fps = target_fps
        self._target_res = target_res
        self._decoder_mode = decoder_mode
        self._num_cpus = num_cpus_per_worker
        self._num_threads = max(1, int(num_cpus_per_worker) + 1)
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(cpus=self._num_cpus)

    def lcm_multiple(self, fps: list[float | int]) -> float | int:
        """Compute LCM of a list of fps targets."""

        def lcm(a: float, b: float) -> float | int:
            return abs(a * b) // math.gcd(int(a), int(b))

        return reduce(lcm, fps)

    def _make_signature(self, policy: FrameExtractionPolicy, fps: float) -> str:
        return FrameExtractionSignature(
            extraction_policy=policy,
            target_fps=fps,
        ).to_str()

    def _use_lcm_fps(self) -> bool:
        return len(self._target_fps) > 1 and all(
            (fps.is_integer() if isinstance(fps, float) else isinstance(fps, int)) for fps in self._target_fps
        )

    def _resize_frames(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self._target_res[0] > 0 and self._target_res[1] > 0:
            interpolation = cv2.INTER_CUBIC
            return np.array(
                [
                    cv2.resize(frame, (self._target_res[1], self._target_res[0]), interpolation=interpolation)
                    for frame in frames
                ]
            )
        return frames

    def _extract_frames_default(
        self,
        data: bytes | npt.NDArray[np.uint8],
        policy: FrameExtractionPolicy,
        fps: float,
    ) -> npt.NDArray[np.uint8]:
        with io.BytesIO(data) as fp:
            return extract_frames(
                fp,
                extraction_policy=policy,
                sample_rate_fps=fps,
                target_res=self._target_res,
                num_threads=self._num_threads,
            )

    def _sample_sensor(
        self,
        sensor: CameraSensor,
        sample_rate_fps: float,
        span: tuple[float, float] | None = None,
    ) -> npt.NDArray[np.uint8]:
        if span is None:
            start_ns, exclusive_end_ns, timestamps_ns = make_decoder_utils_compat_grid(
                start_ns=sensor.start_ns,
                stop_ns=sensor.end_ns,
                sample_rate_hz=float(sample_rate_fps),
            )
        else:
            start_ns = max(sensor.start_ns, sensor.start_ns + round(span[0] * 1_000_000_000))
            exclusive_end_ns = min(sensor.end_ns + 1, sensor.start_ns + round(span[1] * 1_000_000_000))
            start_ns, exclusive_end_ns, timestamps_ns = make_ts_grid(
                start_ns,
                exclusive_end_ns=exclusive_end_ns,
                sample_rate_hz=float(sample_rate_fps),
            )
        grid = SamplingGrid(
            start_ns=start_ns,
            exclusive_end_ns=exclusive_end_ns,
            timestamps_ns=timestamps_ns,
            stride_ns=max(1, exclusive_end_ns - start_ns),
            duration_ns=max(1, exclusive_end_ns - start_ns),
        )
        spec = SamplingSpec(grid=grid)
        sampled_batches = list(sensor.sample(spec))
        if len(sampled_batches) != 1:
            msg = f"Expected exactly one sampled batch, got {len(sampled_batches)}"
            raise RuntimeError(msg)
        return self._resize_frames(sampled_batches[0].frames)

    def _sample_with_camera_sensor(
        self,
        data: bytes | npt.NDArray[np.uint8],
        sample_rate_fps: float,
    ) -> npt.NDArray[np.uint8]:
        return self._sample_sensor(CameraSensor(bytes(data)), sample_rate_fps)

    def _extract_frames_for_policy(
        self,
        data: bytes | npt.NDArray[np.uint8],
        policy: FrameExtractionPolicy,
    ) -> dict[str, npt.NDArray[np.uint8]]:
        local_frames: dict[str, npt.NDArray[np.uint8]] = {}
        use_camera_sensor = self._decoder_mode == self.CAMERA_SENSOR_DECODER_MODE
        if use_camera_sensor and policy is not FrameExtractionPolicy.sequence:
            msg = f"CameraSensor clip frame extraction only supports {FrameExtractionPolicy.sequence!s}, got {policy!s}"
            raise NotImplementedError(msg)

        if not use_camera_sensor and self._use_lcm_fps():
            lcm = self.lcm_multiple(self._target_fps)
            frames = self._extract_frames_default(data, policy, lcm)
            for fps in self._target_fps:
                signature = self._make_signature(policy, fps)
                stride = int(lcm / fps)
                local_frames[signature] = frames[::stride]
            return local_frames

        for fps in self._target_fps:
            if use_camera_sensor:
                frames = self._sample_with_camera_sensor(data, fps)
            else:
                frames = self._extract_frames_default(data, policy, fps)
            local_frames[self._make_signature(policy, fps)] = frames
        return local_frames

    def _extract_source_frames(
        self,
        sensor: CameraSensor,
        span: tuple[float, float],
        rotation_degrees_clockwise: int | None,
    ) -> dict[str, npt.NDArray[np.uint8]]:
        if any(policy is not FrameExtractionPolicy.sequence for policy in self._extraction_policies):
            msg = "Source-backed frame extraction only supports the sequence policy"
            raise NotImplementedError(msg)

        frames_by_signature: dict[str, npt.NDArray[np.uint8]] = {}
        for fps in self._target_fps:
            frames = self._sample_sensor(sensor, float(fps), span)
            frames = self._rotate_source_frames(frames, rotation_degrees_clockwise)
            frames_by_signature[self._make_signature(FrameExtractionPolicy.sequence, fps)] = frames
        return frames_by_signature

    def _extract_source_frames_fallback(
        self,
        source_path: pathlib.Path,
        timestamps: npt.NDArray[np.float32],
        span: tuple[float, float],
        stream_index: int,
        rotation_degrees_clockwise: int | None,
    ) -> dict[str, npt.NDArray[np.uint8]]:
        """Decode source spans whose container has no CameraSensor header index."""
        if any(policy is not FrameExtractionPolicy.sequence for policy in self._extraction_policies):
            msg = "Source-backed frame extraction only supports the sequence policy"
            raise NotImplementedError(msg)

        source_start = float(timestamps[0])
        frames_by_signature: dict[str, npt.NDArray[np.uint8]] = {}
        for fps in self._target_fps:
            frames = decode_video_cpu(
                source_path,
                float(fps),
                timestamps=timestamps,
                start=source_start + span[0],
                stop=source_start + span[1],
                endpoint=False,
                stream_idx=stream_index,
                num_threads=self._num_threads,
            )
            frames = self._resize_frames(frames)
            frames = self._rotate_source_frames(frames, rotation_degrees_clockwise)
            frames_by_signature[self._make_signature(FrameExtractionPolicy.sequence, fps)] = frames
        return frames_by_signature

    @staticmethod
    def _rotate_source_frames(
        frames: npt.NDArray[np.uint8],
        rotation_degrees_clockwise: int | None,
    ) -> npt.NDArray[np.uint8]:
        if rotation_degrees_clockwise not in {None, 0, 90, 180, 270}:
            msg = f"rotation_degrees_clockwise must be one of 0, 90, 180, or 270, got {rotation_degrees_clockwise}"
            raise ValueError(msg)
        if not rotation_degrees_clockwise:
            return frames
        return np.rot90(frames, k=-(rotation_degrees_clockwise // 90), axes=(1, 2)).copy()

    def _extract_clip_frames(  # noqa: PLR0913
        self,
        data: npt.NDArray[np.uint8] | None,
        clip_span: tuple[float, float],
        source_path: pathlib.Path,
        source_sensor: CameraSensor | None,
        source_timestamps: npt.NDArray[np.float32] | None,
        stream_index: int,
        rotation_degrees_clockwise: int | None,
    ) -> dict[str, npt.NDArray[np.uint8]]:
        if data is not None:
            local_frames: dict[str, npt.NDArray[np.uint8]] = {}
            for policy in self._extraction_policies:
                local_frames.update(self._extract_frames_for_policy(data, policy))
            return local_frames
        if source_sensor is not None:
            return self._extract_source_frames(source_sensor, clip_span, rotation_degrees_clockwise)
        if source_timestamps is not None:
            return self._extract_source_frames_fallback(
                source_path,
                source_timestamps,
                clip_span,
                stream_index,
                rotation_degrees_clockwise,
            )
        msg = "Clip has neither encoded_data nor a local source path"
        raise ValueError(msg)

    def _process_video(
        self,
        video: Video,
        *,
        stream_index: int = 0,
        rotation_degrees_clockwise: int | None = None,
    ) -> None:
        if self._verbose:
            logger.info(f"Processing video {video.input_video} with {len(video.clips)} clips")

        source_sensor: CameraSensor | None = None
        source_timestamps: npt.NDArray[np.float32] | None = None
        source_path = pathlib.Path(video.input_path)
        if source_path.is_file() and any(not clip.encoded_data for clip in video.clips):
            try:
                source_sensor = CameraSensor(source_path, stream_idx=stream_index)
            except Exception as error:  # noqa: BLE001
                logger.debug(f"Falling back to PyAV source span extraction for {source_path}: {error}")
                source_timestamps = get_video_timestamps(source_path, stream_idx=stream_index)

        prefetch([clip.encoded_data for clip in video.clips])
        for clip, data in resolve_as_ready([(clip, clip.encoded_data) for clip in video.clips]):
            try:
                local_frames = self._extract_clip_frames(
                    data,
                    clip.span,
                    source_path,
                    source_sensor,
                    source_timestamps,
                    stream_index,
                    rotation_degrees_clockwise,
                )
                if self._verbose:
                    for signature, frame_array in local_frames.items():
                        logger.info(f"Extracted {len(frame_array)} frames from clip {clip.uuid} for {signature=}")

                total_nbytes = sum(arr.nbytes for arr in local_frames.values())
                clip.extracted_frames = LazyData(value=local_frames, nbytes=total_nbytes)
                # Phase 2: call extracted_frames.store() here to push to Plasma
                # before the stage boundary for split-field transport
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Error extracting frames from clip {clip.uuid}: {e}")
                clip.errors["frame_extraction"] = "video_decode_failed"
                # reset the buffer to disable further operations on this clip
                clip.encoded_data.drop()
                continue

    @nvtx.annotate("ClipFrameExtractionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the clip frame extraction stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            for video in task.videos:
                with self._timer.time_process():
                    try:
                        self._process_video(
                            video,
                            stream_index=getattr(task, "stream_index", None) or 0,
                            rotation_degrees_clockwise=getattr(task, "rotation_degrees_clockwise", None),
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"Error processing video {video.input_video}")
                        video.errors[self.__class__.__name__] = str(e)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
