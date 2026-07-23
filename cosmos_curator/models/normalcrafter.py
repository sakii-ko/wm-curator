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

"""Thin NormalCrafter model lifecycle and streaming output adapter."""

import gc
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.interfaces.model_interface import ModelInterface

NORMALCRAFTER_MODEL_ID = "Yanrui95/NormalCrafter"
NORMALCRAFTER_WINDOW_SIZE = 14
NORMALCRAFTER_WINDOW_STRIDE = 10
NORMALCRAFTER_VAE_CHUNK_SIZE = 7
NORMALCRAFTER_PADDING_MULTIPLE = 64
NORMALCRAFTER_MAX_FRAMES = 1_350
NORMALCRAFTER_CONDITIONING_FPS = 7
_CLIP_ENCODE_CHUNK_SIZE = 32
_MOTION_BUCKET_ID = 127
_NOISE_AUG_STRENGTH = 0.0
_NORMAL_EPSILON = 1.0e-6
_VIDEO_NDIM = 4
_RGB_CHANNELS = 3


@dataclass(frozen=True, slots=True)
class NormalCrafterRawChunk:
    """One decoded runtime chunk before axis conversion and normalization."""

    frame_start: int
    values: npt.NDArray[np.float32]

    @property
    def frame_stop(self) -> int:
        """Return the exclusive temporal bound."""
        return self.frame_start + self.values.shape[0]


@dataclass(frozen=True, slots=True)
class NormalCrafterChunk:
    """One canonical CPU output chunk ready for annotation persistence."""

    frame_start: int
    normal: npt.NDArray[np.float16]
    valid: npt.NDArray[np.bool_]

    @property
    def frame_stop(self) -> int:
        """Return the exclusive temporal bound."""
        return self.frame_start + self.normal.shape[0]


class NormalCrafterRuntime(Protocol):
    """Heavy runtime surface loaded only by :meth:`NormalCrafterModel.setup`."""

    def infer(
        self,
        frames: npt.NDArray[np.uint8],
    ) -> Iterator[NormalCrafterRawChunk]:
        """Yield decoded raw normal chunks in temporal order."""

    def close(self) -> None:
        """Release runtime resources."""


type RuntimeFactory = Callable[[Path], NormalCrafterRuntime]


class NormalCrafterModel(ModelInterface):
    """Actor-resident NormalCrafter model with bounded output materialization."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        max_frames: int = NORMALCRAFTER_MAX_FRAMES,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        """Store lightweight configuration without importing Torch or Diffusers."""
        if isinstance(max_frames, bool) or not isinstance(max_frames, int):
            message = "max_frames must be an integer"
            raise TypeError(message)
        if max_frames < NORMALCRAFTER_WINDOW_SIZE:
            message = f"max_frames must be at least {NORMALCRAFTER_WINDOW_SIZE}"
            raise ValueError(message)

        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._max_frames = max_frames
        self._runtime_factory = runtime_factory
        self._runtime: NormalCrafterRuntime | None = None

    @property
    def conda_env_name(self) -> str:
        """Return the actor environment used for NormalCrafter."""
        return "normalcrafter"

    @property
    def model_id_names(self) -> list[str]:
        """Return the checkpoint registered for model download."""
        return [] if self._checkpoint_path is not None else [NORMALCRAFTER_MODEL_ID]

    @property
    def model_id(self) -> str:
        """Return the checkpoint identity used in annotation metadata."""
        return NORMALCRAFTER_MODEL_ID

    @property
    def max_frames(self) -> int:
        """Return the explicit full-latent safety limit."""
        return self._max_frames

    def setup(self) -> None:
        """Load the runtime once inside the model actor."""
        if self._runtime is not None:
            return
        checkpoint = self._checkpoint_path
        if checkpoint is None:
            from cosmos_curator.core.utils.model import model_utils  # noqa: PLC0415

            checkpoint = model_utils.get_local_dir_for_weights_name(NORMALCRAFTER_MODEL_ID)
        if self._runtime_factory is not None:
            self._runtime = self._runtime_factory(checkpoint)
        else:
            self._runtime = _DiffusersNormalCrafterRuntime.load(checkpoint)

    def infer(
        self,
        frames: npt.NDArray[np.uint8],
    ) -> Iterator[NormalCrafterChunk]:
        """Yield canonical ``(-x,+y,+z)`` unit normals without full-clip D2H."""
        _validate_input_frames(frames, max_frames=self._max_frames)
        runtime = self._runtime
        if runtime is None:
            message = "NormalCrafter runtime is not loaded; call setup() first"
            raise RuntimeError(message)

        expected_start = 0
        expected_height, expected_width = frames.shape[1:3]
        for raw_chunk in runtime.infer(frames):
            if raw_chunk.frame_start != expected_start:
                message = (
                    "NormalCrafter runtime chunks must be contiguous: "
                    f"expected start={expected_start}, observed={raw_chunk.frame_start}"
                )
                raise ValueError(message)
            chunk_frames = raw_chunk.values.shape[0]
            expected_chunk_frames = min(
                NORMALCRAFTER_VAE_CHUNK_SIZE,
                frames.shape[0] - expected_start,
            )
            if chunk_frames != expected_chunk_frames:
                message = (
                    "NormalCrafter runtime chunk length mismatch: "
                    f"expected={expected_chunk_frames}, observed={chunk_frames}"
                )
                raise ValueError(message)
            normal, valid = _canonicalize_normals(
                raw_chunk.values,
                expected_height=expected_height,
                expected_width=expected_width,
                epsilon=_NORMAL_EPSILON,
            )
            yield NormalCrafterChunk(
                frame_start=raw_chunk.frame_start,
                normal=normal,
                valid=valid,
            )
            expected_start = raw_chunk.frame_stop

        if expected_start != frames.shape[0]:
            message = f"NormalCrafter runtime produced {expected_start} frames, expected {frames.shape[0]}"
            raise ValueError(message)

    def close(self) -> None:
        """Release the actor-resident runtime."""
        if self._runtime is None:
            return
        self._runtime.close()
        self._runtime = None


class _DiffusersNormalCrafterRuntime:
    """Released latent-window algorithm with chunked encode/decode boundaries."""

    def __init__(self, pipe: Any, device: Any) -> None:  # noqa: ANN401
        self._pipe = pipe
        self._device = device

    @classmethod
    def load(
        cls,
        checkpoint: Path,
    ) -> "_DiffusersNormalCrafterRuntime":
        """Load the vendored inference-only UNet and pinned Diffusers pipeline."""
        import torch  # noqa: PLC0415
        from diffusers import (  # noqa: PLC0415
            AutoencoderKLTemporalDecoder,
            StableVideoDiffusionPipeline,
        )

        if not checkpoint.is_dir():
            message = f"NormalCrafter checkpoint directory does not exist: {checkpoint}"
            raise FileNotFoundError(message)
        execution_device = torch.device("cuda")
        if execution_device.type != "cuda" or not torch.cuda.is_available():
            message = "NormalCrafter requires an available CUDA device"
            raise RuntimeError(message)

        unet_class = _load_vendored_unet_class()
        unet = unet_class.from_pretrained(
            checkpoint,
            subfolder="unet",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(  # type: ignore[no-untyped-call]
            checkpoint,
            subfolder="vae",
            local_files_only=True,
        )
        unet.to(dtype=torch.float16)
        vae.to(dtype=torch.float16)
        pipe = StableVideoDiffusionPipeline.from_pretrained(  # type: ignore[no-untyped-call]
            checkpoint,
            unet=unet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
        )
        pipe.to(execution_device)
        pipe.set_progress_bar_config(disable=True)
        _configure_sdpa(pipe)
        return cls(pipe, execution_device)

    def infer(
        self,
        frames: npt.NDArray[np.uint8],
    ) -> Iterator[NormalCrafterRawChunk]:
        """Run latent overlap inference and decode only one VAE chunk at a time."""
        import torch  # noqa: PLC0415

        pipe = self._pipe
        if pipe is None:
            message = "NormalCrafter runtime is closed"
            raise RuntimeError(message)

        padding = _padding_to_multiple(
            frames.shape[1],
            frames.shape[2],
            NORMALCRAFTER_PADDING_MULTIPLE,
        )
        padded_height = frames.shape[1] + padding[0] + padding[1]
        padded_width = frames.shape[2] + padding[2] + padding[3]

        with torch.inference_mode():
            image_embeddings = self._encode_clip_frames(frames, padding)
            image_latents = self._encode_vae_frames(
                frames,
                padding,
                chunk_size=NORMALCRAFTER_VAE_CHUNK_SIZE,
            )
            image_latents = image_latents.to(image_embeddings.dtype).unsqueeze(0)
            added_time_ids = pipe._get_add_time_ids(  # noqa: SLF001
                NORMALCRAFTER_CONDITIONING_FPS,
                _MOTION_BUCKET_ID,
                _NOISE_AUG_STRENGTH,
                image_embeddings.dtype,
                1,
                1,
                do_classifier_free_guidance=False,
            ).to(self._device)

            prediction = self._predict_latents(
                image_embeddings=image_embeddings,
                image_latents=image_latents,
                added_time_ids=added_time_ids,
                frame_count=frames.shape[0],
                height=padded_height,
                width=padded_width,
            )
            try:
                yield from self._decode_chunks(
                    prediction,
                    frame_count=frames.shape[0],
                    padding=padding,
                    chunk_size=NORMALCRAFTER_VAE_CHUNK_SIZE,
                )
            finally:
                del prediction, image_latents, image_embeddings
                pipe.maybe_free_model_hooks()

    def close(self) -> None:
        """Release model references and cached CUDA allocations."""
        import torch  # noqa: PLC0415

        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode_clip_frames(
        self,
        frames: npt.NDArray[np.uint8],
        padding: tuple[int, int, int, int],
    ) -> Any:  # noqa: ANN401
        import torch  # noqa: PLC0415

        embeddings = []
        for start in range(0, frames.shape[0], _CLIP_ENCODE_CHUNK_SIZE):
            stop = min(start + _CLIP_ENCODE_CHUNK_SIZE, frames.shape[0])
            images = _make_pil_frames(frames[start:stop], padding)
            embeddings.append(
                self._pipe._encode_image(  # noqa: SLF001
                    images,
                    self._device,
                    1,
                    do_classifier_free_guidance=False,
                )
            )
        return torch.cat(embeddings)

    def _encode_vae_frames(
        self,
        frames: npt.NDArray[np.uint8],
        padding: tuple[int, int, int, int],
        *,
        chunk_size: int,
    ) -> Any:  # noqa: ANN401
        import torch  # noqa: PLC0415

        pipe = self._pipe
        vae = pipe.vae
        restore_fp16 = vae.dtype == torch.float16 and bool(vae.config.force_upcast)
        if restore_fp16:
            vae.to(dtype=torch.float32)
        latents = []
        try:
            for start in range(0, frames.shape[0], chunk_size):
                stop = min(start + chunk_size, frames.shape[0])
                images = _make_pil_frames(frames[start:stop], padding)
                video = pipe.video_processor.preprocess_video(images)
                vae_input = video[0].permute(1, 0, 2, 3).to(dtype=vae.dtype)
                latents.append(
                    pipe._encode_vae_image(  # noqa: SLF001
                        vae_input,
                        self._device,
                        1,
                        do_classifier_free_guidance=False,
                    )
                )
        finally:
            if restore_fp16:
                vae.to(dtype=torch.float16)
        return torch.cat(latents)

    def _predict_latents(  # noqa: PLR0913
        self,
        *,
        image_embeddings: Any,  # noqa: ANN401
        image_latents: Any,  # noqa: ANN401
        added_time_ids: Any,  # noqa: ANN401
        frame_count: int,
        height: int,
        width: int,
    ) -> Any:  # noqa: ANN401
        import torch  # noqa: PLC0415

        prediction: Any = None
        previous_stop: int | None = None
        for start, stop in _temporal_windows(
            frame_count,
            window_size=NORMALCRAFTER_WINDOW_SIZE,
            stride=NORMALCRAFTER_WINDOW_STRIDE,
        ):
            replacement: Any = None
            if previous_stop is not None:
                assert prediction is not None
                replacement = prediction[:, start:previous_stop]
            window = self._generate_window(
                height=height,
                width=width,
                image_embeddings=image_embeddings[start:stop],
                image_latents=image_latents[:, start:stop],
                added_time_ids=added_time_ids,
                replacement=replacement,
            )
            if replacement is not None:
                overlap = replacement.shape[1]
                weights = torch.linspace(
                    1.0,
                    0.0,
                    overlap + 2,
                    device="cpu",
                    dtype=torch.float32,
                )[1:-1].to(self._device)
                weights = weights[None, :, None, None, None]
                window[:, :overlap] = replacement * weights + window[:, :overlap] * (1.0 - weights)
            if prediction is None:
                prediction = window.new_empty((window.shape[0], frame_count, *window.shape[2:]))
            prediction[:, start:stop].copy_(window)
            previous_stop = stop

        if prediction is None:
            message = "NormalCrafter produced no temporal windows"
            raise RuntimeError(message)
        return prediction

    def _generate_window(  # noqa: PLR0913
        self,
        *,
        height: int,
        width: int,
        image_embeddings: Any,  # noqa: ANN401
        image_latents: Any,  # noqa: ANN401
        added_time_ids: Any,  # noqa: ANN401
        replacement: Any | None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        import torch  # noqa: PLC0415

        pipe = self._pipe
        pipe.scheduler.set_timesteps(1, device=self._device)
        timestep = pipe.scheduler.timesteps[0]
        latent_shape = _window_latent_shape(
            unet_in_channels=int(pipe.unet.config.in_channels),
            image_latent_channels=int(image_latents.shape[2]),
            height=height,
            width=width,
            vae_scale_factor=int(pipe.vae_scale_factor),
        )
        latents = torch.zeros(
            latent_shape,
            dtype=image_embeddings.dtype,
            device=self._device,
        )
        if replacement is not None:
            latents[:, : replacement.shape[1]].copy_(replacement)
        latent_model_input = pipe.scheduler.scale_model_input(latents, timestep)
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
        noise_prediction = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]
        return pipe.scheduler.step(
            noise_prediction,
            timestep,
            latents,
        ).prev_sample

    def _decode_chunks(
        self,
        prediction: Any,  # noqa: ANN401
        *,
        frame_count: int,
        padding: tuple[int, int, int, int],
        chunk_size: int,
    ) -> Iterator[NormalCrafterRawChunk]:
        pipe = self._pipe
        vae = pipe.vae
        for start in range(0, frame_count, chunk_size):
            stop = min(start + chunk_size, frame_count)
            latents = prediction[:, start:stop] / vae.config.scaling_factor
            decoded = vae.decode(
                latents.flatten(0, 1),
                num_frames=stop - start,
            ).sample
            decoded = decoded.float().clamp_(-1.0, 1.0)
            decoded = _unpad_tensor(decoded, padding)
            values = decoded.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32, copy=False)
            yield NormalCrafterRawChunk(frame_start=start, values=values)


def _load_vendored_unet_class() -> type[Any]:
    from cosmos_curator.models._normalcrafter import (  # noqa: PLC0415
        DiffusersUNetSpatioTemporalConditionModelNormalCrafter,
    )

    return DiffusersUNetSpatioTemporalConditionModelNormalCrafter


def _configure_sdpa(pipe: Any) -> None:  # noqa: ANN401
    from diffusers.models.attention_processor import AttnProcessor2_0  # noqa: PLC0415

    for name, component in (("unet", pipe.unet), ("vae", pipe.vae)):
        setter = getattr(component, "set_attn_processor", None)
        processors = getattr(component, "attn_processors", None)
        if not callable(setter) or not processors:
            message = f"NormalCrafter {name} does not expose attention processors"
            raise TypeError(message)
        setter(AttnProcessor2_0())  # type: ignore[no-untyped-call]
        if not all(isinstance(processor, AttnProcessor2_0) for processor in component.attn_processors.values()):
            message = f"NormalCrafter {name} did not install SDPA on every attention block"
            raise RuntimeError(message)

    image_setter = getattr(pipe.image_encoder, "set_attn_implementation", None)
    if not callable(image_setter):
        message = "NormalCrafter image encoder cannot select SDPA"
        raise TypeError(message)
    image_setter("sdpa")
    implementation = getattr(pipe.image_encoder.config, "_attn_implementation", None)
    if implementation != "sdpa":
        message = f"NormalCrafter image encoder rejected SDPA: effective={implementation!r}"
        raise RuntimeError(message)


def _validate_input_frames(
    frames: npt.NDArray[np.uint8],
    *,
    max_frames: int,
) -> None:
    if frames.dtype != np.uint8:
        message = f"NormalCrafter frames must be uint8 RGB, got {frames.dtype}"
        raise TypeError(message)
    if frames.ndim != _VIDEO_NDIM or frames.shape[-1] != _RGB_CHANNELS:
        message = f"NormalCrafter frames must have THWC RGB shape, got {frames.shape}"
        raise ValueError(message)
    if frames.shape[0] < NORMALCRAFTER_WINDOW_SIZE:
        message = f"NormalCrafter requires at least {NORMALCRAFTER_WINDOW_SIZE} frames, got {frames.shape[0]}"
        raise ValueError(message)
    if frames.shape[0] > max_frames:
        message = (
            f"NormalCrafter input has {frames.shape[0]} frames, exceeding the "
            f"explicit max_frames={max_frames} safety limit"
        )
        raise ValueError(message)
    if frames.shape[1] <= 0 or frames.shape[2] <= 0:
        message = f"NormalCrafter frame dimensions must be positive, got {frames.shape[1:3]}"
        raise ValueError(message)


def _canonicalize_normals(
    values: npt.NDArray[np.float32],
    *,
    expected_height: int,
    expected_width: int,
    epsilon: float,
) -> tuple[npt.NDArray[np.float16], npt.NDArray[np.bool_]]:
    if not isinstance(values, np.ndarray) or not np.issubdtype(values.dtype, np.floating):
        message = "NormalCrafter runtime output must be a floating-point NumPy array"
        raise TypeError(message)
    if values.ndim != _VIDEO_NDIM or values.shape[1:] != (
        expected_height,
        expected_width,
        _RGB_CHANNELS,
    ):
        message = (
            "NormalCrafter runtime output must have THWC shape "
            f"(*,{expected_height},{expected_width},3), got {values.shape}"
        )
        raise ValueError(message)

    canonical = values.astype(np.float32, copy=True)
    canonical[..., 0] *= -1.0
    finite = np.isfinite(canonical).all(axis=-1)
    lengths = np.linalg.norm(canonical, axis=-1)
    valid = finite & (lengths > epsilon)
    normalized = np.zeros_like(canonical)
    np.divide(
        canonical,
        lengths[..., None],
        out=normalized,
        where=valid[..., None],
    )
    return (
        np.ascontiguousarray(normalized.astype(np.float16)),
        np.ascontiguousarray(valid),
    )


def _temporal_windows(
    frame_count: int,
    *,
    window_size: int = NORMALCRAFTER_WINDOW_SIZE,
    stride: int = NORMALCRAFTER_WINDOW_STRIDE,
) -> tuple[tuple[int, int], ...]:
    if frame_count < window_size:
        message = f"frame_count must be at least window_size={window_size}"
        raise ValueError(message)
    windows = [
        (start, start + window_size) for start in range(0, frame_count, stride) if start + window_size <= frame_count
    ]
    if windows[-1][1] < frame_count:
        windows.append((frame_count - window_size, frame_count))
    return tuple(windows)


def _window_latent_shape(
    *,
    unet_in_channels: int,
    image_latent_channels: int,
    height: int,
    width: int,
    vae_scale_factor: int,
) -> tuple[int, int, int, int, int]:
    if image_latent_channels <= 0:
        message = "image_latent_channels must be positive"
        raise ValueError(message)
    if unet_in_channels != image_latent_channels * 2:
        message = (
            "NormalCrafter UNet input channels must equal noise plus image "
            f"latent channels: unet={unet_in_channels}, image={image_latent_channels}"
        )
        raise ValueError(message)
    if vae_scale_factor <= 0 or height % vae_scale_factor or width % vae_scale_factor:
        message = (
            "padded dimensions must be divisible by vae_scale_factor: "
            f"height={height}, width={width}, factor={vae_scale_factor}"
        )
        raise ValueError(message)
    return (
        1,
        NORMALCRAFTER_WINDOW_SIZE,
        image_latent_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )


def _padding_to_multiple(
    height: int,
    width: int,
    multiple: int,
) -> tuple[int, int, int, int]:
    padded_height = math.ceil(height / multiple) * multiple
    padded_width = math.ceil(width / multiple) * multiple
    height_delta = padded_height - height
    width_delta = padded_width - width
    top = height_delta // 2
    left = width_delta // 2
    return top, height_delta - top, left, width_delta - left


def _make_pil_frames(
    frames: npt.NDArray[np.uint8],
    padding: tuple[int, int, int, int],
) -> list[Any]:
    from PIL import Image, ImageOps  # noqa: PLC0415

    top, bottom, left, right = padding
    images = []
    for frame in frames:
        image = Image.fromarray(frame, mode="RGB")
        if any(padding):
            image = ImageOps.expand(
                image,
                border=(left, top, right, bottom),
                fill=(255, 255, 255),
            )
        images.append(image)
    return images


def _unpad_tensor(
    values: Any,  # noqa: ANN401
    padding: tuple[int, int, int, int],
) -> Any:  # noqa: ANN401
    if not any(padding):
        return values
    top, bottom, left, right = padding
    height_stop = values.shape[-2] - bottom
    width_stop = values.shape[-1] - right
    return values[..., top:height_stop, left:width_stop]


__all__ = [
    "NORMALCRAFTER_CONDITIONING_FPS",
    "NORMALCRAFTER_MAX_FRAMES",
    "NORMALCRAFTER_MODEL_ID",
    "NORMALCRAFTER_PADDING_MULTIPLE",
    "NORMALCRAFTER_VAE_CHUNK_SIZE",
    "NORMALCRAFTER_WINDOW_SIZE",
    "NORMALCRAFTER_WINDOW_STRIDE",
    "NormalCrafterChunk",
    "NormalCrafterModel",
    "NormalCrafterRawChunk",
    "NormalCrafterRuntime",
]
