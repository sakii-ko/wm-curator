"""Minimal NormalCrafter UNet inference override.

This module is derived from NormalCrafter commit
``75af9887a2cb14cd1ce3883c5773bc296565777c``. The upstream project is MIT
licensed; see ``LICENSE`` and ``PROVENANCE.md`` in this directory.

The released checkpoint conditions every video frame on its own CLIP
embedding. Diffusers' base SVD UNet repeats one embedding across all frames;
the only semantic difference retained here is accepting an already
frame-expanded ``encoder_hidden_states`` tensor.
"""

# Diffusers builds these registered modules dynamically from the checkpoint
# config, so its static class surface does not expose them to mypy.
# mypy: disable-error-code="attr-defined,import-not-found,misc"

import torch
from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionOutput,
)


class DiffusersUNetSpatioTemporalConditionModelNormalCrafter(UNetSpatioTemporalConditionModel):
    """SVD UNet whose cross-attention context may vary per video frame."""

    def forward(  # noqa: C901, PLR0912
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float,
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,  # noqa: FBT001, FBT002
    ) -> UNetSpatioTemporalConditionOutput | tuple[torch.Tensor]:
        """Run the released inference path with per-frame conditioning."""
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_reduced_precision_device = sample.device.type in {"mps", "npu"}
            if isinstance(timestep, float):
                dtype = torch.float32 if is_reduced_precision_device else torch.float64
            else:
                dtype = torch.int32 if is_reduced_precision_device else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)
        time_embedding = self.time_proj(timesteps).to(dtype=sample.dtype)
        embedding = self.time_embedding(time_embedding)

        added_time_embedding = self.add_time_proj(added_time_ids.flatten())
        added_time_embedding = added_time_embedding.reshape((batch_size, -1))
        added_embedding = self.add_embedding(added_time_embedding.to(embedding.dtype))
        embedding = (embedding + added_embedding).repeat_interleave(
            num_frames,
            dim=0,
            output_size=batch_size * num_frames,
        )

        sample = sample.flatten(0, 1)
        if encoder_hidden_states.shape[0] == batch_size:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                num_frames,
                dim=0,
                output_size=batch_size * num_frames,
            )
        elif encoder_hidden_states.shape[0] != batch_size * num_frames:
            message = (
                "NormalCrafter encoder_hidden_states must contain either one "
                f"context per batch or per frame, got {encoder_hidden_states.shape[0]} "
                f"for batch={batch_size}, frames={num_frames}"
            )
            raise ValueError(message)

        sample = self.conv_in(sample)
        image_only_indicator = torch.zeros(
            batch_size,
            num_frames,
            dtype=sample.dtype,
            device=sample.device,
        )

        down_block_res_samples: tuple[torch.Tensor, ...] = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=embedding,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=embedding,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res_samples

        sample = self.mid_block(
            hidden_states=sample,
            temb=embedding,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        for upsample_block in self.up_blocks:
            residual_count = len(upsample_block.resnets)
            residuals = down_block_res_samples[-residual_count:]
            down_block_res_samples = down_block_res_samples[:-residual_count]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=embedding,
                    res_hidden_states_tuple=residuals,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=embedding,
                    res_hidden_states_tuple=residuals,
                    image_only_indicator=image_only_indicator,
                )

        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        if not return_dict:
            return (sample,)
        return UNetSpatioTemporalConditionOutput(sample=sample)
