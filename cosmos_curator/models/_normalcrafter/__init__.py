"""Inference-only model surface derived from the NormalCrafter release."""

from cosmos_curator.models._normalcrafter.unet import (
    DiffusersUNetSpatioTemporalConditionModelNormalCrafter,
)

__all__ = ["DiffusersUNetSpatioTemporalConditionModelNormalCrafter"]
