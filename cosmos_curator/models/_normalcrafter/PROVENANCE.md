# NormalCrafter inference-only UNet

`unet.py` is a reduced derivative of
[Binyr/NormalCrafter](https://github.com/Binyr/NormalCrafter) at commit
`75af9887a2cb14cd1ce3883c5773bc296565777c`.

Only the inference-time SVD UNet forward override is retained. Training,
gradient checkpointing, DINO features, ControlNet features, the upstream
pipeline wrapper, configuration layers, checkpoint locks, hashing, and
compile-cache machinery are intentionally excluded. Cosmos Curator owns
frame decoding, temporal window composition, VAE chunking, canonicalization,
and artifact output.

The upstream source is MIT licensed. Its license is preserved in `LICENSE`.
