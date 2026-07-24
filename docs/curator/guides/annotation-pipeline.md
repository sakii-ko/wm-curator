# Source-backed video annotation

The lightweight annotation path is for datasets that should keep their original
video files while producing sidecar outputs such as captions, camera/depth
estimates, and normal tensor estimates. It deliberately uses the normal Cosmos Curator
pipeline contract:

1. an input adapter creates `AnnotationTask` objects;
2. a normal `list[CuratorStage | CuratorStageSpec]` selects the work;
3. `run_pipeline()` executes that list.

There is no separate annotation CLI or configuration hierarchy. In particular,
the input layer does not decode media, probe durations, copy files, or calculate
SHA/checksum values.

## Choose an input adapter

Use recursive filesystem discovery when the directory hierarchy is already the
dataset identity:

```python
from pathlib import Path

from cosmos_curator.pipelines.video.annotation.adapters import FilesystemDatasetAdapter

adapter = FilesystemDatasetAdapter(
    Path("/shared/datasets/raw"),
    dataset_metadata={"dataset": "my-dataset", "split": "train"},
)
tasks = adapter.discover()
```

`FilesystemDatasetAdapter` walks recursively, filters common video extensions,
sorts by relative POSIX path, and stores a resolved absolute source path. It
does not inspect codecs. Actual container/codec support is therefore determined
by the decoder used by the selected annotation stage.

Use a JSONL input list when samples need explicit IDs, spans, decode hints, or
metadata:

```json
{"path":"camera/front.mp4","id":"scene-0001-front","stream_index":0,"rotation_degrees_clockwise":90,"span":[12.5,18.0],"metadata":{"camera":"front"}}
{"path":"s3://my-bucket/raw/scene-0002.mkv","id":"scene-0002","relative_path":"scene-0002/main.mkv"}
```

```python
from pathlib import Path

from cosmos_curator.pipelines.video.annotation.adapters import JsonlDatasetAdapter

adapter = JsonlDatasetAdapter(
    Path("/shared/datasets/annotation-inputs.jsonl"),
    source_root=Path("/shared/datasets/raw"),
    dataset_metadata={"dataset": "my-dataset", "split": "train"},
)
tasks = adapter.discover()
```

The supported row fields are:

| Field | Required | Meaning |
| --- | --- | --- |
| `path` | yes | Local path relative to `source_root`, absolute local path, or supported cloud URI |
| `id` | no | Stable task/session ID; defaults to the normalized relative path |
| `relative_path` | no | Dataset-relative output identity |
| `stream_index` | no | Zero-based video-stream hint |
| `rotation_degrees_clockwise` | no | Right-angle decode hint, normalized modulo 360 |
| `span` | no | `[start_seconds, end_seconds]`, represented by the existing `Clip` type |
| `clip_uuid` / `span_uuid` | no | Existing Cosmos clip UUID; if both are present they must match |
| `metadata` | no | Dataset-specific JSON object, merged over adapter-level metadata |

This JSONL is only an optional input list. It is not a dataset manifest,
catalog, cache index, or provenance database, and it contains no content hash
or checksum. Give rows distinct `id` values when one source path has multiple
spans; the default ID is the relative source path and duplicates are rejected.

The adapter preserves hints but cannot force a downstream decoder to honor
them. A stage that supports multiple streams or rotation must read
`AnnotationTask.stream_index` and
`AnnotationTask.rotation_degrees_clockwise` explicitly.

Use the existing Cosmos split output directly when TransNetV2 has already
created source-backed clips:

```python
from pathlib import Path

from cosmos_curator.pipelines.video.annotation.adapters import SourceSpanDatasetAdapter

tasks = SourceSpanDatasetAdapter(
    Path("/shared/datasets/media-prep-output"),
    dataset_metadata={"dataset": "my-dataset", "split": "train"},
).discover()
```

`SourceSpanDatasetAdapter` reads the normal `summary.json` and
`metas/v0/<clip_uuid>.json` files emitted by the split pipeline. Invalid clips
are skipped and valid clips retain the Cosmos UUID, source path, and time span.
It does not create another manifest or hash the media.

## Whole videos and source spans

An omitted `span` means “no preselected clip”; the filesystem adapter therefore
creates a `Video` with an empty `clips` list. This is suitable for stages that
operate directly on a whole source file. Clip-oriented stages need one of:

- a JSONL row with an explicit span;
- a preceding stage that probes/splits the source and creates `Clip` objects;
- the existing split pipeline described below.

The adapters intentionally do not probe duration just to manufacture a
whole-file span. That keeps discovery cheap and prevents a second, hidden decode
pass. When a span is present, the task still points to the original source; no
clip bytes are materialized by the adapter.

## Run ViPE and NormalCrafter

Both geometry stages are ordinary `CuratorStage` implementations, so one
pipeline can run either or both:

```python
from pathlib import Path

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.models.normalcrafter import NormalCrafterModel
from cosmos_curator.models.vipe import ViPEModel, ViPEModelConfig
from cosmos_curator.pipelines.video.annotation.normalcrafter_stage import NormalCrafterStage
from cosmos_curator.pipelines.video.annotation.vipe_stage import ViPEStage

annotation_root = Path("/shared/datasets/annotations")

vipe_model = ViPEModel(
    ViPEModelConfig(
        slam_model_path=Path("/shared/models/DA3NESTED-GIANT-LARGE-1.1"),
        post_model_path=Path("/shared/models/DA3-GIANT-1.1"),
        torch_home=Path("/shared/models/vipe/torch_home"),
    )
)
normal_model = NormalCrafterModel(
    checkpoint_path=Path("/shared/models/normalcrafter"),
)

stages = [
    CuratorStageSpec(
        ViPEStage(annotation_root / "vipe-dav3", vipe_model),
        num_workers_per_node=1,
    ),
    CuratorStageSpec(
        NormalCrafterStage(annotation_root / "normalcrafter", normal_model),
        num_workers_per_node=1,
    ),
]

run_pipeline(tasks, stages)
```

ViPE deliberately takes explicit local SLAM and post-processing weights; there
is no second model catalog or checksum layer. NormalCrafter can instead use the
standard Cosmos model registry entry `normalcrafter`:

```python
normal_model = NormalCrafterModel()
```

When the registry path is used, `run_pipeline()` resolves and synchronizes the
checkpoint on each node through the normal Cosmos model lifecycle before stage
actors start. It can also be staged ahead of time:

```bash
pixi run --as-is model-download --models normalcrafter
```

For the existing shared-storage deployment, the already populated paths are:

```python
vipe_model = ViPEModel(
    ViPEModelConfig(
        slam_model_path=Path("/root/nas/bigdata1/huggingface/DA3NESTED-GIANT-LARGE-1.1"),
        post_model_path=Path("/root/nas/bigdata1/huggingface/DA3-GIANT-1.1"),
        torch_home=Path("/root/nas/bigdata1/cjw/checkpoints/ViPE/torch_home"),
    )
)
normal_model = NormalCrafterModel(
    checkpoint_path=Path("/root/nas/bigdata1/huggingface/NormalCrafter"),
)
```

An 80 GiB H100 can run two measured ViPE workers by setting
`ViPEStage(..., gpus_per_worker=0.5)` and
`CuratorStageSpec(..., num_workers_per_node=2)`. Keep the default one GPU per
worker on a 24 GiB 4090.

The input videos must currently be local `Path` objects visible at the same
path on every geometry worker. The annotation output may be local, S3, or
Azure. Both stages buffer one complete clip before inference. ViPE accepts at
least 8 native-rate frames and defaults to a 2,048-frame limit; NormalCrafter
accepts at least 14 frames sampled at 15 FPS and defaults to a 1,350-frame
limit. An out-of-range clip fails rather than being truncated, so split long
videos upstream. Override the bounds with `ViPEStage(..., max_frames=...)` and
`NormalCrafterModel(..., max_frames=...)` when appropriate.

These limits currently bound only frame count, not `height * width`. Native 4K
clips can still require tens of GiB when buffered as a complete clip. Use short
source spans for now; a shared decode-time annotation grid is intentionally
left as a separate change because it must also record the exact source-to-grid
raster transform.

Each stage writes retry-safe NPZ chunks below `chunks/v1/<clip_uuid>/` and publishes
`metas/v1/<clip_uuid>.json` last as the completion record. ViPE chunks contain
`depth`, `valid`, `K`, `camera_to_world`, and `timestamps_ns`; NormalCrafter
chunks contain `normal`, `valid`, and `timestamps_ns`. The two estimators use
different frame schedules, so consumers should align them by
`timestamps_ns`, not by array index. Their raster sizes may also differ; do not
combine pixels without an explicit spatial transform.

These are tensor sidecars, not encoded depth or normal MP4 files. The geometry
roots do not write another `summary.json` or chunk manifest. The retained
Cosmos UUID joins split metadata to both annotation roots:

```text
media-prep-output/
  summary.json
  metas/v0/<clip_uuid>.json
annotations/
  vipe-dav3/{chunks/v1/<clip_uuid>/..., metas/v1/<clip_uuid>.json}
  normalcrafter/{chunks/v1/<clip_uuid>/..., metas/v1/<clip_uuid>.json}
```

On retry, a valid metadata completion record skips decode and inference.
Residual chunks without metadata are not considered complete; that clip is
rerun from the beginning and the same paths are atomically overwritten.

Training code can stream completed tensors one chunk at a time:

```python
from cosmos_curator.pipelines.video.annotation.artifact_writer import TemporalAnnotationReader

reader = TemporalAnnotationReader(annotation_root / "vipe-dav3")
if reader.is_complete(clip_uuid):
    for chunk in reader.iter_chunks(clip_uuid):
        depth = chunk.arrays["depth"]
        camera_to_world = chunk.arrays["camera_to_world"]
```

The reader validates the metadata schema plus chunk dtypes, shapes, and
timestamps. It derives chunk paths from `frame_count` and `chunk_frames`; it
does not create a manifest, checksum, or catalog.

The stages decode independently on purpose. Passing decoded frame tensors
between Ray stages would retain large buffers in the object store and would
still not remove NormalCrafter's 15 FPS resampling.

## Use Qwen3.6 35B A3B FP8 for captions

`qwen3_6_35b_a3b_fp8` is a regular vLLM caption backend in the existing split
pipeline. It uses model ID `Qwen/Qwen3.6-35B-A3B-FP8`, with thinking disabled
by default. The model is opt-in because of its size:

```bash
pixi run --as-is model-download --models qwen3_6_35b_a3b_fp8
```

An existing flat checkpoint directory can be used directly with
`--vllm-model-path`; it does not need to be copied or linked into the Cosmos
model-cache layout.

For local shared input, captioning can decode source spans without
materializing transcoded clips:

```bash
cosmos-curator local launch \
    --image-name cosmos-curator \
    --image-tag 1.0.0 \
    --curator-path . \
    -- pixi run --as-is video-pipeline split \
    --input-video-path <shared-input-path> \
    --output-clip-path <shared-caption-output> \
    --splitting-algorithm transnetv2 \
    --transcode-encoder none \
    --captioning-algorithm qwen3_6_35b_a3b_fp8 \
    --vllm-model-path /root/nas/bigdata1/huggingface/Qwen3.6-35B-A3B-FP8 \
    --vllm-safetensors-load-strategy prefetch \
    --captioning-prompt-variant world-model \
    --captioning-sampling-fps 4 \
    --captioning-window-size 96 \
    --captioning-remainder-threshold 2 \
    --captioning-max-output-tokens 768 \
    --qwen-batch-size 4 \
    --vllm-video-max-pixels-per-frame 262144 \
    --vllm-sampling-temperature 0.1 \
    --vllm-sampling-top-p 1.0 \
    --vllm-sampling-repetition-penalty 1.0 \
    --vllm-sampling-min-tokens 0 \
    --no-generate-embeddings \
    --no-upload-clips
```

The prefetch strategy is opt-in because it is a storage policy, not a model
requirement. The defaults of 16 reader threads and 16 MiB blocks match the
measured CEPH deployment; omit the strategy on local SSD.

vLLM 0.21 performs a broad DeepGEMM kernel warmup on first startup. For
short-lived workers, prefix the launch command with
`VLLM_DEEP_GEMM_WARMUP=skip`; this moves compilation for actually used shapes
to the first request instead of compiling every candidate shape up front. Keep
vLLM's default for long-lived throughput workers when predictable first-request
latency matters more than startup time.

Each source-backed caption preparation task decodes all of its windows in one
pass and scatters the frames to those windows. The current source-backed decoder
accepts a shared local `Path`, not a remote URI. Captions are stored in
`<shared-caption-output>/metas/v0/<clip_uuid>.json` under each window's
`qwen3_6_35b_a3b_fp8_caption` field; `summary.json` remains the index and no MP4
is generated.

## Use a 4090 as an optional media-prep worker

Cosmos Curator does not provide an SSH file uploader or an SSH job runner. Treat
shared storage as the handoff between independently launched jobs:

```text
videos on a shared local filesystem
        |
        +--> 4090 job: TransNetV2 + optional NVENC
        |          |
        |          v
        |    source spans or clips on shared storage
        |          |
        |          v
        |    ViPE / NormalCrafter jobs on geometry GPUs
        |
        +--> caption GPU job: TransNetV2 + Qwen, no transcode
                   |
                   v
              caption metadata
```

These jobs may run asynchronously, but scheduling, retries across hosts, and SSH
transport remain outside this pipeline. The next job should start only after
the relevant output partition is complete. With source-backed clips, every
worker must be able to resolve the same source paths; a path local only to the
4090 host is not a valid cross-machine handoff.

The current split CLI does not take `SourceSpanDatasetAdapter` as an input, so
the caption branch above runs splitting again over the shared source videos.
`SourceSpanDatasetAdapter` is the direct handoff for ViPE and NormalCrafter.
This keeps the CLI change small and avoids pretending there is a generic remote
job-orchestration layer.

Run TransNetV2 on the 4090 without transcoding:

```bash
cosmos-curator local launch \
    --image-name cosmos-curator \
    --image-tag 1.0.0 \
    --curator-path . \
    -- pixi run --as-is video-pipeline split \
    --input-video-path <shared-input-path> \
    --output-clip-path <shared-media-prep-output> \
    --splitting-algorithm transnetv2 \
    --transcode-encoder none \
    --no-generate-captions \
    --no-generate-embeddings \
    --no-upload-clips
```

This emits clip metadata whose identity is the original source path plus time
span. It does not upload materialized MP4 clips. Point
`SourceSpanDatasetAdapter` at `<shared-media-prep-output>` to feed those spans
directly into ViPE and NormalCrafter. This adapter accepts only
`clip_format=source_span` records.

If standardized clip files are worth the storage and transfer cost, change the
same job to NVENC:

```bash
cosmos-curator local launch \
    --image-name cosmos-curator \
    --image-tag 1.0.0 \
    --curator-path . \
    -- pixi run --as-is video-pipeline split \
    --input-video-path <shared-input-path> \
    --output-clip-path <shared-media-prep-output> \
    --splitting-algorithm transnetv2 \
    --transcode-encoder h264_nvenc \
    --no-generate-captions \
    --no-generate-embeddings
```

Add `--transcode-use-hwaccel` only when the installed FFmpeg build and the input
codec support CUDA decoding. `h264_nvenc` selects hardware encoding by itself.
Keeping decode on CPU is often the more compatible first test for mixed-format
datasets.

NVENC output contains materialized clips rather than source-span records. Scan
that clip directory with `FilesystemDatasetAdapter`; do not pass it to
`SourceSpanDatasetAdapter`.

The split pipeline's JSON input option and the annotation adapter's JSONL input
list are separate formats. Neither one is a checksum-bearing dataset catalog.
