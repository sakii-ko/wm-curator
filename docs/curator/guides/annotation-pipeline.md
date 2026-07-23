# Source-backed video annotation

The lightweight annotation path is for datasets that should keep their original
video files while producing sidecar outputs such as captions, camera/depth
estimates, and normal videos. It deliberately uses the normal Cosmos Curator
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

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.video.annotation.adapters import FilesystemDatasetAdapter

adapter = FilesystemDatasetAdapter(
    Path("/shared/datasets/raw"),
    dataset_metadata={"dataset": "my-dataset", "split": "train"},
)

# Use the actual stage constructors/builders enabled in this checkout.
stages: list[CuratorStage | CuratorStageSpec] = [
    # Caption preparation/caption stages,
    # ViPE camera/depth stages,
    # NormalCrafter stages,
    # artifact writer stages,
]

tasks = adapter.discover()
output_tasks = run_pipeline(tasks, stages)
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
| `metadata` | no | Dataset-specific JSON object, merged over adapter-level metadata |

This JSONL is only an optional input list. It is not a dataset manifest,
catalog, cache index, or provenance database, and it contains no content hash
or checksum. A future Parquet reader can convert each row to a mapping and call
`annotation_task_from_mapping()` without introducing another record model.

The adapter preserves hints but cannot force a downstream decoder to honor
them. A stage that supports multiple streams or rotation must read
`AnnotationTask.stream_index` and
`AnnotationTask.rotation_degrees_clockwise` explicitly.

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

## Use a 4090 as an optional media-prep worker

Cosmos Curator does not provide an SSH file uploader or an SSH job runner. Treat
shared storage as the handoff between independently launched jobs:

```text
raw videos on shared filesystem/S3
        |
        v
4090 job: ingest + TransNetV2 + optional NVENC
        |
        v
source-span metadata or materialized clips on shared storage
        |
        v
separate Cosmos/Ray annotation job: caption + ViPE + NormalCrafter
```

Both jobs may run asynchronously, but scheduling, retries across hosts, and SSH
transport remain outside this pipeline. The next job should start only after
the relevant output partition is complete. With source-backed clips, every
worker must be able to resolve the same source paths; a path local only to the
4090 host is not a valid cross-machine handoff.

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
span. It does not upload materialized MP4 clips. Until an output-metadata
adapter is added, feeding those spans into the lightweight annotation adapter
requires exporting the desired rows to the JSONL input-list shape above.

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

The split pipeline's JSON input option and the annotation adapter's JSONL input
list are separate formats. Neither one is a checksum-bearing dataset catalog.
