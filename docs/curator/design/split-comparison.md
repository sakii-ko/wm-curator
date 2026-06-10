# Split Output Comparison — v3 design (measure / evaluate)

This document is the **orientation + rationale** for the split-comparison
package. The code is the ground truth for schemas, signatures, and config
fields; this doc captures the "why" decisions that grep'ing the code won't
surface (Ray Data shape, Arrow placement, caption batching, what's
deliberately absent).

## Overview

The comparator audits two output trees from the split video pipeline — the same
input data run through two pipeline configurations (e.g. two model versions, two
encoder settings) — and reports, as a structured report, whether they diverged
in any material way.

**v3 splits the per-clip metadata comparison into two phases:**

- **Measure** — read both output trees once and record the *comparison outcome*
  for every comparable element of the clip metadata. This is the expensive pass
  (BGE caption embedding + metadata JSON I/O).
- **Evaluate** — load what the measure phase recorded and apply thresholds to
  produce the report. Cheap, source-free, re-runnable.

The point: retuning an evaluation parameter (caption `min_similarity`, a score
tolerance) or writing a new analysis tool must not re-run the expensive
measurement pass.

**The litmus test for which phase a thing belongs to:** evaluate must be a pure,
cheap function of `(measurements, thresholds)` — no source I/O, no model load.
Anything that changes the *measured value*, or is expensive to produce, is
measure; anything that merely *thresholds* a recorded value is evaluate.

That test sorts every comparison into two kinds:

- **Decoupled** (per-clip metadata): measure records the measurement, evaluate
  applies a threshold to produce an issue. The retunable case.
- **Coupled** (summary, video index): no separately-retunable knob, so
  measure-and-evaluate are fused — they run during measure and their issues are
  persisted already-evaluated. Retuning a coupled tolerance means re-measuring,
  the accepted cost of coupling. Future versions of this tool may decouple summary
  and video index, like with per-clip metadata. It was decided not to decouple
  those in an effort to limit the scope of the refactor.

Measure emits a **bundle** — a directory of Lance datasets carrying everything evaluate
needs, so evaluate never touches the source:

1. **Measurements table** — the decoupled per-clip metadata measurements
   (`MEASUREMENT_SCHEMA`).
2. **Precomputed coupled issues** — summary + video-index issue tables
   (`ISSUE_SCHEMA`), already evaluated.
3. **Run identity** — `output_a/b`, derived source roots, the `videos` table,
   and clip counts.

(The config knobs split by the same litmus test; see "Config".)

## Data model

`measurement_model.py` defines one **tidy/long** table. Every comparable element
of the clip metadata becomes one row whose `value` is the comparison outcome.

```python
MEASUREMENT_SCHEMA: pa.Schema = pa.schema(
    [
        ("video_key", pa.string()),
        ("clip_id", pa.string()),
        ("window_id", pa.int64()),     # null for clip-level rows; window start_frame otherwise
        ("model", pa.string()),        # null for non-model measurements; caption/token model otherwise
        ("measurement_type", pa.string()),
        ("value", pa.float64()),       # null unless both sides present and neither corrupt
        ("output_a_present", pa.bool_()),
        ("output_b_present", pa.bool_()),
        ("output_a_corrupt", pa.bool_()),
        ("output_b_corrupt", pa.bool_()),
    ],
)
```

Composite key: `(video_key, clip_id, window_id, model, measurement_type)`.

**Value semantics by mode.** The mode is a property of the `measurement_type`
(declared in the field spec):

- `*_diff` (tolerance) → `value` = absolute difference.
- `*_similarity` → `value` = cosine similarity.
- `*_equal` (equality) → `value` = `1.0` (equal) or `0.0` (not equal).

The underlying A/B values are **not** stored — re-read the source metas for
specifics. Distributions stay meaningful: `mean(value)` over an `*_equal` type is
the match rate; `value` over a `*_diff` type is the diff distribution.

**Per-side status — four booleans.** `output_X_present` = the field/JSON exists
on side X; `output_X_corrupt` = it exists but is unusable (wrong type /
unparseable). `corrupt` implies `present` (`present=False, corrupt=True` is an
invalid combo the producer never writes). `value` is non-null **iff**
`a_present and b_present and not a_corrupt and not b_corrupt`; both sides absent ⇒
no row. The four bools give which-side for free and represent both-sides-corrupt
naturally. A clip whose metadata JSON fails to load on a side has no dict to
walk, so measure emits a single clip-level `clip_metadata` row directly instead
of per-field rows. This is the one `measurement_type` with no `field_spec` entry
(it has no field/accessor) — it's produced outside the catalog walk, not by it.

**Field spec.** `field_spec.py` is a catalog — one entry per `measurement_type`
mapping to `(accessor, mode, scope ∈ {clip, window, filtered_window},
model-qualified?)`. Measure walks the catalog, not whatever keys the metadata
dict happens to hold, so:

- fields that legitimately differ are normalized or excluded — `clip_location`
  embeds the output root and is compared only after stripping `output_a` /
  `output_b`; `embedding*` is excluded.
- a span becomes multiple scalar types — `duration_span` →
  `duration_span_start_diff` + `duration_span_end_diff` (float seconds, so a
  tiny tolerance rather than equality).
- the catalog is a closed, stable vocabulary; the `model` column carries the
  caption/token model (not baked into the type name), and `window_id` carries
  per-window identity as `"{start_frame}_{end_frame}"` — so a window whose end
  frame drifts between the two sides no longer aligns and instead surfaces as
  window-set divergence. A `window_present` membership row per window key records
  that divergence even when a window produced no caption.

See `field_spec.py` for the authoritative list (score diffs, caption /
enhanced-caption similarity, count diffs, and the equality set).

## Flow

Measure produces the bundle; evaluate consumes it. A combined `compare` runs
both back-to-back for the one-shot case.

```text
  ── measure ───────────────────────────────────────────────────────────────
  summary_a/b.json ─► compare_summaries ─► summary_issues (coupled)
  clips = discover_clips
        │
        ├─► run_measure_stage      (MeasureStage pool) ─► measurements : pa.Table
        └─► run_video_index_stage  (VideoIndexStage pool) ─► video_index_issues (coupled)
        │
        ▼
  bundle = { measurements, summary_issues + video_index_issues, run identity }  ──► Lance

  ── evaluate ──────────────────────────────────────────────────────────────
  bundle ─► apply thresholds to measurements ─► metadata_issues
         ─► union with precomputed coupled issues ─► Report.issues ──► Lance report
```

The measure stage and the coupled video-index stage run as **two independent Ray
Data pipelines over the same clip table** (see "Execution rationale"). Evaluate is
plain Python over the bundle: no Ray, no source reads, no model.

**Measure stage.** One Ray Data pipeline, one actor pool. Each `MeasureStage`
actor loads the caption embedding model once at construction (skipped when
captions are off) and reuses it across every clip routed to it. On each
`__call__` it reads both outputs' metadata JSON once, then walks the field spec
to emit `MEASUREMENT_SCHEMA` rows — presence/corruption into the four bools, the
outcome into `value`. Structural cases (one-sided, corrupt, clip-load failure)
are encoded in the bools, so no separate issue stream leaves the actor.
Identical-text caption windows record `value = 1.0` without invoking the model
(see "Captions").

**Coupled comparisons.** These run during measure and emit `ISSUE_SCHEMA` issues
directly into the bundle:

- **summary** — `compare_summaries` is pure Python over the two loaded summaries
  (token totals under `summary` tolerances, the rest by equality).
- **video index** — a Ray Data pipeline with a different resource shape (no
  caption model; the indexer does enough CPU work per row that the default
  reserves a full core per actor). Each actor loads both MP4s, builds
  `VideoIndex` + `VideoMetadata`, and emits one issue per divergent field
  (`clip_mp4_index_mismatch`, `clip_mp4_metadata_mismatch`,
  `clip_mp4_index_dtype_mismatch`). Missing / unreadable MP4s map to dedicated
  codes; the comparator never propagates exceptions.

## Execution rationale

**Two independent pipelines, not chained.** The measure stage and video-index
stage run as two separate Ray Data pipelines over the same clip list. Their
resource shapes are unrelated (CPU-heavy with a held caption model vs CPU + IO on
MP4 indexing); pipelining one through the other would force the resources onto
one actor pool or an explicit handoff. They have independent outputs and no
shared state, so re-running just one is "call the function."

**Where Ray Data earns its keep:**

- **Caption model** — `ActorPoolStrategy` loads the model once per actor and
  amortizes it across routed clips; the textbook fit.
- **Per-actor smart_open params** — precomputed at construction and held for the
  actor's life. Actors are stateful; tasks aren't.
- **Spilling** — at millions of clips Ray Data spills to disk; a thread pool
  would fill RAM with results.

It earns nothing on dispatch, plan-variant routing, or sharing loaded data
between features — all eliminated by design.

**Where Arrow lives (and where it doesn't).** Arrow is the format at the
**driver / cross-stage boundary**, not deep inside the per-row comparators:

- `discover_clips` returns `pa.Table`; `ray.data.from_arrow(clips)` keeps blocks
  Arrow in the object store.
- Inside an actor's `__call__`, Ray Data hands a `pa.Table` batch; the actor
  walks rows as plain dicts and dispatches to module-level functions taking plain
  args, which return `make_measurement(...)` rows or `make_issue(...)` issues —
  neither sees `pa.RecordBatch`.
- The actor materializes rows back into a `pa.Table` at the boundary
  (`MEASUREMENT_SCHEMA` for measure, `ISSUE_SCHEMA` for video-index).

The Arrow wins (schema enforcement, columnar groupby/filter, Lance persistence)
live where they matter — at the driver and in the bundle. Inside the actor, dict
I/O keeps each function plain Python: readable, unit-testable without Ray.

## Outputs

**Issue model — generic-by-mode codes.** The decoupled field set is large, so
evaluate uses generic-by-mode codes rather than one per field; the specificity
lives in the issue's `field` (= `measurement_type`) and `details` (value,
threshold), so `ISSUE_SCHEMA` does not change:

- `measurement_out_of_tolerance` — a `*_diff` row over its `abs_tolerance`.
- `measurement_not_equal` — a `*_equal` row with `value == 0.0`.
- `caption_similarity_below_threshold` — a `*_similarity` row under
  `min_similarity`.
- `summary_*` and `clip_mp4_*` — the precomputed coupled issues.

Structural codes are derived from the four bools rather than a mode:

| measurement_type tier        | bools                          | issue code                                          |
|------------------------------|--------------------------------|-----------------------------------------------------|
| field-level                  | one side present, other absent | `metadata_value_one_sided`                          |
| field-level                  | a side corrupt                 | `metadata_value_invalid_type` (one per offending side) |
| `clip_metadata` (clip-level) | one side absent                | `metadata_one_sided`                                |
| `clip_metadata` (clip-level) | present + corrupt              | `metadata_unreadable`                               |

Filtering "all width mismatches" is a filter on `field`, not `code`.
`ISSUE_SCHEMA` is wide — core columns (`code`, `feature`, `video`, `clip`,
`field`, `output`) are queryable without JSON unpacking, plus a JSON-encoded
`details` tail for per-code variant data (detail rows flattened at the source: 8
mismatched index fields → 8 rows). `Issue` is a `TypedDict`; the Arrow table is
canonical; rows are built via `make_issue(...)`.

**Persistence — the Lance bundle.** Measure writes the bundle as a set of Lance
datasets (mirroring the v2 report's sidecar pattern); JSON is not supported for
the bundle or the evaluate report. It is self-describing — the run-identity
sidecar means evaluate (or any downstream analysis) needs no round-trip to
`summary.json`. Lance handles cloud backends natively (`s3://`, `gs://`, `az://`)
via `object_store`; local paths get their parent directory materialized first.

**Videos table and source roots.** Run identity carries a `videos` table plus
per-side `source_a` / `source_b` roots — per-video data kept out of the per-row
tables so a root string isn't repeated across thousands of rows.

```python
VIDEOS_SCHEMA: pa.Schema = pa.schema(
    [
        ("video_key", pa.string()),
        ("in_a", pa.bool_()),
        ("in_b", pa.bool_()),
    ],
)
```

Consumers reconstruct a full source path by **plain concatenation** —
`f"{source_a}{row['video_key']}"` (or `source_b`), with no separator inserted or
stripped. This is exact by construction: `source_a` is each side's
`source_video` with the `video_key` suffix removed, so whatever separator sits
between root and key already lives inside one of the two parts — neither a
trailing separator on `source_a` nor a leading one on `video_key` is required or
assumed (e.g. `source_video = "s3://b/clips/vid.mp4"`, `video_key = "vid.mp4"` ⇒
`source_a = "s3://b/clips/"`). The comparator never asks the user for the source
root; it derives each side by **trust + assert**:

1. Take the first `(video_key, source_video)` entry from `summary.videos`.
2. Strip `video_key` off the end of `source_video` to recover a candidate root.
3. Walk every other entry and assert `root + video_key` reconstructs that row's
   `source_video`.

On success the root ships; on failure the comparator emits a structured
`summary_source_layout_inconsistent` issue, leaves that side's root `""`, and
continues (the videos table still ships). This is a string-shape check, not an IO
existence check — the comparator never opens the source MP4.

## Captions

**Batching strategy.** The caption model runs on CPU. Two batching levers are in
scope; a third is rejected:

1. **Cross-clip batching inside one batch.** `MeasureStage` is invoked via
   `map_batches`; the actor gathers caption-window pairs from every clip in the
   batch and embeds them in one `model.encode(...)` call, capturing
   tokenizer-overhead + framework-dispatch savings across N clips. Batch size is
   `measure_batch_size`; the driver derives block count as
   `ceil(num_rows / batch_size)`, floored at the worker count.
2. **Cross-actor parallelism.** `ActorPoolStrategy(size=N)` runs N independent
   model copies on N CPU workers; embedding parallelizes near-linearly. The
   worker count defaults to `os.cpu_count() // 2` so co-tenant pipelines aren't
   starved.
3. **Per-clip intra-window batching only — not used.** Embedding one clip's few
   windows per call gives up the cross-clip win for no readability benefit; (1)
   subsumes it.

Identical-text windows skip the model entirely (`value = 1.0`), so only divergent
pairs cost an embedding. If caption comparison becomes the bottleneck, the
higher-leverage moves are model-side (smaller / quantized model) and embedding
caches (memoize by `(clip_id, output, prompt_hash)` so re-runs only embed what
changed) rather than rebatching.

**Model.** The caption comparator uses
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) via
`sentence-transformers`:

- ~30M params, 384-dim embeddings — loads fast, meaningful CPU throughput.
- Pre-normalized embeddings: cosine similarity reduces to a dot product.
- English-only — matches the caption pipeline output.
- Needs no instruction prefix for symmetric STS (comparing two caption strings);
  embed both sides as-is.

Per-actor memory ~200 MB; with `os.cpu_count() // 2` workers on a 16-core box
that's ~1.6 GB total — fine for a CI/audit host.

*Registration.* The model needs an entry in
`cosmos_curator/configs/all_models.json` so the standard model-download path
resolves it. Weights are pre-downloaded to the project's local cache:

```bash
cosmos-curator local launch --image-name cosmos-curator -- \
  pixi run --as-is python -m cosmos_curator.core.managers.model_cli download \
  --models bge_small_en_v1_5
```

The actor resolves the cached path via
`model_utils.get_local_dir_for_weights_name(model_id)` and loads with
`local_files_only=True` — deterministic, network-independent runs.

## Config

`config.py` defines two frozen pydantic v2 models, loaded independently:

- **`MeasureConfig`** — `output_a/b`, `profile_name`, filters (`clip_limit`,
  `video_key`), ray tuning (measure + video-index workers/cpus/batch),
  `caption.model_id` + `caption.encode_batch_size`, toggles (`compare_captions`,
  `compare_video_index`), the coupled tolerances (`summary` + `video_index`), and
  `measurements_path` (output bundle).
- **`EvalConfig`** — `measurements_path` (input), `report_path`, and a
  per-measurement-type threshold table: `abs_tolerance` for `*_diff` types,
  `min_similarity` for `*_similarity` types. `*_equal` needs no parameter.

`CaptionPolicy` straddles the split: `model_id` / `encode_batch_size` → measure,
`min_similarity` → eval. The threshold table ships conservative defaults; the
intended workflow is *measure → inspect the distribution summary → set
thresholds*. `framerate_*` and `duration_span_*` default to a tiny float
tolerance.

## Module layout

```text
cosmos_curator/pipelines/video/split_comparison/
  __init__.py              # package marker
  cli.py                   # argparse; measure / evaluate / compare subcommands
  driver.py                # compare_split_outputs, measure_split_outputs, stage runners
  clip_discovery.py        # build the pa.Table of clip rows (CLIP_ROW_SCHEMA)
  summary_compare.py       # A/B summary.json comparison (no Ray; coupled)
  measure_stage.py         # measure actor + per-clip measurement extraction
  metadata_stage.py        # (transitional) v2 issue producer; still the active issue path, to be retired once the evaluate phase lands
  video_index_stage.py     # video index actor + per-comparison helpers (coupled)
  field_spec.py            # measurement catalog: type -> (accessor, mode, scope, ...)
  measurement_model.py     # MEASUREMENT_SCHEMA, MeasurementMode, make_measurement, empty_measurements
  measurement_io.py        # write/load the Lance measurements bundle
  result_model.py          # Issue TypedDict, ISSUE_SCHEMA, make_issue, Report -- issue/report contract
  config.py                # MeasureConfig + EvalConfig (frozen pydantic v2) + nested policies
  summary_schema.py        # pydantic v2 OutputSummary + discriminated union
  summary_loader.py        # load_summary: read summary.json via smart_open
```

Tests mirror the module path under
`tests/cosmos_curator/pipelines/video/split_comparison/`.
