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
"""Stage output comparison helpers."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import attrs
import numpy as np
import numpy.typing as npt
import smart_open  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    CuratorStageSpec,
    PipelineTask,
)
from cosmos_curator.core.utils.environment import MODEL_WEIGHTS_PREFIX
from cosmos_curator.core.utils.misc.stage_replay import (
    DirectStageExecutor,
    PickleTaskSerializer,
    StageExecutor,
    TaskPath,
    TaskSerializer,
)
from cosmos_curator.core.utils.storage import storage_utils
from cosmos_xenna import file_distribution
from cosmos_xenna.pipelines.private.resources import NodeInfo, Resources, WorkerMetadata

type StageCompareBackend = Literal["serial", "xenna"]
type StageCompareStage = CuratorStage | CuratorStageSpec

_IGNORED_COMPARE_FIELD_NAMES = {"stage_perf"}


class TaskComparator(Protocol):
    """Protocol for comparing two pipeline tasks."""

    def checked_fields(self, golden: PipelineTask) -> tuple[str, ...]:
        """Return the leaf field paths this comparator evaluates."""
        ...  # pragma: no cover

    def compare(
        self,
        golden: PipelineTask,
        candidate: PipelineTask,
        *,
        atol: float,
    ) -> list["FieldDiff"]:
        """Compare tasks and return a list of field-level failures."""
        ...  # pragma: no cover


@attrs.define(frozen=True)
class FieldDiff:
    """A single field-level comparison failure."""

    field: str
    detail: str
    max_diff_observed: float | None = None
    shape_mismatch: bool = False


@attrs.define(frozen=True)
class TaskDiff:
    """Comparison result for one task within a batch."""

    batch_file: str
    task_index: int
    checked_fields: tuple[str, ...] = ()
    failures: tuple[FieldDiff, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether the compared task passed."""
        return len(self.failures) == 0


@attrs.define(frozen=True)
class FieldCompareSummary:
    """Aggregated results for one field path."""

    passed: int = 0
    failed: int = 0
    max_diff_observed: float | None = None
    shape_mismatches: int = 0


@attrs.define(frozen=True)
class CompareReport:
    """Aggregated output comparison report."""

    stage: str
    atol: float
    total_batches: int
    passed_batches: int
    failed_batches: int
    pass_rate: float
    fields: dict[str, FieldCompareSummary]
    failures: list[TaskDiff]
    profile_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        return {
            "stage": self.stage,
            "atol": self.atol,
            "total_batches": self.total_batches,
            "passed_batches": self.passed_batches,
            "failed_batches": self.failed_batches,
            "pass_rate": self.pass_rate,
            "fields": {
                name: {
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "max_diff_observed": summary.max_diff_observed,
                    "shape_mismatches": summary.shape_mismatches,
                }
                for name, summary in self.fields.items()
            },
            "failures": [
                {
                    "batch_file": task_diff.batch_file,
                    "task_index": task_diff.task_index,
                    "field": failure.field,
                    "detail": failure.detail,
                }
                for task_diff in self.failures
                for failure in task_diff.failures
            ],
        }

    def write_json(self, path: TaskPath) -> None:
        """Write the report to disk as JSON."""
        if isinstance(path, Path):
            storage_utils.create_path(str(path.parent))
        client = (
            storage_utils.get_storage_client(str(path), profile_name=self.profile_name)
            if self.profile_name is not None
            else storage_utils.get_storage_client(str(path))
        )
        client_params = storage_utils.get_smart_open_client_params(client) if client is not None else {}
        with smart_open.open(str(path), "w", encoding="utf-8", **client_params) as f:
            json.dump(self.to_dict(), f, indent=2)


@attrs.define(frozen=True)
class StageCompareResult:
    """Result of a stage compare run."""

    report: CompareReport
    report_path: TaskPath
    pass_threshold: float

    @property
    def passed(self) -> bool:
        """Return whether the run passed the configured threshold."""
        return self.report.pass_rate >= self.pass_threshold


@attrs.define
class StageCompareBatch(PipelineTask):
    """Task envelope that preserves saved-batch identity through Xenna execution."""

    batch_name: str
    input_path: TaskPath | None = None
    golden_path: TaskPath | None = None
    tasks: list[PipelineTask] | None = None

    def get_major_size(self) -> int:
        """Return aggregate major size for inner tasks."""
        if self.tasks is None:
            return 0
        return sum(task.get_major_size() for task in self.tasks)


@attrs.define
class StageCompareBatchResult(PipelineTask):
    """Small compare result for one saved batch."""

    batch_name: str
    task_diffs: list[TaskDiff]


class StageCompareBatchStage(CuratorStage):
    """Adapt a normal curator stage to process saved stage-compare batches."""

    def __init__(self, stage: CuratorStage, serializer: TaskSerializer) -> None:
        """Store the wrapped stage."""
        self._stage = stage
        self._serializer = serializer

    @property
    def resources(self) -> CuratorStageResource:
        """Forward resource requests to the wrapped stage."""
        return self._stage.resources

    @property
    def required_resources(self) -> Resources:
        """Forward resolved resource requests to the wrapped stage."""
        return self._stage.required_resources

    @property
    def model(self) -> ModelInterface | None:
        """Forward model information to the wrapped stage."""
        return self._stage.model

    @property
    def conda_env_name(self) -> str | None:
        """Forward runtime environment selection to the wrapped stage."""
        return self._stage.conda_env_name

    @property
    def download_requests(self) -> list[file_distribution.DownloadRequest]:
        """Forward distributed download requests to the wrapped stage."""
        return self._stage.download_requests

    @property
    def stage_batch_size(self) -> int:
        """Schedule each saved pickle batch as an independent Xenna sample."""
        return 1

    def stage_setup_on_node(self) -> None:
        """Run wrapped stage node setup."""
        self._stage.stage_setup_on_node()

    def stage_setup(self) -> None:
        """Run wrapped stage setup."""
        self._stage.stage_setup()

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Run the wrapped stage on each envelope's inner task batch."""
        output_batches: list[PipelineTask] = []
        for task in tasks:
            batch = cast("StageCompareBatch", task)
            input_tasks = _load_stage_compare_batch_tasks(batch, self._serializer)
            output_tasks = self._stage.process_data(input_tasks) or []
            output_batches.append(
                StageCompareBatch(
                    batch_name=batch.batch_name,
                    input_path=batch.input_path,
                    golden_path=batch.golden_path,
                    tasks=output_tasks,
                )
            )
        return output_batches

    def destroy(self) -> None:
        """Destroy the wrapped stage."""
        self._stage.destroy()


class StageCompareFinalizeStage(CuratorStage):
    """Compare candidate batches against golden batches inside Xenna workers."""

    def __init__(self, serializer: TaskSerializer, atol: float) -> None:
        """Store compare dependencies."""
        self._serializer = serializer
        self._atol = atol

    @property
    def stage_batch_size(self) -> int:
        """Compare each saved pickle batch independently."""
        return 1

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Load golden batches, compare, and return compact diff results."""
        results: list[PipelineTask] = []
        for task in tasks:
            batch = cast("StageCompareBatch", task)
            candidate_tasks = _load_stage_compare_batch_tasks(batch, self._serializer)
            if batch.golden_path is None:
                msg = f"Stage compare batch {batch.batch_name} has no golden path"
                raise ValueError(msg)
            golden_tasks = self._serializer.load(batch.golden_path)
            task_diffs = _compare_task_lists(batch.batch_name, golden_tasks, candidate_tasks, atol=self._atol)
            results.append(StageCompareBatchResult(batch_name=batch.batch_name, task_diffs=task_diffs))
        return results


def _make_stage_compare_batch_stage_class(base_name: str) -> type[StageCompareBatchStage]:
    """Make a readable stage-compare wrapper class for Xenna progress output."""

    class WrappedStageCompareBatchStage(StageCompareBatchStage):
        pass

    WrappedStageCompareBatchStage.__name__ = f"{base_name}WithStageCompare"
    WrappedStageCompareBatchStage.__qualname__ = WrappedStageCompareBatchStage.__name__
    return WrappedStageCompareBatchStage


def _make_stage_compare_finalize_stage_class(base_name: str) -> type[StageCompareFinalizeStage]:
    """Make a readable stage-compare finalizer class for Xenna progress output."""

    class WrappedStageCompareFinalizeStage(StageCompareFinalizeStage):
        pass

    WrappedStageCompareFinalizeStage.__name__ = f"{base_name}WithStageCompareFinalize"
    WrappedStageCompareFinalizeStage.__qualname__ = WrappedStageCompareFinalizeStage.__name__
    return WrappedStageCompareFinalizeStage


def _load_stage_compare_batch_tasks(batch: StageCompareBatch, serializer: TaskSerializer) -> list[PipelineTask]:
    """Load batch tasks if they are not already materialized."""
    if batch.tasks is not None:
        return batch.tasks
    if batch.input_path is None:
        msg = f"Stage compare batch {batch.batch_name} has no input path"
        raise ValueError(msg)
    return serializer.load(batch.input_path)


_registry: dict[type[PipelineTask], TaskComparator] = {}


def register_comparator(task_type: type[PipelineTask], comparator: TaskComparator) -> None:
    """Register a comparator for a task type."""
    _registry[task_type] = comparator


def get_comparator(task_type: type[PipelineTask]) -> TaskComparator:
    """Return the comparator for a task type."""
    return _registry.get(task_type, _GenericComparator())


def _compare_arrays(
    field_path: str,
    golden: npt.NDArray[Any],
    candidate: npt.NDArray[Any],
    *,
    atol: float,
) -> list[FieldDiff]:
    """Compare NumPy arrays."""
    if golden.shape != candidate.shape:
        return [
            FieldDiff(
                field=field_path,
                detail=f"shape mismatch golden={golden.shape} new={candidate.shape}",
                shape_mismatch=True,
            )
        ]

    if np.issubdtype(golden.dtype, np.number) and np.issubdtype(candidate.dtype, np.number):
        if np.allclose(golden, candidate, atol=atol, rtol=0.0, equal_nan=True):
            return []
        golden_float = golden.astype(np.float64)
        candidate_float = candidate.astype(np.float64)
        both_nan_mask = np.isnan(golden_float) & np.isnan(candidate_float)
        diff = np.abs(golden_float - candidate_float)
        comparable_diff = diff[~both_nan_mask]
        max_diff = float(np.nanmax(comparable_diff)) if comparable_diff.size > 0 else 0.0
        return [FieldDiff(field=field_path, detail=f"max diff {max_diff}", max_diff_observed=max_diff)]

    if np.array_equal(golden, candidate):
        return []
    return [FieldDiff(field=field_path, detail="array values differ")]


def _compare_attrs(
    field_path: str,
    golden: object,
    candidate: object,
    *,
    atol: float,
) -> list[FieldDiff]:
    failures: list[FieldDiff] = []
    golden_attrs = cast("attrs.AttrsInstance", golden)
    for field in attrs.fields(golden_attrs.__class__):
        if field.name in _IGNORED_COMPARE_FIELD_NAMES:
            continue
        child_path = f"{field_path}.{field.name}" if field_path else field.name
        failures.extend(
            _compare_values(
                child_path,
                getattr(golden_attrs, field.name),
                getattr(candidate, field.name),
                atol=atol,
            )
        )
    return failures


def _compare_mapping(
    field_path: str,
    golden: Mapping[object, object],
    candidate: Mapping[object, object],
    *,
    atol: float,
) -> list[FieldDiff]:
    golden_keys = {key for key in golden if key not in _IGNORED_COMPARE_FIELD_NAMES}
    candidate_keys = {key for key in candidate if key not in _IGNORED_COMPARE_FIELD_NAMES}
    if golden_keys != candidate_keys:
        return [
            FieldDiff(
                field=field_path,
                detail=(
                    f"dict key mismatch golden={sorted(golden_keys, key=repr)!r} "
                    f"new={sorted(candidate_keys, key=repr)!r}"
                ),
            )
        ]
    failures: list[FieldDiff] = []
    for key in sorted(golden_keys, key=repr):
        child_path = f"{field_path}.{key}" if field_path else str(key)
        failures.extend(_compare_values(child_path, golden[key], candidate[key], atol=atol))
    return failures


def _compare_sequence(
    field_path: str,
    golden: Sequence[object],
    candidate: Sequence[object],
    *,
    atol: float,
) -> list[FieldDiff]:
    if len(golden) != len(candidate):
        return [FieldDiff(field=field_path, detail=f"length mismatch golden={len(golden)} new={len(candidate)}")]
    failures: list[FieldDiff] = []
    for index, (golden_item, candidate_item) in enumerate(zip(golden, candidate, strict=True)):
        failures.extend(_compare_values(f"{field_path}[{index}]", golden_item, candidate_item, atol=atol))
    return failures


def _compare_values(
    field_path: str,
    golden: object,
    candidate: object,
    *,
    atol: float,
) -> list[FieldDiff]:
    """Recursively compare values for the generic comparator."""
    if type(golden) is not type(candidate):
        return [
            FieldDiff(
                field=field_path,
                detail=f"type mismatch golden={type(golden).__name__} new={type(candidate).__name__}",
            )
        ]
    if isinstance(golden, np.ndarray):
        return _compare_arrays(field_path, golden, cast("npt.NDArray[Any]", candidate), atol=atol)
    if attrs.has(golden.__class__):
        return _compare_attrs(field_path, golden, candidate, atol=atol)
    if isinstance(golden, Mapping):
        golden_m = cast("Mapping[object, object]", golden)
        return _compare_mapping(field_path, golden_m, cast("Mapping[object, object]", candidate), atol=atol)
    if isinstance(golden, Sequence) and not isinstance(golden, (str, bytes, bytearray)):
        golden_s = cast("Sequence[object]", golden)
        return _compare_sequence(field_path, golden_s, cast("Sequence[object]", candidate), atol=atol)
    return (
        []
        if golden == candidate
        else [FieldDiff(field=field_path, detail=f"value mismatch golden={golden!r} new={candidate!r}")]
    )


def _collect_attrs_paths(field_path: str, value: object) -> set[str]:
    paths: set[str] = set()
    value_attrs = cast("attrs.AttrsInstance", value)
    for field in attrs.fields(value_attrs.__class__):
        if field.name in _IGNORED_COMPARE_FIELD_NAMES:
            continue
        child_path = f"{field_path}.{field.name}" if field_path else field.name
        paths.update(_collect_field_paths(child_path, getattr(value_attrs, field.name)))
    return paths


def _collect_mapping_paths(field_path: str, value: Mapping[object, object]) -> set[str]:
    keys = [key for key in value if key not in _IGNORED_COMPARE_FIELD_NAMES]
    if len(keys) == 0:
        return {field_path}
    paths: set[str] = set()
    for key in sorted(keys, key=repr):
        child_path = f"{field_path}.{key}" if field_path else str(key)
        paths.update(_collect_field_paths(child_path, value[key]))
    return paths


def _collect_sequence_paths(field_path: str, value: Sequence[object]) -> set[str]:
    if len(value) == 0:
        return {field_path}
    paths: set[str] = set()
    for index, item in enumerate(value):
        paths.update(_collect_field_paths(f"{field_path}[{index}]", item))
    return paths


def _collect_field_paths(field_path: str, value: object) -> set[str]:
    """Collect comparable leaf field paths for a value."""
    if isinstance(value, np.ndarray):
        return {field_path}
    if attrs.has(value.__class__):
        return _collect_attrs_paths(field_path, value)
    if isinstance(value, Mapping):
        return _collect_mapping_paths(field_path, cast("Mapping[object, object]", value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _collect_sequence_paths(field_path, cast("Sequence[object]", value))
    return {field_path}


def compare_attrs_fields(
    golden: object,
    candidate: object,
    *,
    field_names: Sequence[str],
    atol: float,
) -> list[FieldDiff]:
    """Compare a selected set of attrs fields using generic recursion."""
    failures: list[FieldDiff] = []
    for field_name in field_names:
        failures.extend(
            _compare_values(
                field_name,
                getattr(golden, field_name),
                getattr(candidate, field_name),
                atol=atol,
            )
        )
    return failures


def collect_checked_attrs_fields(golden: object, *, field_names: Sequence[str]) -> tuple[str, ...]:
    """Collect leaf field paths for a selected set of attrs fields."""
    checked_fields: set[str] = set()
    for field_name in field_names:
        checked_fields.update(_collect_field_paths(field_name, getattr(golden, field_name)))
    return tuple(sorted(checked_fields))


class _GenericComparator:
    """Generic attrs-aware task comparator."""

    def checked_fields(self, golden: PipelineTask) -> tuple[str, ...]:
        """Return all comparable leaf fields for the task."""
        return tuple(sorted(_collect_field_paths("", golden)))

    def compare(
        self,
        golden: PipelineTask,
        candidate: PipelineTask,
        *,
        atol: float,
    ) -> list[FieldDiff]:
        """Compare two tasks using recursive attrs reflection."""
        return _compare_values("", golden, candidate, atol=atol)


def _summarize_task_diffs(task_diffs: list[TaskDiff]) -> dict[str, FieldCompareSummary]:
    """Aggregate field-level failures across tasks."""
    summaries: dict[str, FieldCompareSummary] = {}
    for task_diff in task_diffs:
        failed_fields = {failure.field for failure in task_diff.failures}
        for checked_field in task_diff.checked_fields:
            summary = summaries.get(checked_field, FieldCompareSummary())
            summaries[checked_field] = FieldCompareSummary(
                passed=summary.passed + int(checked_field not in failed_fields),
                failed=summary.failed + int(checked_field in failed_fields),
                max_diff_observed=summary.max_diff_observed,
                shape_mismatches=summary.shape_mismatches,
            )

        for failure in task_diff.failures:
            summary = summaries.get(failure.field, FieldCompareSummary())
            max_diff_observed = summary.max_diff_observed
            if failure.max_diff_observed is not None:
                max_diff_observed = (
                    failure.max_diff_observed
                    if max_diff_observed is None
                    else max(max_diff_observed, failure.max_diff_observed)
                )
            summaries[failure.field] = FieldCompareSummary(
                passed=summary.passed,
                failed=summary.failed,
                max_diff_observed=max_diff_observed,
                shape_mismatches=summary.shape_mismatches + int(failure.shape_mismatch),
            )

    return summaries


def _compare_task_lists(
    batch_file: str,
    golden_tasks: list[PipelineTask],
    candidate_tasks: list[PipelineTask],
    *,
    atol: float,
) -> list[TaskDiff]:
    """Compare two task lists."""
    if len(golden_tasks) != len(candidate_tasks):
        return [
            TaskDiff(
                batch_file=batch_file,
                task_index=0,
                checked_fields=("tasks",),
                failures=(
                    FieldDiff(
                        field="tasks",
                        detail=f"task count mismatch golden={len(golden_tasks)} new={len(candidate_tasks)}",
                    ),
                ),
            )
        ]

    task_diffs: list[TaskDiff] = []
    for task_index, (golden_task, candidate_task) in enumerate(zip(golden_tasks, candidate_tasks, strict=True)):
        comparator = get_comparator(type(golden_task))
        failures = comparator.compare(golden_task, candidate_task, atol=atol)
        checked_fields = comparator.checked_fields(golden_task)
        task_diffs.append(
            TaskDiff(
                batch_file=batch_file,
                task_index=task_index,
                checked_fields=checked_fields,
                failures=tuple(failures),
            )
        )
    return task_diffs


def _stage_from_entry(stage_or_spec: StageCompareStage) -> CuratorStage:
    """Return the concrete stage from a bare stage or stage spec."""
    if isinstance(stage_or_spec, CuratorStageSpec):
        return cast("CuratorStage", stage_or_spec.stage)
    return stage_or_spec


def _stage_name(stage_or_spec: StageCompareStage) -> str:
    """Return the concrete stage class name."""
    return _stage_from_entry(stage_or_spec).__class__.__name__


def get_stage_name_after(stages: Sequence[StageCompareStage], stage_name: str) -> str:
    """Return the stage name immediately after the requested stage."""
    normalized_stages = [_stage_from_entry(stage) for stage in stages]
    for index, stage in enumerate(normalized_stages):
        if stage.__class__.__name__ != stage_name:
            continue
        if index + 1 >= len(normalized_stages):
            msg = f"--stage-compare cannot infer golden for last stage {stage_name}"
            raise ValueError(msg)
        return normalized_stages[index + 1].__class__.__name__
    msg = f"Stage {stage_name} not found in pipeline"
    raise ValueError(msg)


def get_stages_to_compare(
    stages: Sequence[StageCompareStage],
    start_stage_name: str,
    end_stage_name: str,
) -> list[StageCompareStage]:
    """Return stages in the half-open compare interval [start_stage_name, end_stage_name)."""
    compare_stages: list[StageCompareStage] = []
    started = False
    found_start = False
    found_end = False

    for stage in stages:
        name = _stage_name(stage)

        if name == end_stage_name:
            if not started:
                msg = f"End stage {end_stage_name} occurs before start stage {start_stage_name}"
                raise ValueError(msg)
            found_end = True
            break

        if name == start_stage_name:
            started = True
            found_start = True

        if started:
            compare_stages.append(stage)

    if not found_start:
        msg = f"Start stage {start_stage_name} not found in pipeline"
        raise ValueError(msg)
    if not found_end:
        msg = f"End stage {end_stage_name} not found in pipeline"
        raise ValueError(msg)
    if len(compare_stages) == 0:
        msg = f"No stages found to compare in [{start_stage_name}, {end_stage_name})"
        raise ValueError(msg)
    return compare_stages


def _wrap_stage_for_xenna_compare(stage_or_spec: StageCompareStage, serializer: TaskSerializer) -> StageCompareStage:
    """Wrap a stage for Xenna compare while preserving spec settings."""
    stage = _stage_from_entry(stage_or_spec)
    wrapped_stage_class = _make_stage_compare_batch_stage_class(stage.__class__.__name__)
    wrapped_stage = wrapped_stage_class(stage, serializer)
    if isinstance(stage_or_spec, CuratorStageSpec):
        return attrs.evolve(stage_or_spec, stage=wrapped_stage)
    return wrapped_stage


def _run_stage_compare_serial(
    stages: Sequence[StageCompareStage],
    input_batches: list[list[PipelineTask]],
    executor: StageExecutor | None,
) -> list[list[PipelineTask]]:
    """Run compare stages serially through the direct executor."""
    _executor = executor if executor is not None else DirectStageExecutor()
    node_info, worker_metadata = NodeInfo(node_id="localhost"), WorkerMetadata.make_dummy()
    candidate_batches = input_batches
    for stage in stages:
        candidate_batches = _executor.execute_stage(
            _stage_from_entry(stage),
            candidate_batches,
            node_info,
            worker_metadata,
        )
    return candidate_batches


def _run_stage_compare_xenna(  # noqa: PLR0913
    stages: Sequence[StageCompareStage],
    batch_paths_by_name: list[tuple[str, TaskPath, TaskPath]],
    *,
    model_weights_prefix: str,
    runner: RunnerInterface | None,
    args: argparse.Namespace | None,
    serializer: TaskSerializer,
    atol: float,
) -> list[StageCompareBatchResult]:
    """Run compare stages through the normal Xenna pipeline entry point."""
    from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline  # noqa: PLC0415

    input_batches = [
        StageCompareBatch(batch_name=batch_name, input_path=input_path, golden_path=golden_path)
        for batch_name, input_path, golden_path in batch_paths_by_name
    ]
    wrapped_stages = [_wrap_stage_for_xenna_compare(stage, serializer) for stage in stages]
    finalize_stage_class = _make_stage_compare_finalize_stage_class(_stage_name(stages[-1]))
    wrapped_stages.append(finalize_stage_class(serializer, atol))
    output_batches = cast(
        "list[PipelineTask]",
        run_pipeline(
            input_batches,
            wrapped_stages,
            model_weights_prefix=model_weights_prefix,
            runner=runner,
            args=args,
        ),
    )

    for batch in output_batches:
        if not isinstance(batch, StageCompareBatchResult):
            msg = f"Xenna stage compare returned unexpected task type {type(batch).__name__}"
            raise TypeError(msg)
    return cast("list[StageCompareBatchResult]", output_batches)


def _index_compare_results(compare_results: Sequence[StageCompareBatchResult]) -> dict[str, list[TaskDiff]]:
    """Index compare results by saved pickle filename."""
    results_by_name: dict[str, list[TaskDiff]] = {}
    for batch in compare_results:
        if batch.batch_name in results_by_name:
            msg = f"Duplicate candidate batch {batch.batch_name}"
            raise ValueError(msg)
        results_by_name[batch.batch_name] = batch.task_diffs
    return results_by_name


def _validate_candidate_batch_names(
    candidate_batch_names: set[str],
    golden_batch_names: set[str],
) -> None:
    """Validate that candidate output names match golden input names."""
    if candidate_batch_names == golden_batch_names:
        return

    missing = sorted(golden_batch_names - candidate_batch_names)
    extra = sorted(candidate_batch_names - golden_batch_names)
    msg = f"Candidate/golden batch file mismatch: missing={missing} extra={extra}"
    raise ValueError(msg)


def _make_serializer(serializer: TaskSerializer | None, profile_name: str | None) -> TaskSerializer:
    """Return the configured serializer."""
    if serializer is not None:
        return serializer
    if profile_name is not None:
        return PickleTaskSerializer(profile_name=profile_name)
    return PickleTaskSerializer()


def _find_matching_batch_files(
    input_path: TaskPath,
    golden_path: TaskPath,
    limit: int,
    serializer: TaskSerializer,
) -> tuple[list[str], dict[str, TaskPath], dict[str, TaskPath]]:
    """Find input and golden batch files that have matching pickle filenames."""
    input_files = serializer.find_task_files(input_path, "*.pkl", limit)
    golden_files = serializer.find_task_files(golden_path, "*.pkl", limit)

    if len(input_files) == 0:
        msg = f"No input tasks found in {input_path}"
        raise ValueError(msg)
    if len(golden_files) == 0:
        msg = f"No golden tasks found in {golden_path}"
        raise ValueError(msg)
    if len(input_files) != len(golden_files):
        msg = f"Input/golden batch count mismatch: input={len(input_files)} golden={len(golden_files)}"
        raise ValueError(msg)

    input_files_by_name = {Path(str(path)).name: path for path in input_files}
    golden_files_by_name = {Path(str(path)).name: path for path in golden_files}
    if input_files_by_name.keys() != golden_files_by_name.keys():
        msg = (
            "Input/golden batch file mismatch: "
            f"input={sorted(input_files_by_name)} golden={sorted(golden_files_by_name)}"
        )
        raise ValueError(msg)

    sorted_batch_names = sorted(input_files_by_name)
    return sorted_batch_names, input_files_by_name, golden_files_by_name


def _load_batches_by_name(
    batch_names: Sequence[str],
    files_by_name: dict[str, TaskPath],
    serializer: TaskSerializer,
) -> dict[str, list[PipelineTask]]:
    """Load batches for serial compare execution."""
    return {name: serializer.load(files_by_name[name]) for name in batch_names}


def _build_report_from_task_diffs(
    stage_name: str,
    sorted_batch_names: list[str],
    task_diffs_by_name: dict[str, list[TaskDiff]],
    *,
    atol: float,
    profile_name: str | None,
) -> CompareReport:
    """Build the stage compare report from per-batch task diffs."""
    all_task_diffs: list[TaskDiff] = []
    passed_batches = 0
    total_batches = len(sorted_batch_names)

    for batch_index, batch_name in enumerate(sorted_batch_names, start=1):
        task_diffs = task_diffs_by_name[batch_name]
        all_task_diffs.extend(task_diffs)
        if all(task_diff.passed for task_diff in task_diffs):
            passed_batches += 1
        if batch_index % 100 == 0:
            logger.info(f"[stage-compare] {stage_name}: {batch_index}/{total_batches} batches processed...")

    failed_batches = total_batches - passed_batches
    pass_rate = passed_batches / total_batches
    return CompareReport(
        stage=stage_name,
        atol=atol,
        total_batches=total_batches,
        passed_batches=passed_batches,
        failed_batches=failed_batches,
        pass_rate=pass_rate,
        fields=_summarize_task_diffs(all_task_diffs),
        failures=[task_diff for task_diff in all_task_diffs if not task_diff.passed],
        profile_name=profile_name,
    )


def _build_report_from_batches(  # noqa: PLR0913
    stage_name: str,
    sorted_batch_names: list[str],
    candidate_batches_by_name: dict[str, list[PipelineTask]],
    golden_batches_by_name: dict[str, list[PipelineTask]],
    *,
    atol: float,
    profile_name: str | None,
) -> CompareReport:
    """Build the stage compare report from materialized task batches."""
    task_diffs_by_name = {
        batch_name: _compare_task_lists(
            batch_name,
            golden_batches_by_name[batch_name],
            candidate_batches_by_name[batch_name],
            atol=atol,
        )
        for batch_name in sorted_batch_names
    }
    return _build_report_from_task_diffs(
        stage_name,
        sorted_batch_names,
        task_diffs_by_name,
        atol=atol,
        profile_name=profile_name,
    )


def run_stage_compare(  # noqa: PLR0913
    stages: Sequence[StageCompareStage],
    input_path: TaskPath,
    golden_path: TaskPath,
    atol: float,
    limit: int,
    pass_threshold: float,
    *,
    report_path: TaskPath,
    profile_name: str | None = None,
    executor: StageExecutor | None = None,
    serializer: TaskSerializer | None = None,
    backend: StageCompareBackend = "xenna",
    runner: RunnerInterface | None = None,
    args: argparse.Namespace | None = None,
    model_weights_prefix: str = MODEL_WEIGHTS_PREFIX,
) -> StageCompareResult:
    """Run stage comparison from saved input tasks against golden tasks."""
    if len(stages) == 0:
        msg = "No stages to compare"
        raise ValueError(msg)
    if backend not in ("serial", "xenna"):
        msg = f"Unsupported stage compare backend {backend}"
        raise ValueError(msg)
    if backend == "xenna" and executor is not None:
        msg = "executor is only supported with backend='serial'"
        raise ValueError(msg)

    _serializer = _make_serializer(serializer, profile_name)
    sorted_batch_names, input_files_by_name, golden_files_by_name = _find_matching_batch_files(
        input_path,
        golden_path,
        limit,
        _serializer,
    )
    stage_name = _stage_name(stages[-1])
    if backend == "serial":
        input_batches_by_name = _load_batches_by_name(sorted_batch_names, input_files_by_name, _serializer)
        golden_batches_by_name = _load_batches_by_name(sorted_batch_names, golden_files_by_name, _serializer)
        candidate_batches = _run_stage_compare_serial(
            stages,
            [input_batches_by_name[name] for name in sorted_batch_names],
            executor,
        )
        if len(candidate_batches) != len(golden_batches_by_name):
            msg = (
                "Candidate/golden batch count mismatch: "
                f"candidate={len(candidate_batches)} golden={len(golden_batches_by_name)}"
            )
            raise ValueError(msg)
        candidate_batches_by_name = dict(zip(sorted_batch_names, candidate_batches, strict=True))
        report = _build_report_from_batches(
            stage_name,
            sorted_batch_names,
            candidate_batches_by_name,
            golden_batches_by_name,
            atol=atol,
            profile_name=profile_name,
        )
    else:
        compare_results = _run_stage_compare_xenna(
            stages,
            [(name, input_files_by_name[name], golden_files_by_name[name]) for name in sorted_batch_names],
            model_weights_prefix=model_weights_prefix,
            runner=runner,
            args=args,
            serializer=_serializer,
            atol=atol,
        )
        task_diffs_by_name = _index_compare_results(compare_results)
        _validate_candidate_batch_names(set(task_diffs_by_name), set(golden_files_by_name))
        report = _build_report_from_task_diffs(
            stage_name,
            sorted_batch_names,
            task_diffs_by_name,
            atol=atol,
            profile_name=profile_name,
        )
    report.write_json(report_path)

    status = "PASSED" if report.pass_rate >= pass_threshold else "FAILED"
    logger.info(
        f"[stage-compare] {status}  {report.passed_batches}/{report.total_batches} "
        f"({report.pass_rate * 100:.1f}%)  report: {report_path}"
    )

    return StageCompareResult(report=report, report_path=report_path, pass_threshold=pass_threshold)
