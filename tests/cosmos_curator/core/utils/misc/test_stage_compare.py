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
"""Tests for cosmos_curator.core.utils.misc.stage_compare."""

import json
from pathlib import Path
from typing import cast

import attrs
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    CuratorStageSpec,
    PipelineTask,
)
from cosmos_curator.core.utils.misc import stage_compare
from cosmos_curator.core.utils.misc.stage_compare import (
    CompareReport,
    FieldCompareSummary,
    FieldDiff,
    TaskDiff,
    _GenericComparator,
    get_comparator,
    get_stage_name_after,
    get_stages_to_compare,
    register_comparator,
    run_stage_compare,
)
from cosmos_curator.core.utils.misc.stage_replay import DirectStageExecutor, PickleTaskSerializer, TaskPath


@attrs.define
class NestedPayload:
    """Nested attrs payload for comparison tests."""

    name: str
    values: npt.NDArray[np.float32]


@attrs.define
class CompareTask(PipelineTask):
    """Pipeline task used in stage compare tests."""

    value: int
    array: npt.NDArray[np.float32]
    nested: NestedPayload
    stage_perf: dict[str, object] = attrs.Factory(dict)


class AddOneStage(CuratorStage):
    """Test stage that increments task value."""

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
        """Increment values in CompareTask instances."""
        output: list[PipelineTask] = []
        for task in tasks:
            compare_task = cast("CompareTask", task)
            output.append(
                CompareTask(
                    value=compare_task.value + 1,
                    array=compare_task.array + 1,
                    nested=NestedPayload(name=compare_task.nested.name, values=compare_task.nested.values + 1),
                )
            )
        return output


class CpuHalfStage(AddOneStage):
    """Stage with distinctive resources for wrapper forwarding tests."""

    @property
    def resources(self) -> CuratorStageResource:
        """Return distinctive resources."""
        return CuratorStageResource(cpus=0.5, gpus=0.0)


class CompareStartStage(CuratorStage):
    """First stage for range-selection tests."""


class CompareMiddleStage(CuratorStage):
    """Middle stage for range-selection tests."""


class CompareEndStage(CuratorStage):
    """End stage for range-selection tests."""


class ReversingRunner(RunnerInterface):
    """Runner that returns final stage outputs in reverse order."""

    def run(
        self,
        input_tasks: list[PipelineTask],
        stage_specs: list[CuratorStageSpec],
        _model_weights_prefix: str,
        _execution_mode: str = "AUTO",
    ) -> list[PipelineTask] | None:
        """Execute wrapped stages and reverse the final outputs."""
        tasks = input_tasks
        for spec in stage_specs:
            result = spec.stage.process_data(tasks)
            tasks = result if result is not None else []
        return list(reversed(tasks))


class DuplicateBatchRunner(RunnerInterface):
    """Runner that returns duplicate candidate batch names."""

    def run(
        self,
        input_tasks: list[PipelineTask],
        stage_specs: list[CuratorStageSpec],
        _model_weights_prefix: str,
        _execution_mode: str = "AUTO",
    ) -> list[PipelineTask] | None:
        """Return duplicate envelopes to exercise validation."""
        del input_tasks, stage_specs, _model_weights_prefix, _execution_mode
        return [
            stage_compare.StageCompareBatchResult(batch_name="batch_000.task.pkl", task_diffs=[]),
            stage_compare.StageCompareBatchResult(batch_name="batch_000.task.pkl", task_diffs=[]),
        ]


class CountingPickleTaskSerializer(PickleTaskSerializer):
    """Pickle serializer that tracks load calls."""

    def __init__(self) -> None:
        """Initialize load tracking."""
        super().__init__()
        self.load_count = 0

    def load(self, path: TaskPath) -> list[PipelineTask]:
        """Track each load call."""
        self.load_count += 1
        return super().load(path)


class LazyLoadingRunner(RunnerInterface):
    """Runner that verifies the driver did not load task payloads before Xenna execution."""

    def __init__(self, serializer: CountingPickleTaskSerializer) -> None:
        """Store the serializer to inspect load counts."""
        self._serializer = serializer

    def run(
        self,
        input_tasks: list[PipelineTask],
        stage_specs: list[CuratorStageSpec],
        _model_weights_prefix: str,
        _execution_mode: str = "AUTO",
    ) -> list[PipelineTask] | None:
        """Assert lazy inputs, then execute wrapped stages in process."""
        assert self._serializer.load_count == 0
        assert all(isinstance(task, stage_compare.StageCompareBatch) for task in input_tasks)
        assert all(cast("stage_compare.StageCompareBatch", task).tasks is None for task in input_tasks)

        tasks = input_tasks
        for spec in stage_specs:
            result = spec.stage.process_data(tasks)
            tasks = result if result is not None else []
        return tasks


class SpecInspectingRunner(RunnerInterface):
    """Runner that asserts compare wrapping preserves stage spec settings."""

    def run(
        self,
        input_tasks: list[PipelineTask],
        stage_specs: list[CuratorStageSpec],
        _model_weights_prefix: str,
        _execution_mode: str = "AUTO",
    ) -> list[PipelineTask] | None:
        """Assert spec settings and execute the wrapped stage."""
        assert len(stage_specs) == 2
        spec = stage_specs[0]
        assert spec.num_workers_per_node == 3
        assert spec.num_run_attempts_python == 2
        assert spec.over_provision_factor == 1.5
        assert isinstance(spec.stage, stage_compare.StageCompareBatchStage)
        assert spec.stage.required_resources.cpus == 0.5

        tasks = input_tasks
        for spec in stage_specs:
            result = spec.stage.process_data(tasks)
            tasks = result if result is not None else []
        return tasks


class OffsetComparator:
    """Custom comparator for registry tests."""

    def checked_fields(self, golden: PipelineTask) -> tuple[str, ...]:
        """Return the field this comparator reports."""
        del golden
        return ("custom",)

    def compare(self, golden: PipelineTask, candidate: PipelineTask, *, atol: float) -> list[FieldDiff]:
        """Return a fixed failure to prove the registry lookup path."""
        del golden, candidate, atol
        return [FieldDiff(field="custom", detail="forced mismatch")]


class ArrayOnlyComparator:
    """Custom comparator that intentionally checks only one field."""

    def checked_fields(self, golden: PipelineTask) -> tuple[str, ...]:
        """Return the single field this comparator evaluates."""
        return stage_compare.collect_checked_attrs_fields(golden, field_names=("array",))

    def compare(self, golden: PipelineTask, candidate: PipelineTask, *, atol: float) -> list[FieldDiff]:
        """Compare only the array field and ignore the rest of the task."""
        return stage_compare.compare_attrs_fields(
            golden,
            candidate,
            field_names=("array",),
            atol=atol,
        )


def _make_task(value: int) -> CompareTask:
    """Create a test task."""
    base = np.array([value, value + 1], dtype=np.float32)
    return CompareTask(
        value=value,
        array=base.copy(),
        nested=NestedPayload(name=f"task-{value}", values=base.copy()),
    )


def test_get_stage_name_after_returns_immediate_successor() -> None:
    """The helper should return the immediate successor stage name."""
    stages = [CompareStartStage(), CompareMiddleStage(), CompareEndStage()]

    assert get_stage_name_after(stages, "CompareStartStage") == "CompareMiddleStage"


def test_get_stage_name_after_last_stage_raises() -> None:
    """The helper should reject one-stage compare inference for the final stage."""
    stages = [CompareStartStage(), CompareMiddleStage()]

    with pytest.raises(ValueError, match="cannot infer golden for last stage"):
        get_stage_name_after(stages, "CompareMiddleStage")


def test_get_stages_to_compare_uses_half_open_interval_and_preserves_specs() -> None:
    """The compare range helper should select [start, end) while preserving stage specs."""
    middle_spec = CuratorStageSpec(CompareMiddleStage(), num_workers_per_node=2)
    stages: list[CuratorStage | CuratorStageSpec] = [CompareStartStage(), middle_spec, CompareEndStage()]

    compare_stages = get_stages_to_compare(stages, "CompareStartStage", "CompareEndStage")

    assert [type(stage.stage if isinstance(stage, CuratorStageSpec) else stage) for stage in compare_stages] == [
        CompareStartStage,
        CompareMiddleStage,
    ]
    assert compare_stages[1] is middle_spec


def test_get_stages_to_compare_rejects_end_before_start() -> None:
    """The compare range helper should report when the end stage appears before the start stage."""
    stages = [CompareEndStage(), CompareStartStage()]

    with pytest.raises(ValueError, match="End stage CompareEndStage occurs before start stage CompareStartStage"):
        get_stages_to_compare(stages, "CompareStartStage", "CompareEndStage")


def test_generic_comparator_exact_match() -> None:
    """The generic comparator should pass on equal attrs tasks."""
    task = _make_task(1)
    assert _GenericComparator().compare(task, _make_task(1), atol=0.0) == []


def test_generic_comparator_allclose_within_tolerance() -> None:
    """Numeric arrays should use allclose with the configured atol."""
    golden = _make_task(1)
    candidate = _make_task(1)
    candidate.array = np.array([1.25, 2.0], dtype=np.float32)
    failures = _GenericComparator().compare(golden, candidate, atol=0.3)
    assert failures == []


def test_compare_arrays_reports_unsigned_max_diff_without_wraparound() -> None:
    """Unsigned numeric diffs should be reported without subtraction wraparound."""
    failures = stage_compare._compare_arrays(
        "array",
        np.array([0], dtype=np.uint8),
        np.array([255], dtype=np.uint8),
        atol=0.0,
    )

    assert len(failures) == 1
    assert failures[0].field == "array"
    assert failures[0].max_diff_observed == 255.0


def test_compare_arrays_matching_nans_do_not_report_nan_diff() -> None:
    """Matching NaN positions should not produce a failure with a NaN max diff."""
    failures = stage_compare._compare_arrays(
        "array",
        np.array([np.nan, 1.0], dtype=np.float32),
        np.array([np.nan, 1.0], dtype=np.float32),
        atol=0.0,
    )

    assert failures == []


def test_generic_comparator_ignores_stage_perf() -> None:
    """Stage performance timing metadata should not affect semantic task comparison."""
    golden = _make_task(1)
    golden.stage_perf["StageA"] = {"elapsed": 1.0}
    candidate = _make_task(1)
    candidate.stage_perf["StageB"] = {"elapsed": 2.0}

    comparator = _GenericComparator()

    assert all(not field.startswith("stage_perf") for field in comparator.checked_fields(golden))
    assert comparator.compare(golden, candidate, atol=0.0) == []


def test_generic_comparator_shape_mismatch() -> None:
    """Shape mismatches should be reported explicitly."""
    golden = _make_task(1)
    candidate = _make_task(1)
    candidate.array = np.array([[1.0, 2.0]], dtype=np.float32)
    failures = _GenericComparator().compare(golden, candidate, atol=0.0)
    assert len(failures) == 1
    assert failures[0].field == "array"
    assert failures[0].shape_mismatch is True


def test_get_comparator_uses_registry() -> None:
    """Registered comparators should override the generic comparator."""
    register_comparator(CompareTask, OffsetComparator())
    try:
        comparator = get_comparator(CompareTask)
        failures = comparator.compare(_make_task(1), _make_task(1), atol=0.0)
        assert len(failures) == 1
        assert failures[0].field == "custom"
    finally:
        stage_compare._registry.pop(CompareTask, None)


def test_compare_report_to_dict() -> None:
    """CompareReport should serialize to the expected JSON shape."""
    report = CompareReport(
        stage="StageA",
        atol=0.0,
        total_batches=2,
        passed_batches=1,
        failed_batches=1,
        pass_rate=0.5,
        fields={"array": FieldCompareSummary(passed=0, failed=1, max_diff_observed=3.0, shape_mismatches=0)},
        failures=[
            TaskDiff(
                batch_file="a.task.pkl",
                task_index=0,
                checked_fields=("array",),
                failures=(FieldDiff(field="array", detail="max diff 3.0"),),
            )
        ],
    )
    data = report.to_dict()
    assert data["stage"] == "StageA"
    assert data["fields"]["array"]["failed"] == 1
    assert data["failures"][0]["batch_file"] == "a.task.pkl"


def test_summarize_task_diffs_does_not_double_count_failures() -> None:
    """Field summaries should count a failing checked field once per task diff."""
    task_diffs = [
        TaskDiff(
            batch_file="a.task.pkl",
            task_index=0,
            checked_fields=("array",),
            failures=(FieldDiff(field="array", detail="max diff 1.0"),),
        )
    ]

    summary = stage_compare._summarize_task_diffs(task_diffs)

    assert summary["array"].passed == 0
    assert summary["array"].failed == 1


def test_compare_task_lists_custom_comparator_summary_ignores_unchecked_fields() -> None:
    """Custom comparators should not report ignored fields as passed."""
    register_comparator(CompareTask, ArrayOnlyComparator())
    try:
        golden_task = _make_task(1)
        candidate_task = _make_task(1)
        candidate_task.value = 999
        candidate_task.nested = NestedPayload(name="changed", values=candidate_task.nested.values)

        task_diffs = stage_compare._compare_task_lists(
            "batch_000.task.pkl",
            [golden_task],
            [candidate_task],
            atol=0.0,
        )
        summary = stage_compare._summarize_task_diffs(task_diffs)

        assert set(summary) == {"array"}
        assert summary["array"].passed == 1
        assert summary["array"].failed == 0
    finally:
        stage_compare._registry.pop(CompareTask, None)


def test_run_stage_compare_writes_report_and_passes(tmp_path: Path) -> None:
    """run_stage_compare should write a report and report success for matching outputs."""
    serializer = PickleTaskSerializer()
    input_dir = tmp_path / "tasks" / "StageA"
    golden_dir = tmp_path / "tasks" / "StageB"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)

    serializer.save(input_dir / "batch_000.task.pkl", [_make_task(1)])
    serializer.save(golden_dir / "batch_000.task.pkl", [AddOneStage().process_data([_make_task(1)])[0]])

    report_path = tmp_path / "compare" / "StageA" / "report.json"
    result = run_stage_compare(
        [AddOneStage()],
        input_dir,
        golden_dir,
        atol=0.0,
        limit=0,
        pass_threshold=1.0,
        report_path=report_path,
        backend="serial",
        executor=DirectStageExecutor(),
        serializer=serializer,
    )

    assert result.report.failed_batches == 0
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["passed_batches"] == 1
    assert data["failed_batches"] == 0


def test_run_stage_compare_reports_failures(tmp_path: Path) -> None:
    """run_stage_compare should capture field failures in the report."""
    serializer = PickleTaskSerializer()
    input_dir = tmp_path / "tasks" / "StageA"
    golden_dir = tmp_path / "tasks" / "StageB"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)

    serializer.save(input_dir / "batch_000.task.pkl", [_make_task(1)])
    serializer.save(golden_dir / "batch_000.task.pkl", [_make_task(999)])

    report_path = tmp_path / "compare" / "StageA" / "report.json"
    result = run_stage_compare(
        [AddOneStage()],
        input_dir,
        golden_dir,
        atol=0.0,
        limit=0,
        pass_threshold=1.0,
        report_path=report_path,
        backend="serial",
        executor=DirectStageExecutor(),
        serializer=serializer,
    )

    assert result.report.failed_batches == 1
    data = json.loads(report_path.read_text())
    assert data["failed_batches"] == 1
    assert len(data["failures"]) >= 1


def test_run_stage_compare_field_summary_counts_include_passing_batches(tmp_path: Path) -> None:
    """Field summaries should count passing tasks, not just failures."""
    serializer = PickleTaskSerializer()
    input_dir = tmp_path / "tasks" / "StageA"
    golden_dir = tmp_path / "tasks" / "StageB"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)

    serializer.save(input_dir / "batch_000.task.pkl", [_make_task(1)])
    serializer.save(golden_dir / "batch_000.task.pkl", [AddOneStage().process_data([_make_task(1)])[0]])

    serializer.save(input_dir / "batch_001.task.pkl", [_make_task(1)])
    serializer.save(golden_dir / "batch_001.task.pkl", [_make_task(999)])

    report_path = tmp_path / "compare" / "StageA" / "report.json"
    result = run_stage_compare(
        [AddOneStage()],
        input_dir,
        golden_dir,
        atol=0.0,
        limit=0,
        pass_threshold=0.0,
        report_path=report_path,
        backend="serial",
        executor=DirectStageExecutor(),
        serializer=serializer,
    )

    assert result.report.fields["value"].passed == 1
    assert result.report.fields["value"].failed == 1


def test_run_stage_compare_matches_batches_by_filename(tmp_path: Path) -> None:
    """Compare should reject mismatched batch filenames even when counts match."""
    serializer = PickleTaskSerializer()
    input_dir = tmp_path / "tasks" / "StageA"
    golden_dir = tmp_path / "tasks" / "StageB"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)

    serializer.save(input_dir / "a.task.pkl", [_make_task(1)])
    serializer.save(golden_dir / "b.task.pkl", [AddOneStage().process_data([_make_task(1)])[0]])

    report_path = tmp_path / "compare" / "StageA" / "report.json"
    with pytest.raises(ValueError, match="Input/golden batch file mismatch"):
        run_stage_compare(
            [AddOneStage()],
            input_dir,
            golden_dir,
            atol=0.0,
            limit=0,
            pass_threshold=1.0,
            report_path=report_path,
            backend="serial",
            executor=DirectStageExecutor(),
            serializer=serializer,
        )


def test_run_stage_compare_xenna_matches_candidate_batches_by_filename(tmp_path: Path) -> None:
    """Xenna-backed compare should not rely on returned batch order."""
    serializer = PickleTaskSerializer()
    input_dir = tmp_path / "tasks" / "StageA"
    golden_dir = tmp_path / "tasks" / "StageB"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)

    serializer.save(input_dir / "batch_000.task.pkl", [_make_task(1)])
    serializer.save(input_dir / "batch_001.task.pkl", [_make_task(10)])
    serializer.save(golden_dir / "batch_000.task.pkl", [AddOneStage().process_data([_make_task(1)])[0]])
    serializer.save(golden_dir / "batch_001.task.pkl", [AddOneStage().process_data([_make_task(10)])[0]])

    result = run_stage_compare(
        [AddOneStage()],
        input_dir,
        golden_dir,
        atol=0.0,
        limit=0,
        pass_threshold=1.0,
        report_path=tmp_path / "compare" / "report.json",
        backend="xenna",
        runner=ReversingRunner(),
        serializer=serializer,
    )

    assert result.passed is True
    assert result.report.passed_batches == 2


def test_run_stage_compare_xenna_defers_task_loading_to_runner(tmp_path: Path) -> None:
    """Xenna-backed compare should pass path envelopes to the runner instead of loaded task payloads."""
    serializer = CountingPickleTaskSerializer()
    input_dir = tmp_path / "tasks" / "StageA"
    golden_dir = tmp_path / "tasks" / "StageB"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)

    serializer.save(input_dir / "batch_000.task.pkl", [_make_task(1)])
    serializer.save(golden_dir / "batch_000.task.pkl", [AddOneStage().process_data([_make_task(1)])[0]])

    result = run_stage_compare(
        [AddOneStage()],
        input_dir,
        golden_dir,
        atol=0.0,
        limit=0,
        pass_threshold=1.0,
        report_path=tmp_path / "compare" / "report.json",
        runner=LazyLoadingRunner(serializer),
        serializer=serializer,
    )

    assert result.passed is True
    assert serializer.load_count == 2


def test_run_stage_compare_xenna_rejects_duplicate_candidate_batch_names(tmp_path: Path) -> None:
    """Xenna-backed compare should reject duplicate returned batch names."""
    serializer = PickleTaskSerializer()
    input_dir = tmp_path / "tasks" / "StageA"
    golden_dir = tmp_path / "tasks" / "StageB"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)

    serializer.save(input_dir / "batch_000.task.pkl", [_make_task(1)])
    serializer.save(golden_dir / "batch_000.task.pkl", [AddOneStage().process_data([_make_task(1)])[0]])

    with pytest.raises(ValueError, match="Duplicate candidate batch"):
        run_stage_compare(
            [AddOneStage()],
            input_dir,
            golden_dir,
            atol=0.0,
            limit=0,
            pass_threshold=1.0,
            report_path=tmp_path / "compare" / "report.json",
            backend="xenna",
            runner=DuplicateBatchRunner(),
            serializer=serializer,
        )


def test_run_stage_compare_xenna_preserves_stage_spec_settings(tmp_path: Path) -> None:
    """Xenna-backed compare should preserve CuratorStageSpec settings when wrapping stages."""
    serializer = PickleTaskSerializer()
    input_dir = tmp_path / "tasks" / "StageA"
    golden_dir = tmp_path / "tasks" / "StageB"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)

    serializer.save(input_dir / "batch_000.task.pkl", [_make_task(1)])
    serializer.save(golden_dir / "batch_000.task.pkl", [CpuHalfStage().process_data([_make_task(1)])[0]])
    stage_spec = CuratorStageSpec(
        CpuHalfStage(),
        num_workers_per_node=3,
        num_run_attempts_python=2,
        over_provision_factor=1.5,
    )

    result = run_stage_compare(
        [stage_spec],
        input_dir,
        golden_dir,
        atol=0.0,
        limit=0,
        pass_threshold=1.0,
        report_path=tmp_path / "compare" / "report.json",
        backend="xenna",
        runner=SpecInspectingRunner(),
        serializer=serializer,
    )

    assert result.passed is True
