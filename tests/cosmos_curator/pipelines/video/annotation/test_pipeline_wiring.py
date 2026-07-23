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
"""Exercise the intentionally thin annotation pipeline wiring."""

from pathlib import Path

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.video.annotation.adapters import FilesystemDatasetAdapter
from cosmos_curator.pipelines.video.annotation.data_model import AnnotationTask


class _RecordAnnotationStage(CuratorStage):
    """Small stand-in for a caption, geometry, or normal annotation stage."""

    def __init__(self, annotation_name: str) -> None:
        """Store the metadata key this stage should mark."""
        self._annotation_name = annotation_name

    def process_data(self, tasks: list[AnnotationTask]) -> list[AnnotationTask]:  # type: ignore[override]
        """Mark every task so the test can verify stage order and task transport."""
        for task in tasks:
            completed = task.dataset_metadata.setdefault("completed_annotations", [])
            assert isinstance(completed, list)
            completed.append(self._annotation_name)
        return tasks


def test_adapter_tasks_flow_through_an_ordinary_stage_list(
    tmp_path: Path,
    sequential_runner: RunnerInterface,
) -> None:
    """Annotation inputs should need no pipeline wrapper or parallel task model."""
    source = tmp_path / "nested" / "video.mkv"
    source.parent.mkdir()
    source.touch()

    adapter = FilesystemDatasetAdapter(tmp_path, dataset_metadata={"dataset": "demo"})
    tasks = adapter.discover()
    stages: list[CuratorStage | CuratorStageSpec] = [
        _RecordAnnotationStage("caption"),
        CuratorStageSpec(_RecordAnnotationStage("vipe")),
        _RecordAnnotationStage("normal"),
    ]

    output_tasks = run_pipeline(tasks, stages, runner=sequential_runner)

    assert len(output_tasks) == 1
    assert output_tasks[0] is tasks[0]
    assert output_tasks[0].video.input_video == source.resolve()
    assert output_tasks[0].dataset_metadata == {
        "dataset": "demo",
        "completed_annotations": ["caption", "vipe", "normal"],
    }
