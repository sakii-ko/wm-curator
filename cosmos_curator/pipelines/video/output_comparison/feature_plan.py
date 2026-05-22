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
"""Shared feature-planning contracts for split output comparison."""

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Protocol

import attrs

from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue
from cosmos_curator.pipelines.video.output_comparison.report import FeatureComparison, Issue
from cosmos_curator.pipelines.video.output_comparison.summary_loader import OutputRoot
from cosmos_curator.pipelines.video.output_comparison.summary_schema import OutputSummary
from cosmos_curator.pipelines.video.output_comparison.video_schema import ClipComparisonSpec, VideoComparisonSpec


@attrs.define(frozen=True)
class FeatureComparisonContext:
    """Planning data passed to each feature comparison planner.

    This contains the two output roots, loaded summaries, selected video specs,
    storage profile, and optional selectors. A feature uses it to decide whether
    summary data is enough or clip-level Ray Data work is needed.
    """

    output_a: OutputRoot
    output_b: OutputRoot
    summary_a: OutputSummary
    summary_b: OutputSummary
    profile_name: str
    specs: tuple[VideoComparisonSpec, ...]
    video_limit: int | None = None
    selected_video_key: str | None = None


@attrs.define(frozen=True)
class FeatureComparisonResult:
    """Issues and report data emitted by one feature comparison planner."""

    issues: tuple[Issue, ...]
    comparison: FeatureComparison


type FeatureComparisonResults = FeatureComparisonResult | Mapping[str, FeatureComparisonResult]


@attrs.define(frozen=True)
class ResolvedFeaturePlan:
    """Feature comparison plan that is already resolved without artifact work."""

    feature_name: str
    result: FeatureComparisonResults


@attrs.define(frozen=True)
class ClipFeaturePlan:
    """Feature comparison plan that runs one Ray row per selected clip.

    Attributes:
        feature_name: Feature name used in report output.
        clip_specs: Clip rows that should enter the Ray Data stage.
        load_worker_class: Callable class used as an actor-backed load/normalize
            worker.
        load_worker_constructor_kwargs: Constructor kwargs for
            ``load_worker_class``.
        compare_row: Worker-side comparison callable for one loaded row.
        reduce_rows: Driver-side reducer for compact comparison rows. Reducers
            may return one feature result or a mapping of feature names to
            results when one clip-load pass feeds multiple report features.
        load_group_id: Optional stable identity for grouping plans that can
            share one load stage when constructor kwargs are not JSON-compatible.

    """

    feature_name: str
    clip_specs: tuple[ClipComparisonSpec, ...]
    load_worker_class: type
    load_worker_constructor_kwargs: Mapping[str, Any]
    compare_row: Callable[[Mapping[str, JsonValue]], JsonDictObject]
    reduce_rows: Callable[[Sequence[Mapping[str, JsonValue]]], FeatureComparisonResults]
    load_group_id: Hashable | None = None


type FeatureComparisonPlan = ResolvedFeaturePlan | ClipFeaturePlan


class FeatureComparisonPlanner(Protocol):
    """Feature-specific output comparison planner."""

    @property
    def name(self) -> str:
        """Return the feature name used in report output."""

    def build_plan(self, context: FeatureComparisonContext) -> FeatureComparisonPlan:
        """Build a resolved or clip-row feature comparison plan."""
