# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Test nvcf_split_benchmark."""

import json
from pathlib import Path
from secrets import randbelow
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

from benchmarks.secrets import KratosSecrets
from benchmarks.split_pipeline.nvcf_split_benchmark import (
    _read_optional_json,
    _summary_counts_are_valid,
    report_metrics,
)


def _make_caption_quality_stats() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "split_video_pipeline",
        "caption_windows_checked": 5,
        "caption_status_counts": {
            "success": 2,
            "truncated": 1,
            "blocked": 1,
            "error": 1,
            "skipped": 0,
        },
        "empty_caption_count": 1,
        "sentinel_caption_count": 2,
    }


def _make_caption_quality_metrics() -> dict[str, Any]:
    return {
        "caption_quality_stats_present": 1,
        "caption_windows_checked": 5,
        "caption_status_success": 2,
        "caption_status_truncated": 1,
        "caption_status_blocked": 1,
        "caption_status_error": 1,
        "caption_status_skipped": 0,
        "empty_caption_count": 1,
        "sentinel_caption_count": 2,
    }


def _make_kratos_secrets() -> KratosSecrets:
    bearer_token = "test_token"  # noqa: S105
    return KratosSecrets(api_key="test_api", bearer_token=bearer_token)


@patch("benchmarks.split_pipeline.nvcf_split_benchmark.push_cloudevent")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_cloudevent")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.print_json")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_summary_metrics")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_optional_json")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_summary_json")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.logger")
def test_report_metrics_happy_path(  # noqa: PLR0913
    mock_logger: MagicMock,  # noqa: ARG001
    mock_read_summary_json: MagicMock,
    mock_read_optional_json: MagicMock,
    mock_make_summary_metrics: MagicMock,
    mock_print_json: MagicMock,  # noqa: ARG001
    mock_make_cloudevent: MagicMock,
    mock_push_cloudevent: MagicMock,
) -> None:
    """Test report_metrics function happy path."""
    # Arrange
    test_summary_path = "s3://bucket/path/summary.json"
    test_transport_params: dict[str, Any] = {
        "client_kwargs": {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "region_name": "us-east-1",
        }
    }
    test_num_nodes = 2
    test_gpus_per_node = 4
    test_caption = True
    test_image = "nvcr.io/test-org/staging-cosmos-curator"
    test_tag = "test-tag"
    test_scheduled = False
    test_backend = "test-backend"
    test_gpu = "H100"
    test_instance_type = "OCI.GPU.H100_8x"
    test_captioning_algorithm = "qwen"
    test_max_concurrency = 2
    test_limit = 5000
    test_funcid = "test-funcid"
    test_version = "test-version"
    test_ci_pipeline_source = "web"
    test_gitlab_user_login = "test-user"
    test_input_path = "s3://bucket/input"
    test_output_path = "s3://bucket/output"
    test_vllm_sampling_temperature = 0.01
    test_kratos_metrics_endpoint = "https://metrics.example.com"
    bearer_token = "test_token"  # noqa: S105
    test_kratos_secrets = KratosSecrets(api_key="test_api", bearer_token=bearer_token)

    # Mock data
    test_summary_data = {"pipeline_run_time": 60, "total_video_duration": 3600}
    test_summary_metrics = {"env": "nvcf", "caption": True, "num_nodes": 2}
    test_caption_quality_stats = _make_caption_quality_stats()
    test_caption_quality_metrics = _make_caption_quality_metrics()
    test_metrics_metadata = {
        "image": test_image,
        "tag": test_tag,
        "scheduled": test_scheduled,
        "backend": test_backend,
        "gpu": test_gpu,
        "instance_type": test_instance_type,
        "captioning_algorithm": test_captioning_algorithm,
        "gpus_per_node": test_gpus_per_node,
        "max_concurrency": test_max_concurrency,
        "limit": test_limit,
        "funcid": test_funcid,
        "version": test_version,
        "ci_pipeline_source": test_ci_pipeline_source,
        "gitlab_user_login": test_gitlab_user_login,
        "input_path": test_input_path,
        "output_path": test_output_path,
        "vllm_sampling_temperature": test_vllm_sampling_temperature,
    }
    expected_summary_metrics = {**test_summary_metrics, **test_caption_quality_metrics, **test_metrics_metadata}
    test_cloudevent = {"specversion": "1.0", "data": test_summary_metrics}

    # Configure mocks
    mock_read_summary_json.return_value = test_summary_data
    mock_read_optional_json.return_value = test_caption_quality_stats
    mock_make_summary_metrics.return_value = test_summary_metrics
    mock_make_cloudevent.return_value = test_cloudevent
    mock_push_cloudevent.return_value = {"status": "success", "message": "Event pushed successfully"}

    # Act
    report_metrics(
        summary_path=test_summary_path,
        transport_params=test_transport_params,
        num_nodes=test_num_nodes,
        gpus_per_node=test_gpus_per_node,
        caption=test_caption,
        splitting_algorithm="transnetv2",
        metrics_metadata=test_metrics_metadata,
        kratos_metrics_endpoint=test_kratos_metrics_endpoint,
        kratos_secrets=test_kratos_secrets,
    )

    # Assert
    mock_read_summary_json.assert_called_once_with(test_summary_path, test_transport_params)
    mock_read_optional_json.assert_called_once_with(
        "s3://bucket/path/caption_quality_stats.json", test_transport_params
    )
    mock_make_summary_metrics.assert_called_once_with(
        test_summary_data,
        test_num_nodes,
        test_gpus_per_node,
        caption=test_caption,
        env="nvcf",
        splitting_algorithm="transnetv2",
    )
    mock_make_cloudevent.assert_called_once_with(expected_summary_metrics)
    mock_push_cloudevent.assert_called_once_with(test_cloudevent, test_kratos_metrics_endpoint, test_kratos_secrets)


def test_report_metrics_missing_caption_quality_stats_emits_absent() -> None:
    """Missing optional caption quality stats should not fail reporting."""
    summary_path = "s3://bucket/path/summary.json"
    transport_params: dict[str, Any] = {}
    base_summary_metrics = {"env": "nvcf", "caption": 1}

    with (
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_summary_json") as mock_read_summary_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_optional_json") as mock_read_optional_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_summary_metrics") as mock_make_summary_metrics,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_cloudevent") as mock_make_cloudevent,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.push_cloudevent") as mock_push_cloudevent,
    ):
        mock_read_summary_json.return_value = {"pipeline_run_time": 60}
        mock_read_optional_json.return_value = None
        mock_make_summary_metrics.return_value = dict(base_summary_metrics)
        mock_make_cloudevent.return_value = {"specversion": "1.0"}
        mock_push_cloudevent.return_value = {"status": "success"}

        report_metrics(
            summary_path=summary_path,
            transport_params=transport_params,
            num_nodes=1,
            gpus_per_node=8,
            caption=True,
            splitting_algorithm="transnetv2",
            kratos_metrics_endpoint="https://metrics.example.com",
            kratos_secrets=_make_kratos_secrets(),
        )

    mock_read_optional_json.assert_called_once_with("s3://bucket/path/caption_quality_stats.json", transport_params)
    mock_make_cloudevent.assert_called_once_with({**base_summary_metrics, "caption_quality_stats_present": 0})


def test_report_metrics_bare_summary_path_uses_bare_caption_quality_sibling() -> None:
    """Bare relative summary paths should resolve to a bare relative sibling path."""
    with (
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_summary_json") as mock_read_summary_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_optional_json") as mock_read_optional_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_summary_metrics") as mock_make_summary_metrics,
    ):
        mock_read_summary_json.return_value = {"pipeline_run_time": 60}
        mock_read_optional_json.return_value = None
        mock_make_summary_metrics.return_value = {"env": "nvcf", "caption": 1}

        report_metrics(
            summary_path="summary.json",
            transport_params={},
            num_nodes=1,
            gpus_per_node=8,
            caption=True,
            splitting_algorithm="transnetv2",
            metrics_path=None,
        )

    mock_read_optional_json.assert_called_once_with("caption_quality_stats.json", {})


def test_report_metrics_malformed_caption_quality_stats_warns_and_emits_absent() -> None:
    """Malformed caption quality stats should be visible in logs but not fail reporting."""
    summary_path = "s3://bucket/path/summary.json"
    base_summary_metrics = {"env": "nvcf", "caption": 1}

    with (
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_summary_json") as mock_read_summary_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_optional_json") as mock_read_optional_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_summary_metrics") as mock_make_summary_metrics,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_cloudevent") as mock_make_cloudevent,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.push_cloudevent") as mock_push_cloudevent,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.logger") as mock_logger,
    ):
        mock_read_summary_json.return_value = {"pipeline_run_time": 60}
        mock_read_optional_json.return_value = {"schema_version": 2}
        mock_make_summary_metrics.return_value = dict(base_summary_metrics)
        mock_make_cloudevent.return_value = {"specversion": "1.0"}
        mock_push_cloudevent.return_value = {"status": "success"}

        report_metrics(
            summary_path=summary_path,
            transport_params={},
            num_nodes=1,
            gpus_per_node=8,
            caption=True,
            splitting_algorithm="transnetv2",
            kratos_metrics_endpoint="https://metrics.example.com",
            kratos_secrets=_make_kratos_secrets(),
        )

    mock_make_cloudevent.assert_called_once_with({**base_summary_metrics, "caption_quality_stats_present": 0})
    mock_logger.warning.assert_called_once_with(
        "Unusable caption quality stats in s3://bucket/path/caption_quality_stats.json"
    )


def test_report_metrics_caption_false_skips_caption_quality_read() -> None:
    """Caption-disabled runs should emit the absent marker without reading a sibling artifact."""
    base_summary_metrics = {"env": "nvcf", "caption": 0}

    with (
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_summary_json") as mock_read_summary_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_optional_json") as mock_read_optional_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_summary_metrics") as mock_make_summary_metrics,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_cloudevent") as mock_make_cloudevent,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.push_cloudevent") as mock_push_cloudevent,
    ):
        mock_read_summary_json.return_value = {"pipeline_run_time": 60}
        mock_make_summary_metrics.return_value = dict(base_summary_metrics)
        mock_make_cloudevent.return_value = {"specversion": "1.0"}
        mock_push_cloudevent.return_value = {"status": "success"}

        report_metrics(
            summary_path="s3://bucket/path/summary.json",
            transport_params={},
            num_nodes=1,
            gpus_per_node=8,
            caption=False,
            splitting_algorithm="transnetv2",
            kratos_metrics_endpoint="https://metrics.example.com",
            kratos_secrets=_make_kratos_secrets(),
        )

    mock_read_optional_json.assert_not_called()
    mock_make_cloudevent.assert_called_once_with({**base_summary_metrics, "caption_quality_stats_present": 0})


def test_report_metrics_metrics_path_includes_caption_quality_stats(tmp_path: Path) -> None:
    """Saved metrics should include the same caption quality fields sent to other sinks."""
    base_summary_metrics = {"env": "nvcf", "caption": 1}
    expected_metrics = {**base_summary_metrics, **_make_caption_quality_metrics()}
    metrics_path = tmp_path / "metrics.json"

    with (
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_summary_json") as mock_read_summary_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark._read_optional_json") as mock_read_optional_json,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_summary_metrics") as mock_make_summary_metrics,
    ):
        mock_read_summary_json.return_value = {"pipeline_run_time": 60}
        mock_read_optional_json.return_value = _make_caption_quality_stats()
        mock_make_summary_metrics.return_value = dict(base_summary_metrics)

        report_metrics(
            summary_path="s3://bucket/path/summary.json",
            transport_params={},
            num_nodes=1,
            gpus_per_node=8,
            caption=True,
            splitting_algorithm="transnetv2",
            metrics_path=str(metrics_path),
        )

    assert json.loads(metrics_path.read_text()) == expected_metrics


def test_read_optional_json_invalid_json_warns_and_returns_none() -> None:
    """Invalid optional JSON should warn and fail open."""
    mock_file = mock_open()

    with (
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.smart_open.open") as mock_smart_open,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.json.load") as mock_json_load,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.logger") as mock_logger,
    ):
        mock_smart_open.return_value = mock_file.return_value
        mock_json_load.side_effect = json.JSONDecodeError("bad json", "{}", 0)

        assert _read_optional_json("s3://bucket/path/caption_quality_stats.json", {}) is None

    mock_logger.warning.assert_called_once()


def test_read_optional_json_read_error_returns_none() -> None:
    """Read errors for optional JSON should stay quiet and fail open."""
    with (
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.smart_open.open") as mock_smart_open,
        patch("benchmarks.split_pipeline.nvcf_split_benchmark.logger") as mock_logger,
    ):
        mock_smart_open.side_effect = FileNotFoundError("missing")

        assert _read_optional_json("s3://bucket/path/caption_quality_stats.json", {}) is None

    mock_logger.debug.assert_called_once()


@patch("benchmarks.split_pipeline.nvcf_split_benchmark.json.load")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.smart_open.open")
def test_summary_counts_are_valid_happy_path(
    mock_smart_open: MagicMock,
    mock_json_load: MagicMock,
) -> None:
    """Test summary count validation for valid counts."""
    limit = 100 + randbelow(9901)  # gives 100..10000
    mock_file = mock_open()
    mock_smart_open.return_value = mock_file.return_value
    mock_json_load.return_value = {
        "num_input_videos": limit + 25,
        "num_input_videos_selected": limit,
        "num_processed_videos": limit - 1,
    }

    assert _summary_counts_are_valid("s3://bucket/path/summary.json", transport_params={}, limit=limit)


@patch("benchmarks.split_pipeline.nvcf_split_benchmark.json.load")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.smart_open.open")
def test_summary_counts_are_valid_fallback_for_legacy_summary(
    mock_smart_open: MagicMock,
    mock_json_load: MagicMock,
) -> None:
    """Support older summary.json files that do not include num_input_videos_selected."""
    limit = 100 + randbelow(9901)  # gives 100..10000
    mock_file = mock_open()
    mock_smart_open.return_value = mock_file.return_value
    # Legacy summaries may report full input listing counts, which can exceed limit.
    mock_json_load.return_value = {"num_input_videos": limit + 500, "num_processed_videos": limit - 1}

    assert _summary_counts_are_valid("s3://bucket/path/summary.json", transport_params={}, limit=limit)


@patch("benchmarks.split_pipeline.nvcf_split_benchmark.json.load")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.smart_open.open")
def test_summary_counts_are_valid_rejects_invalid_cases(
    mock_smart_open: MagicMock,
    mock_json_load: MagicMock,
) -> None:
    """Test summary count validation rejects invalid count combinations."""
    mock_file = mock_open()
    mock_smart_open.return_value = mock_file.return_value
    limit = 100 + randbelow(9901)  # gives 100..10000
    invalid_summaries = [
        {"num_input_videos": limit + 10, "num_input_videos_selected": limit + 1, "num_processed_videos": limit},
        {"num_input_videos": limit + 10, "num_input_videos_selected": limit, "num_processed_videos": limit + 1},
        {"num_input_videos": limit + 10, "num_input_videos_selected": limit - 1, "num_processed_videos": limit},
        {"num_input_videos": "not-an-int", "num_input_videos_selected": limit - 1, "num_processed_videos": limit - 1},
        {"num_input_videos": limit - 1, "num_input_videos_selected": limit, "num_processed_videos": limit - 1},
    ]

    for summary in invalid_summaries:
        mock_json_load.return_value = summary
        assert not _summary_counts_are_valid("s3://bucket/path/summary.json", transport_params={}, limit=limit)
