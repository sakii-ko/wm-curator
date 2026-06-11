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

"""Schema and resolver for Ray Data video split pipeline configs."""

import copy
import json
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cosmos_curator.core.utils.environment import MODEL_WEIGHTS_PREFIX

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_CONFIG_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")

SchemaVersion = Literal[1]
VideoSplitKind = Literal["video_split"]
VideoSplitTemplateProfile = Literal["base", "smoke"]
SplitMethod = Literal["transnetv2", "fixed_stride"]
TransNetV2MaxLengthMode = Literal["truncate", "stride"]
CaptionBackend = Literal["ray_data_llm"]
CaptionModel = Literal["qwen"]
MetadataFormat = Literal["json"]
TranscodeEncoder = Literal["libopenh264"]


class ConfigResolutionError(ValueError):
    """Raised when defaults, presets, user config, and overrides cannot resolve."""


class VideoSplitInputConfig(BaseModel):
    """Input video discovery configuration."""

    model_config = _CONFIG_MODEL_CONFIG

    video_path: str = Field(min_length=1)
    limit: int = Field(default=0, ge=0)


class FixedStrideSplitConfig(BaseModel):
    """Fixed-stride split settings."""

    model_config = _CONFIG_MODEL_CONFIG

    duration_s: float = Field(default=10.0, gt=0.0)
    stride_s: float = Field(default=10.0, gt=0.0)
    min_clip_length_s: float = Field(default=2.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_lengths(self) -> Self:
        if self.min_clip_length_s > self.duration_s:
            msg = "fixed_stride.min_clip_length_s cannot be greater than fixed_stride.duration_s"
            raise ValueError(msg)
        return self


class TransNetV2SplitConfig(BaseModel):
    """TransNetV2 split settings."""

    model_config = _CONFIG_MODEL_CONFIG

    threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    min_length_s: float = Field(default=2.0, gt=0.0)
    min_length_frames: int = Field(default=48, ge=1)
    max_length_s: float = Field(default=60.0, gt=0.0)
    max_length_mode: TransNetV2MaxLengthMode = "stride"
    crop_s: float = Field(default=0.5, ge=0.0)
    frame_decode_cpus_per_worker: int = Field(default=3, ge=1)
    gpus_per_worker: float = Field(default=0.25, gt=0.0)

    @model_validator(mode="after")
    def _validate_length_bounds(self) -> Self:
        if self.max_length_s < self.min_length_s:
            msg = "Max length is smaller than min length!"
            raise ValueError(msg)
        return self


class VideoSplitSplitConfig(BaseModel):
    """Clip span generation configuration."""

    model_config = _CONFIG_MODEL_CONFIG

    method: SplitMethod = "transnetv2"
    limit_clips: int = Field(default=0, ge=0)
    fixed_stride: FixedStrideSplitConfig = Field(default_factory=FixedStrideSplitConfig)
    transnetv2: TransNetV2SplitConfig = Field(default_factory=TransNetV2SplitConfig)


class TranscodeConfig(BaseModel):
    """Clip transcoding settings."""

    model_config = _CONFIG_MODEL_CONFIG

    encoder: TranscodeEncoder = "libopenh264"
    encoder_threads: int = Field(default=1, ge=1)
    ffmpeg_batch_size: int = Field(default=16, ge=1)
    cpus_per_worker: float = Field(default=5.0, gt=0.0)
    use_input_video_bit_rate: bool = False


class CaptionConfig(BaseModel):
    """Ray Data captioning settings."""

    model_config = _CONFIG_MODEL_CONFIG

    enabled: bool = True
    backend: CaptionBackend = "ray_data_llm"
    model: CaptionModel = "qwen"
    batch_size: int = Field(default=32, ge=1)


class VideoSplitOutputConfig(BaseModel):
    """Output artifact settings."""

    model_config = _CONFIG_MODEL_CONFIG

    clip_path: str = Field(min_length=1)
    metadata_format: MetadataFormat = "json"


class VideoSplitExecutionConfig(BaseModel):
    """Execution settings outside the pipeline's semantic stages."""

    model_config = _CONFIG_MODEL_CONFIG

    model_weights_path: str = Field(default=MODEL_WEIGHTS_PREFIX, min_length=1)
    progress: bool = False


class ResolvedVideoSplitConfig(BaseModel):
    """Fully resolved v1 Ray Data ``video_split`` config used for execution."""

    model_config = _CONFIG_MODEL_CONFIG

    schema_version: SchemaVersion
    kind: VideoSplitKind
    input: VideoSplitInputConfig
    split: VideoSplitSplitConfig = Field(default_factory=VideoSplitSplitConfig)
    transcode: TranscodeConfig = Field(default_factory=TranscodeConfig)
    caption: CaptionConfig = Field(default_factory=CaptionConfig)
    output: VideoSplitOutputConfig
    execution: VideoSplitExecutionConfig = Field(default_factory=VideoSplitExecutionConfig)


class RawVideoSplitInputConfig(BaseModel):
    """Partial input config accepted from user files before default resolution."""

    model_config = _CONFIG_MODEL_CONFIG

    video_path: str | None = Field(default=None, min_length=1)
    limit: int | None = Field(default=None, ge=0)


class RawFixedStrideSplitConfig(BaseModel):
    """Partial fixed-stride settings accepted from user files."""

    model_config = _CONFIG_MODEL_CONFIG

    duration_s: float | None = Field(default=None, gt=0.0)
    stride_s: float | None = Field(default=None, gt=0.0)
    min_clip_length_s: float | None = Field(default=None, gt=0.0)


class RawTransNetV2SplitConfig(BaseModel):
    """Partial TransNetV2 settings accepted from user files."""

    model_config = _CONFIG_MODEL_CONFIG

    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    min_length_s: float | None = Field(default=None, gt=0.0)
    min_length_frames: int | None = Field(default=None, ge=1)
    max_length_s: float | None = Field(default=None, gt=0.0)
    max_length_mode: TransNetV2MaxLengthMode | None = None
    crop_s: float | None = Field(default=None, ge=0.0)
    frame_decode_cpus_per_worker: int | None = Field(default=None, ge=1)
    gpus_per_worker: float | None = Field(default=None, gt=0.0)


class RawVideoSplitSplitConfig(BaseModel):
    """Partial split config accepted from user files."""

    model_config = _CONFIG_MODEL_CONFIG

    preset: str | None = Field(default=None, min_length=1)
    method: SplitMethod | None = None
    limit_clips: int | None = Field(default=None, ge=0)
    fixed_stride: RawFixedStrideSplitConfig | None = None
    transnetv2: RawTransNetV2SplitConfig | None = None


class RawTranscodeConfig(BaseModel):
    """Partial transcode settings accepted from user files."""

    model_config = _CONFIG_MODEL_CONFIG

    encoder: TranscodeEncoder | None = None
    encoder_threads: int | None = Field(default=None, ge=1)
    ffmpeg_batch_size: int | None = Field(default=None, ge=1)
    cpus_per_worker: float | None = Field(default=None, gt=0.0)
    use_input_video_bit_rate: bool | None = None


class RawCaptionConfig(BaseModel):
    """Partial caption config accepted from user files."""

    model_config = _CONFIG_MODEL_CONFIG

    preset: str | None = Field(default=None, min_length=1)
    enabled: bool | None = None
    backend: CaptionBackend | None = None
    model: CaptionModel | None = None
    batch_size: int | None = Field(default=None, ge=1)


class RawVideoSplitOutputConfig(BaseModel):
    """Partial output config accepted from user files."""

    model_config = _CONFIG_MODEL_CONFIG

    clip_path: str | None = Field(default=None, min_length=1)
    metadata_format: MetadataFormat | None = None


class RawVideoSplitExecutionConfig(BaseModel):
    """Partial execution config accepted from user files."""

    model_config = _CONFIG_MODEL_CONFIG

    preset: str | None = Field(default=None, min_length=1)
    model_weights_path: str | None = Field(default=None, min_length=1)
    progress: bool | None = None


class UserVideoSplitConfig(BaseModel):
    """User-authored v1 ``video_split`` config before defaults and presets resolve."""

    model_config = _CONFIG_MODEL_CONFIG

    schema_version: SchemaVersion
    kind: VideoSplitKind
    input: RawVideoSplitInputConfig | None = None
    split: RawVideoSplitSplitConfig | None = None
    transcode: RawTranscodeConfig | None = None
    caption: RawCaptionConfig | None = None
    output: RawVideoSplitOutputConfig | None = None
    execution: RawVideoSplitExecutionConfig | None = None


@dataclass(frozen=True)
class VideoSplitResolution:
    """Resolved config plus the preset fragments selected during resolution."""

    config: ResolvedVideoSplitConfig
    selected_presets: tuple[str, ...]


_DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "kind": "video_split",
    "input": {"limit": 0},
    "split": {
        "method": "transnetv2",
        "limit_clips": 0,
        "fixed_stride": {
            "duration_s": 10.0,
            "stride_s": 10.0,
            "min_clip_length_s": 2.0,
        },
        "transnetv2": {
            "threshold": 0.4,
            "min_length_s": 2.0,
            "min_length_frames": 48,
            "max_length_s": 60.0,
            "max_length_mode": "stride",
            "crop_s": 0.5,
            "frame_decode_cpus_per_worker": 3,
            "gpus_per_worker": 0.25,
        },
    },
    "transcode": {
        "encoder": "libopenh264",
        "encoder_threads": 1,
        "ffmpeg_batch_size": 16,
        "cpus_per_worker": 5.0,
        "use_input_video_bit_rate": False,
    },
    "caption": {
        "enabled": True,
        "backend": "ray_data_llm",
        "model": "qwen",
        "batch_size": 32,
    },
    "output": {
        "metadata_format": "json",
    },
    "execution": {
        "model_weights_path": MODEL_WEIGHTS_PREFIX,
        "progress": False,
    },
}

_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "split": {
        "transnetv2_default": {"method": "transnetv2"},
        "fixed_stride_10s": {
            "method": "fixed_stride",
            "fixed_stride": {"duration_s": 10.0, "stride_s": 10.0, "min_clip_length_s": 2.0},
        },
    },
    "caption": {
        "balanced": {"enabled": True, "backend": "ray_data_llm", "model": "qwen", "batch_size": 32},
        "off": {"enabled": False},
    },
    "execution": {
        "local_gpu": {"model_weights_path": MODEL_WEIGHTS_PREFIX, "progress": False},
        "progress": {"progress": True},
    },
}

_PRESET_SECTIONS = frozenset(_PRESETS)

_VIDEO_SPLIT_REQUIRED_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "path": "schema_version",
        "description": "Config schema version. Use 1 for v1 video_split configs.",
        "example": 1,
    },
    {
        "path": "kind",
        "description": "Pipeline kind. Use video_split for the Ray Data video split pipeline.",
        "example": "video_split",
    },
    {
        "path": "input.video_path",
        "description": "Input video directory or storage prefix visible to the runtime environment.",
        "example": "/config/input",
    },
    {
        "path": "output.clip_path",
        "description": "Output directory or storage prefix for generated clips and metadata.",
        "example": "/config/output",
    },
)

_VIDEO_SPLIT_TEMPLATE_DESCRIPTIONS: dict[VideoSplitTemplateProfile, str] = {
    "base": "Smallest runnable video_split config; relies on packaged defaults.",
    "smoke": "Cheap first-run config; uses fixed-stride splitting and disables captioning.",
}

_VIDEO_SPLIT_TEMPLATES: dict[VideoSplitTemplateProfile, dict[str, Any]] = {
    "base": {
        "schema_version": 1,
        "kind": "video_split",
        "input": {"video_path": "/config/input"},
        "output": {"clip_path": "/config/output"},
    },
    "smoke": {
        "schema_version": 1,
        "kind": "video_split",
        "input": {"video_path": "/config/input"},
        "split": {"preset": "fixed_stride_10s"},
        "caption": {"enabled": False},
        "output": {"clip_path": "/config/output"},
    },
}


def load_video_split_config_data(config_path: str | pathlib.Path) -> dict[str, Any]:
    """Load a user pipeline config file as a mapping."""
    path = pathlib.Path(config_path)
    if not path.exists():
        msg = f"Pipeline config file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with path.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file) if path.suffix.lower() in _YAML_SUFFIXES else json.load(config_file)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        msg = f"Failed to parse config {path}: {exc}"
        raise ValueError(msg) from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)

    return cast("dict[str, Any]", loaded)


def resolve_video_split_config(
    config_path: str | pathlib.Path,
    *,
    overrides: Sequence[str] = (),
) -> VideoSplitResolution:
    """Load and resolve a user-authored ``video_split`` config file."""
    raw_data = load_video_split_config_data(config_path)
    return resolve_video_split_config_data(raw_data, overrides=overrides)


def resolve_video_split_config_data(
    raw_data: Mapping[str, Any],
    *,
    overrides: Sequence[str] = (),
) -> VideoSplitResolution:
    """Resolve raw user config data into the canonical execution config."""
    user_config = UserVideoSplitConfig.model_validate(_normalize_yaml_preset_fields(raw_data))
    user_values = user_config.model_dump(mode="python", exclude_none=True)
    selected_presets = _selected_presets(user_values)

    resolved: dict[str, Any] = copy.deepcopy(_DEFAULT_CONFIG)
    for qualified_name in selected_presets:
        section, preset_name = qualified_name.split(".", maxsplit=1)
        _merge_section_preset(resolved, section, preset_name)

    _deep_merge(resolved, _without_preset_fields(user_values))
    _apply_cli_overrides(resolved, overrides)
    return VideoSplitResolution(
        config=ResolvedVideoSplitConfig.model_validate(resolved),
        selected_presets=selected_presets,
    )


def resolved_config_to_json(config: ResolvedVideoSplitConfig, *, indent: int = 2) -> str:
    """Render a resolved config as canonical JSON."""
    return json.dumps(config.model_dump(mode="json"), indent=indent) + "\n"


def user_video_split_schema_json(*, indent: int = 2) -> str:
    """Return JSON Schema for user-authored v1 ``video_split`` configs."""
    return json.dumps(UserVideoSplitConfig.model_json_schema(), indent=indent) + "\n"


def video_split_config_template(profile: VideoSplitTemplateProfile = "base") -> dict[str, Any]:
    """Return a packaged user-authored ``video_split`` config template."""
    try:
        template = _VIDEO_SPLIT_TEMPLATES[profile]
    except KeyError as exc:
        msg = f"Unknown video_split template profile: {profile}"
        raise ConfigResolutionError(msg) from exc
    return copy.deepcopy(template)


def video_split_required_fields() -> list[dict[str, Any]]:
    """Return required author inputs for a runnable ``video_split`` config."""
    return [copy.deepcopy(field) for field in _VIDEO_SPLIT_REQUIRED_FIELDS]


def video_split_template_payload(profile: VideoSplitTemplateProfile = "base") -> dict[str, Any]:
    """Return machine-readable template metadata for agent and tool use."""
    try:
        description = _VIDEO_SPLIT_TEMPLATE_DESCRIPTIONS[profile]
    except KeyError as exc:
        msg = f"Unknown video_split template profile: {profile}"
        raise ConfigResolutionError(msg) from exc
    return {
        "kind": "video_split",
        "profile": profile,
        "description": description,
        "required_fields": video_split_required_fields(),
        "config": video_split_config_template(profile),
    }


def user_config_to_yaml(data: Mapping[str, Any]) -> str:
    """Render a user-authored config mapping as editable YAML."""
    return yaml.safe_dump(copy.deepcopy(data), sort_keys=False)


def list_video_split_presets() -> list[dict[str, Any]]:
    """Return packaged preset metadata."""
    presets: list[dict[str, Any]] = []
    for section, section_presets in _PRESETS.items():
        for name, fragment in section_presets.items():
            presets.append(
                {
                    "section": section,
                    "name": name,
                    "qualified_name": f"{section}.{name}",
                    "fragment": copy.deepcopy(fragment),
                }
            )
    return presets


def show_video_split_preset(name: str) -> dict[str, Any]:
    """Return one packaged preset by qualified or unique short name."""
    if "." in name:
        section, preset_name = name.split(".", maxsplit=1)
        try:
            fragment = _PRESETS[section][preset_name]
        except KeyError as exc:
            msg = f"Unknown video_split preset: {name}"
            raise ConfigResolutionError(msg) from exc
        return {
            "section": section,
            "name": preset_name,
            "qualified_name": name,
            "fragment": copy.deepcopy(fragment),
        }

    matches = [preset for preset in list_video_split_presets() if preset["name"] == name]
    if not matches:
        msg = f"Unknown video_split preset: {name}"
        raise ConfigResolutionError(msg)
    if len(matches) > 1:
        qualified_names = ", ".join(str(match["qualified_name"]) for match in matches)
        msg = f"Ambiguous preset {name!r}; use one of: {qualified_names}"
        raise ConfigResolutionError(msg)
    return matches[0]


def _normalize_yaml_preset_fields(raw_data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(raw_data))
    for section in _PRESET_SECTIONS:
        section_values = normalized.get(section)
        if isinstance(section_values, dict) and section_values.get("preset") is False:
            section_values["preset"] = "off"
    return normalized


def _selected_presets(user_values: Mapping[str, Any]) -> tuple[str, ...]:
    selected: list[str] = []
    for section in _PRESET_SECTIONS:
        section_values = user_values.get(section)
        if not isinstance(section_values, dict) or "preset" not in section_values:
            continue
        preset_name = str(section_values["preset"])
        if preset_name not in _PRESETS[section]:
            msg = f"Unknown {section} preset for video_split: {preset_name}"
            raise ConfigResolutionError(msg)
        selected.append(f"{section}.{preset_name}")
    return tuple(sorted(selected))


def _merge_section_preset(resolved: dict[str, Any], section: str, preset_name: str) -> None:
    section_values = resolved.get(section)
    if not isinstance(section_values, dict):
        msg = f"Cannot apply {section}.{preset_name}; section {section!r} is not a mapping"
        raise ConfigResolutionError(msg)
    _deep_merge(section_values, _PRESETS[section][preset_name])


def _without_preset_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(dict(values))
    for section in _PRESET_SECTIONS:
        section_values = cleaned.get(section)
        if isinstance(section_values, dict):
            section_values.pop("preset", None)
    return cleaned


def _apply_cli_overrides(data: dict[str, Any], overrides: Sequence[str]) -> None:
    for raw_override in overrides:
        path, value = _parse_cli_override(raw_override)
        if path[-1] == "preset":
            msg = f"--set cannot override preset fields in resolved config: {'.'.join(path)}"
            raise ConfigResolutionError(msg)
        target = data
        for key in path[:-1]:
            next_value = target.get(key)
            if not isinstance(next_value, dict):
                msg = f"--set path {'.'.join(path)} cannot descend into non-object key {key!r}"
                raise ConfigResolutionError(msg)
            target = cast("dict[str, Any]", next_value)
        target[path[-1]] = value


def _parse_cli_override(raw_override: str) -> tuple[list[str], Any]:
    if "=" not in raw_override:
        msg = f"--set override must be PATH=VALUE, got {raw_override!r}"
        raise ConfigResolutionError(msg)
    raw_path, raw_value = raw_override.split("=", maxsplit=1)
    path = raw_path.split(".")
    if any(not part for part in path):
        msg = f"--set override path must contain non-empty keys, got {raw_path!r}"
        raise ConfigResolutionError(msg)
    try:
        value = yaml.safe_load(raw_value) if raw_value else ""
    except yaml.YAMLError as exc:
        msg = f"Failed to parse --set value for {raw_path}: {exc}"
        raise ConfigResolutionError(msg) from exc
    return path, value


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        target_value = target.get(key)
        if isinstance(target_value, dict) and isinstance(value, Mapping):
            _deep_merge(target_value, value)
        else:
            target[key] = copy.deepcopy(value)
