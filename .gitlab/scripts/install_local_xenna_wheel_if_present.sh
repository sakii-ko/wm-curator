#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# If a locally-built cosmos-xenna wheel artifact is present (downloaded from
# the build_xenna_wheels matrix job), pip-install the wheel matching the host
# architecture into the active Pixi environment so lint/test jobs run against
# the unreleased ref selected by XENNA_REF + the 'use-local-xenna-build' MR
# label.
#
# This is a no-op when the wheel artifact is absent (i.e. when the label is
# not set), so it is safe to call unconditionally from setup_curator. The
# build_xenna_wheels matrix uploads both amd64 and arm64 manylinux wheels;
# this script narrows to the wheel whose tag matches `uname -m`.

set -euo pipefail

WHEEL_DIR="${WHEEL_DIR:-cosmos-xenna/target/wheels}"
ARCH="${ARCH:-$(uname -m)}"

shopt -s nullglob
wheels=("${WHEEL_DIR}"/cosmos_xenna-*manylinux*_"${ARCH}".whl)
shopt -u nullglob

if [[ ${#wheels[@]} -eq 0 ]]; then
    echo "No locally-built cosmos-xenna ${ARCH} wheel found at '${WHEEL_DIR}'; using the Pixi-installed version."
    exit 0
fi

if [[ ${#wheels[@]} -gt 1 ]]; then
    echo "ERROR: multiple cosmos-xenna ${ARCH} wheels found in '${WHEEL_DIR}':"
    printf '  %s\n' "${wheels[@]}"
    echo "Expected exactly one. Aborting."
    exit 1
fi

wheel="${wheels[0]}"
echo "Installing locally-built cosmos-xenna wheel into the active Pixi environment: ${wheel}"
python -m pip install --force-reinstall --no-deps "${wheel}"

# Surface the active cosmos-xenna install path so the install is verifiable
# from the job log.
python - <<'PY'
import importlib.metadata as md
import cosmos_xenna

try:
    version = md.version("cosmos-xenna")
except md.PackageNotFoundError:
    version = "(unknown)"

print(f"cosmos-xenna {version} installed from {cosmos_xenna.__file__}")
PY
