#!/bin/bash

# Install local developer hooks and run a packaging smoke test.

set -euo pipefail

pixi run -e dev-hooks pre-commit install
pixi run -e dev-hooks python -m build
