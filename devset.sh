#!/bin/bash

# Install local developer hooks and run a packaging smoke test.

set -euo pipefail

pixi run pre-commit install
pixi run build
