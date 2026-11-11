#!/bin/bash
set -euo pipefail
umask 027

: "${HUNGER_RESEARCH_ROOT:?source campaign.env first}"
: "${HUNGER_CAMPAIGN_ROOT:?source campaign.env first}"
: "${HUNGER_HPC_VENV:?source campaign.env first}"

if [[ -n "${HUNGER_PYTHON_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "Environment Modules command is unavailable" >&2
    exit 2
  fi
  module load "${HUNGER_PYTHON_MODULE}"
fi

python_bin="${HUNGER_PYTHON_BIN:-python3}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python binary not found: ${python_bin}" >&2
  echo "Check HUNGER_PYTHON_MODULE and HUNGER_PYTHON_BIN, then retry." >&2
  exit 2
fi

python_major_minor="$(${python_bin} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_major_minor}" != "3.11" ]]; then
  echo "Expected Python 3.11, found ${python_major_minor} at ${python_bin}" >&2
  exit 2
fi

if [[ -e "${HUNGER_HPC_VENV}" ]]; then
  echo "Refusing to modify existing environment: ${HUNGER_HPC_VENV}" >&2
  exit 2
fi

mkdir -p "$(dirname "${HUNGER_HPC_VENV}")"
"${python_bin}" -m venv "${HUNGER_HPC_VENV}"
"${HUNGER_HPC_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${HUNGER_HPC_VENV}/bin/python" -m pip install \
  -r "${HUNGER_RESEARCH_ROOT}/hpc/v2_5seed_cpu/requirements-v2-cpu.txt"
"${HUNGER_HPC_VENV}/bin/python" -m pip install \
  --no-deps --editable "${HUNGER_RESEARCH_ROOT}"

cd "${HUNGER_RESEARCH_ROOT}"
HG_MAX_QTY=5 HG_MAX_PRICE=50 \
  "${HUNGER_HPC_VENV}/bin/python" -m hpc.v2_5seed_cpu.preflight \
  --research-root "${HUNGER_RESEARCH_ROOT}" \
  --campaign-root "${HUNGER_CAMPAIGN_ROOT}"
