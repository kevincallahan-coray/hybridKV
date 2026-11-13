#!/usr/bin/env bash
# Sourced by every GPU job: builds a venv in /tmp (never on CephFS, per NRP policy)
# and prints the GPU it landed on.  No sleeps, no long-lived idle processes.
set -euo pipefail
python -m venv /tmp/santa-venv
source /tmp/santa-venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -q -r /repo/requirements.txt
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(), 'bf16', torch.cuda.is_bf16_supported())"
nvidia-smi --query-gpu=name,memory.total --format=csv
