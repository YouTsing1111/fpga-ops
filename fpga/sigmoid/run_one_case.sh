#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ -z "${XDMA_TOOLS_DIR:-}" ]]; then
    if [[ -x /home/zwz/RTL_FPGA/Jianyuan-host_xdma/linux-kernel/tools/dma_to_device ]]; then
        XDMA_TOOLS_DIR=/home/zwz/RTL_FPGA/Jianyuan-host_xdma/linux-kernel/tools
    else
        XDMA_TOOLS_DIR=/home/zwz/Jianyuan-host_xdma/linux-kernel/tools
    fi
fi
python3 "$HERE/aiacc_u280_case_runner.py" \
  --case-dir "$HERE/case" \
  --tools-dir "$XDMA_TOOLS_DIR" \
  "$@"
