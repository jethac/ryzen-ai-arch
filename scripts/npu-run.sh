#!/usr/bin/env bash
# Raise only this process's memlock limit, then run the requested command as this user.
# sudo authorizes prlimit; the requested command does not run as root.
set -euo pipefail
[[ $# -gt 0 ]] || { echo "Usage: $0 COMMAND [ARG ...]" >&2; exit 2; }
if [[ $(ulimit -Hl) != unlimited ]]; then
  sudo prlimit --pid "$$" --memlock=unlimited
fi
ulimit -l unlimited
source /opt/xilinx/xrt/setup.sh >/dev/null
if [[ -n ${RYZENAI_XRT_ROOT:-} ]]; then
  # A matched, source-built userspace stack can be compared without changing system packages.
  export XILINX_XRT="$(realpath "$RYZENAI_XRT_ROOT")"
  for library in libxrt_core.so libxrt_coreutil.so libxrt_driver_xdna.so; do
    [[ -f "$XILINX_XRT/lib/$library" ]] || { echo "Missing $XILINX_XRT/lib/$library" >&2; exit 1; }
  done
  export LD_LIBRARY_PATH="$XILINX_XRT/lib:$LD_LIBRARY_PATH"
fi
exec "$@"
