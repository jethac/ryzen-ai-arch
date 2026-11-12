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
exec "$@"
