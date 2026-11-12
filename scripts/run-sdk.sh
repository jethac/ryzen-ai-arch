#!/usr/bin/env bash
# Run a Ryzen AI command with libraries scoped to this process.
# Usage: run-sdk.sh SDK_ROOT VENV_PYTHON [PYTHON_ARGS ...]
set -euo pipefail
[[ $# -ge 2 ]] || { echo "Usage: $0 SDK_ROOT VENV_PYTHON [PYTHON_ARGS ...]" >&2; exit 2; }
sdk=$(realpath "$1")
python=$2
shift 2
compat="$(dirname "$python")/../compat-lib"
mkdir -p "$compat"
if [[ ! -e "$compat/libncurses.so.6" && -e /usr/lib/libncursesw.so.6 ]]; then
  ln -s /usr/lib/libncursesw.so.6 "$compat/libncurses.so.6"
fi
pylib="$(dirname "$(dirname "$(readlink -f "$python")")")/lib"
export RYZEN_AI_INSTALLATION_PATH="$sdk"
export LD_LIBRARY_PATH="$compat:$sdk/deployment/lib:$sdk/onnxruntime/lib:$pylib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec bash "$(dirname "$0")/npu-run.sh" "$python" "$@"
