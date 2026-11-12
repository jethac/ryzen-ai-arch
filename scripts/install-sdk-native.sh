#!/usr/bin/env bash
# Install the AMD 1.9 userspace SDK natively, using Arch's bsdtar instead of dpkg.
# Usage: install-sdk-native.sh AMD_SETUP_SCRIPT AMD_RUNTIME_DEB INSTALL_ROOT
# The vendor installer presents/accepts its own EULA through its -a option.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: $0 AMD_SETUP_SCRIPT AMD_RUNTIME_DEB INSTALL_ROOT" >&2
  exit 2
fi
setup=$(realpath "$1")
deb=$(realpath "$2")
root=$(realpath -m "$3")
command -v bsdtar >/dev/null
uv_bin=$(command -v uv || true)
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then uv_bin="$HOME/.local/bin/uv"; fi
[[ -n "$uv_bin" ]] || { echo 'Install uv first.' >&2; exit 1; }
"$uv_bin" python install 3.12
py=$("$uv_bin" python find 3.12)
export PATH="$(dirname "$py"):$PATH"
mkdir -p "$root"
python3.12 - "$setup" "$root/setup-native.sh" <<'PY'
import pathlib, sys
source = pathlib.Path(sys.argv[1]).read_text()
old = '    dpkg-deb -x "${DEB_FILE}" "${CPP_INSTALL_PATH}"'
new = '''    data_member=$(bsdtar -tf "${DEB_FILE}" | sed -n '/^data\\.tar/p')
    [[ -n "$data_member" && "$data_member" != *$'\\n'* ]] || exit 1
    bsdtar -xOf "${DEB_FILE}" "$data_member" | bsdtar -xf - -C "${CPP_INSTALL_PATH}"'''
if source.count(old) != 1:
    raise SystemExit('Vendor installer changed: inspect extraction before adapting it.')
pathlib.Path(sys.argv[2]).write_text(source.replace(old, new))
PY
sha256sum "$setup" "$deb" > "$root/source-sha256.txt"
bash "$root/setup-native.sh" -a yes -y -d "$deb" -c "$root/runtime" -p "$root/venv"
