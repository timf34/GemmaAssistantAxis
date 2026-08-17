#!/usr/bin/env bash
# CUDA forward-compatibility helpers. Side-effect free: defines functions only, never exits, and
# safe to `source` from anywhere (upgrade_for_gemma4.sh deliberately avoids common.sh because a
# `source` of a script that calls `exit` would kill its caller).
#
# WHY THIS FILE EXISTS
# --------------------
# doctor.sh and upgrade_for_gemma4.sh used to hardcode CUDA 12.8 ("cuda-compat-12-8" under
# /usr/local/cuda-12.8/compat). That was the right guess for vLLM 0.19-era wheels, but the version
# the venv actually needs is a property of the INSTALLED TORCH, not a constant: this pod resolved to
# torch 2.13.0+cu130, so the correct package was cuda-compat-13-0 and the 12.8 path never existed.
# Hardcoding it meant the doctor looked for a directory that could not appear and declared the pod
# unfixable. So: ask torch which CUDA it was built for, and look for THAT compat dir.
#
# The forward-compat package ships a newer USER-mode libcuda.so that drives an older KERNEL module
# (datacenter GPUs only — H100/H200/A100). Verified on this pod: driver 550.144.03 (CUDA 12.4)
# kernel module + cuda-compat-13-0 (libcuda 580.65.06) => cuInit rc=0, driver API 13000, H200 visible
# to torch. The kernel driver is a host property and is NOT changed by any of this.

# CUDA version the venv's torch was built against, e.g. "13.0". Empty if torch cannot be queried.
# Note this works even when the GPU is unusable — torch.version.cuda is a build constant, and
# `import torch` itself succeeds on a too-old driver (only cuda.is_available() goes False).
cuda_compat_torch_cuda() {
  local axis="${1:-${AXIS_DIR:-}}"
  [[ -d "$axis" ]] || return 0
  (cd "$axis" && uv run python -c \
    "import torch; print(torch.version.cuda or '')" 2>/dev/null) | tr -d '[:space:]'
}

# apt package name for a CUDA version: 13.0 -> cuda-compat-13-0
cuda_compat_pkg() { echo "cuda-compat-${1//./-}"; }

# Directory holding usable compat libs, or empty. Prefers the version torch wants; otherwise falls
# back to the highest-versioned /usr/local/cuda-*/compat that actually contains libcuda.so.1.
# The "actually contains" test matters: this image shipped a cuda-compat-12-8 dpkg entry whose files
# had been stripped out (the version in dpkg status is not even in NVIDIA's repo), so
# "package installed" and "libs present" are genuinely different questions here.
cuda_compat_find_dir() {
  local want="${1:-}" d
  if [[ -n "$want" && -e "/usr/local/cuda-$want/compat/libcuda.so.1" ]]; then
    echo "/usr/local/cuda-$want/compat"; return 0
  fi
  for d in $(ls -d /usr/local/cuda-*/compat 2>/dev/null | sort -V -r); do
    [[ -e "$d/libcuda.so.1" ]] && { echo "$d"; return 0; }
  done
  echo ""
}

# Install the compat package for a CUDA version (best effort, quiet). Adds NVIDIA's apt repo if the
# package is unknown. Returns 0 only if usable libs are on disk afterwards.
cuda_compat_install() {
  local ver="$1" pkg; pkg="$(cuda_compat_pkg "$ver")"
  local sudo_cmd=""; [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1 && sudo_cmd="sudo -n"
  ( export DEBIAN_FRONTEND=noninteractive
    $sudo_cmd apt-get update -qq >/dev/null 2>&1 || true
    if ! $sudo_cmd apt-get install -y -qq "$pkg" >/dev/null 2>&1; then
      . /etc/os-release; local distro="${ID}${VERSION_ID//./}"
      curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb" \
        -o /tmp/cuda-keyring.deb 2>/dev/null \
        && $sudo_cmd dpkg -i /tmp/cuda-keyring.deb >/dev/null 2>&1 \
        && $sudo_cmd apt-get update -qq >/dev/null 2>&1 \
        && $sudo_cmd apt-get install -y -qq "$pkg" >/dev/null 2>&1 || true
    fi ) || true
  [[ -n "$(cuda_compat_find_dir "$ver")" ]]
}
