# Strix Halo / Omarchy investigation

Work in progress: native userspace setup for the AMD Ryzen AI Max+ 395 / gfx1151 / XDNA2 on Omarchy (Arch), kernel
7.2.8-9-omarchy. The kernel already exposes PCI 1022:17f0 through amdxdna and /dev/accel/accel0. Keep that driver.

## Native Ryzen AI 1.9 SDK extraction

`scripts/install-sdk-native.sh` adapts AMD's supplied 1.9 setup script to extract the runtime .deb with bsdtar, without
installing it into a Debian database. It creates a private Python 3.12 environment through uv and uses AMD's package
index. AMD's original script and runtime package are required; retrieve them from the official Linux installation page:
https://ryzenai.docs.amd.com/en/latest/linux.html . The wrapper invokes the vendor's EULA acceptance option.

```bash
bash scripts/install-sdk-native.sh /path/setup_ryzen_ai.sh \
  /path/ryzen-ai-runtime_1.9.0-1_amd64.deb "$HOME/ryzen-ai-native"
```

This does not install a kernel module. Installation is still being exercised; successful SDK extraction is not yet
proof of working NPU inference. Source download hashes are retained in INSTALL_ROOT/source-sha256.txt.

## XRT source build in progress

Source revisions under investigation:
- amd/xdna-driver: 1d3e56f8cfe2033ddd35b534b8b598e94ea2663b
- pinned XRT submodule: 06c43a7f93fee129d32def1d372e17a8e69c05d2

Additional native build dependencies required on the target: boost, ninja, rapidjson, opencl-headers, ocl-icd,
pybind11. Existing libdrm, protobuf, cmake and development tools are also used.

```bash
cd xdna-driver/xrt/build
./build.sh -npu -opt -noctest -noinit -j 12
```

The initial configure attempts identified missing OpenCL headers and pybind11; installing the native Arch packages
resolved those configure failures. The build and native execution results are recorded below.

## Native execution confirmed (2026-11-12)

XRT 2.27.0 and its matching userspace shim built and installed as native pacman packages using the two new
`PKGBUILD-*-native` recipes. Source revisions are listed above. CMake built the shim with `SKIP_KMOD=ON`.
The original Omarchy kernel driver remains loaded. `xrt-smi examine` reports NPU Strix Halo, aie2p, 6x8, firmware
1.1.2.65. The hardware validation completed successfully: GEMM 51.0 TOPS, command latency 46.0 us, throughput
94682 ops/s. These are XRT microbenchmarks, not HairDeck/model throughput measurements.

The SSH session initially had an 8192 KiB memlock limit; the 64 MiB device mapping failed with EAGAIN. The diagnostic
succeeded with a raised limit for that process:

```bash
sudo prlimit --memlock=unlimited -- bash -c \
  'source /opt/xilinx/xrt/setup.sh >/dev/null && xrt-smi validate --device 0000:c6:00.1'
```

The process-scoped `npu-run.sh` solution below now supports model execution without running inference as root.

The SDK extraction wrapper successfully created its Python 3.12 environment and installed CPU PyTorch. The vendor
1.9 wheel-index endpoint returned HTTP 403 for onnxruntime-vitisai, so dependency resolution failed. The official
bundled .tgz distribution supplies the working wheel used below. The earlier Ubuntu experiment container is stopped;
all results here were obtained on native Arch.

## Real model execution confirmed

AMD's 1.9 quicktest model now runs through Vitis AI on the native Arch stack. The partition log explicitly reports
398 NPU operators, 2 VITIS_EP_CPU operators, one NPU subgraph and `Actually running on NPU 1`. This is a small vendor
model, not yet a HairDeck stage. XRT 2.27 worked for this case despite the SDK's 2.26 reference version.

Two compatibility fixes were needed: clear executable-stack metadata on the private environment's
`onnxruntime/capi/onnxruntime_pybind11_state.so` with `patchelf --clear-execstack`, and a private
`libncurses.so.6 -> /usr/lib/libncursesw.so.6` link. No system library was replaced. The packaged SDK's
`quicktest.py` originally reported success after silently falling back to CPU; always check provider creation and
actual graph placement, not just its exit status.

The new `npu-run.sh` raises the invoking shell's memlock limit through sudo/prlimit, then runs the requested command
as the original user. It avoids running inference as root. `run-sdk.sh` configures private library paths, including
the managed Python shared library; use it with `check-npu-provider.py` and retain the partition log.

The working runtime wheel is `onnxruntime_vitisai-1.30.0-py3-none-linux_x86_64.whl` from AMD's official 1.9 bundle.
It was installed locally with --no-deps plus coloredlogs, flatbuffers, packaging and protobuf; numpy and sympy were
already installed. This is a minimal working inference environment, not a claim that all SDK compiler/quantizer
packages are installed. The full archive remains downloading for the additional model-conversion work.

## BF16 correctness limitation and private XRT comparison

A HairDeck Linear probe with released weights and synthetic input produces half-zero output under the BF16 VAIML
path (relative L2 error 0.70665 for shape 384x768). CPU ONNX agrees with CPU PyTorch; GPU BF16 is also accurate.
Vendor quicktest success therefore does not establish arbitrary BF16 model correctness.

The SDK-matched source branch was built independently: xdna-driver 1.9 at `a1f7a2fe`, pinned XRT
`e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6`, version 2.26.0. Direct CMake with `SKIP_KMOD=ON` built core, coreutil
and the shim. Their shared libraries were copied with symlinks into a private prefix's `lib` directory. Running with
`RYZENAI_XRT_ROOT=/path/to/private/prefix` makes `npu-run.sh` select those libraries without changing pacman packages
or the kernel driver. `/proc/self/maps` confirmed all three private libraries were loaded.

Fresh compilation on 2.26 reproduced the same error with both automatic and vectorized data layout. This rules out
the newer XRT version as the sole explanation; the BF16 path remains experimental and is not enabled in HairDeck.
See the [HairDeck investigation](https://github.com/jethac/HairDeck/blob/rocm/docs/strix-halo-investigation.md) for
the model probe and measurements.
