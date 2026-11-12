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
resolved those configure failures. Compilation is in progress. The userspace-only XDNA plugin build and real NPU
inference remain to be completed. No inference/performance claims are made by this document yet.

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

A permanent per-user memlock setup is still to be configured before normal unprivileged model execution.

The SDK extraction wrapper successfully created its Python 3.12 environment and installed CPU PyTorch. The vendor
1.9 wheel-index endpoint returned HTTP 403 for onnxruntime-vitisai, so dependency resolution failed. The official
bundled .tgz distribution is downloading as an alternative source for those wheels. SDK-based model inference is
not yet verified. The earlier Ubuntu experiment container is stopped; all results here were obtained on native Arch.
