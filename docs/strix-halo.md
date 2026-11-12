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
