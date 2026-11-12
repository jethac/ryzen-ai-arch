# Ryzen AI 1.9 BF16 output truncation: diagnosis and working mitigation

On the native Arch Strix Halo, some BF16 Gemm shapes lose half their output channels. The failure mechanism is a
compiler tiling/code-generation mismatch: the tiler divides output channels across two compute stamps (groups of
AIE cores), while the generated graph contains only stamp 0. Constraining compilation to **`tp_size: 1`** produces
the complete output. This is a verified integration workaround, not a patch to AMD's proprietary compiler.

## Reproduce without HairDeck or model downloads

Requires the native SDK environment described in [strix-halo.md](strix-halo.md), including `onnx` and `numpy`.
The probe generates a single ONNX Gemm with identity weights, zero bias and exactly representable positive integer
inputs. CPU ORT must match NumPy exactly. The probe checks that a VAIML partition exists and saves inputs, outputs,
ONNX, configuration, numerical diagnostics and loaded XRT library paths.

```bash
SDK=$HOME/hairdeck-npu/native/runtime/opt/amd/ryzenai/1.9.0
PYTHON=$HOME/hairdeck-npu/native/venv/bin/python

# Baseline: exits 2; columns 384..767 are zero in every row.
bash scripts/run-sdk.sh "$SDK" "$PYTHON" scripts/repro-bf16-linear.py \
  --root "$HOME/hairdeck-npu/repro-baseline" --m 128 --k 768 --n 768 --keep-artifacts

# Workaround: exits 0; every output equals the CPU reference exactly.
bash scripts/run-sdk.sh "$SDK" "$PYTHON" scripts/repro-bf16-linear.py \
  --root "$HOME/hairdeck-npu/repro-single-stamp" --m 128 --k 768 --n 768 --keep-artifacts --tp-size 1
```

Use separate cache directories for different compiler configurations. Existing compiled caches must not be reused
when introducing this workaround. `--keep-artifacts` sets `keep_outputs=true` and `enable_temp_file_io_in_mem=false`
to retain the compiler intermediates. These are diagnostic/internal compiler options and may change between SDKs.

## Evidence locating the defect

1. The identical graph structure works at M=32, K=N=768, and at M=384, K=N=64. The defect reproduces at M=128 or
   M=384 with K=N=768. This is shape-dependent; it is not an inability to perform BF16 arithmetic generally.
2. `trace-xrt-buffers.cpp` interposes only `xrt::bo::sync` in the diagnostic process. The 196608-byte input buffer
   for M=128, K=768 decodes from BF16 to the exact CPU input. The returned 196608-byte output buffer already has
   zeroes in columns 384..767, and decodes exactly to ORT's output. Host input/output conversion is not introducing
   the missing channels. Both transfers use the expected full buffer size.
3. The failing `mkt_gemm_overrides.json` requests `stamps: 2`, `stamp_split: N`. The DSE MLIR records
   `stamp_size = 2`, but `stamp_group_size = 1`. Its output L2 tiles cover 192 columns, with two iterations per
   stamp: 384 columns per stamp.
4. Generated `backend/multi_layer_overlay.h` contains only `single_layer_overlay<4,4,0> flexml_layer_0`, weights
   and data connections for `stamp_0`, and calls to `mini_stamp_0`. There is no second compute stamp. The output
   buffer is nevertheless declared as 768 columns. This accounts for exactly the missing half.
5. With `tp_size=1`, the tiler chooses one stamp, 256-column L2 tiles and three iterations: all 768 columns are
   covered by the generated graph. The identity probe becomes bitwise equal to the reference.

This identifies the failing boundary between the compiler's tiling plan and emitted graph. It does not identify
the exact implementation line inside the closed compiler; the relevant 1.9 backend source is not in the public
`amd/vaip` checkout examined here. XRT 2.27 and SDK-matched source-built 2.26 both reproduced the original defect.
An experimental `backend_args=use-L1L2-connections-overlay-api=true` was accepted but still emitted only stamp 0;
it is not a fix. No kernel driver, firmware or system library was replaced for this mitigation.

Optional transfer tracing (diagnostic data only; do not enable globally):

```bash
c++ -shared -fPIC -O2 scripts/trace-xrt-buffers.cpp -ldl -o /tmp/trace-xrt-buffers.so
mkdir -p /tmp/bf16-transfers
XRT_TRACE_DIR=/tmp/bf16-transfers LD_PRELOAD=/tmp/trace-xrt-buffers.so \
  bash scripts/run-sdk.sh "$SDK" "$PYTHON" scripts/repro-bf16-linear.py \
  --root "$HOME/hairdeck-npu/repro-baseline" --m 128 --k 768 --n 768 --keep-artifacts
```

The interposer targets the tested XRT C++ ABI. `dir0` is host-to-device, `dir1` device-to-host; filenames record
byte counts and offsets. It captures weights as well as activations: treat traces from private models accordingly.

## Applying the workaround

Pass [configs/vaiml-bf16-single-stamp.json](../configs/vaiml-bf16-single-stamp.json) as the VitisAI provider's
`config_file`, with a new `cache_dir` and `enable_cache_file_io_in_mem: "0"`. It preserves the model operation and
dimensions while changing the compiler's work partitioning. It does not synthesize missing channels, halve the
model, run the matrix multiplication on CPU, or reinterpret a failed output as success.

The HairDeck experimental `tools/rocm/bench_npu_mlp.py` now defaults to `--tp-size 1`, includes the setting in its
cache name, and exits with an error if relative L2 exceeds 0.02 against the FP32 reference. `--tp-size 0` reproduces
the original automatic choice. The 2% check is an explicit acceptance bound for this probe, not a universal
model-quality guarantee or permission to enable arbitrary NPU models.

## Real-weight verification and limits

With the original released HairDeck weights and saved seeded activations, the fixed Linear (384x768 to 3072)
has relative L2 0.00944; the original three-dimensional MLP (2x192x768, hidden size 3072, tanh GELU) has relative
L2 0.00807. All outputs are finite and no channels are entirely zero. The compiler selects the
`GemmBfp16WBf16ABf16` kernel with BFP16 weight packing, so this remaining deviation must not be described as pure
BF16 activation rounding. The deterministic identity reproducer, whose values are exactly representable, has
zero numerical error with the workaround.

Correctness was checked before recording timings. The corrected NPU MLP remains slower than the resident ROCm
GPU for this workload; the production HairDeck demo stays on ROCm. This establishes a working NPU probe, not
end-to-end HairDeck quality or a speed improvement. General two-stamp compilation still needs an upstream fix.

Version details, library SHA-256 hashes, shape results and isolated probe timings are retained in
[bf16-stamp-mismatch.json](evidence/bf16-stamp-mismatch.json). Local detailed artifacts and XRT transfer dumps remain
under `~/hairdeck-npu/repro-debug`; the fixed compiler intermediates are under `~/hairdeck-npu/repro-tp1`.


## Follow-up: the workaround does not guarantee whole-model compatibility

A later HairDeck BiRefNet experiment used this same native Arch runtime and `tp_size=1`. The deployed
768px model exported and passed CPU comparisons, but compiler tiling failures caused every subgraph
to fall back to CPU. ORT still listed the VitisAI provider; execution profiling contained only CPU events.
A captured real decoder block reproduces `transpose4d failed to find tiling solution for perm13`
with dimensions `[9,2,48,48]`. An equivalent row-packed sampling layout compiles for NPU but fails at
execution with `ERT_CMD_STATE_TIMEOUT`. The already-validated attention probe still runs correctly
after that timeout, so the device/runtime is not generally unavailable.

These are additional unresolved graph/compiler/runtime limitations, not a recurrence proved to share
the missing-stamp root cause. No kernel, firmware or global library change was made to pursue them.
HairDeck's [continued optimization notes](https://github.com/jethac/HairDeck/blob/rocm/docs/continued-optimization.md)
retain the exact reduced export scripts, numerical comparisons and rejected runtime results.
Whole-stage offload remains disabled; no NPU speedup or GPU/NPU-overlap result is claimed.
