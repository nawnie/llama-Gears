# Experimental decode expert-cache runtime

This is executable GGML backend code, not a metadata selector or a simulated slot map. It remains an experimental implementation until native build and model acceptance results are recorded. There is no measured speedup claim.

## Execution

The optional `Gears-MoE` ACCEL backend uses the same shared GGML core as llama-server. It claims only eligible single-token `MUL_MAT_ID` operations whose expert weights are already in standard contiguous host memory. CPU misses run native GGML matrix multiplication. Resident hits run a native matrix-multiplication graph on a separate CUDA stream; that graph is submitted before CPU work. Outputs are written back in the original selected-expert order. Model routing weights, activations, attention and final reduction remain in the existing graph.

A byte-bounded LRU owns real backend allocations. A miss becomes eligible for caching on its second observation. At most one eligible expert matrix is uploaded after each routed operation, after all users of resident pointers have completed. The key includes weight/data addresses, type, dimensions and expert ID. Each inference context has its own cache. Weights must remain immutable for that context's lifetime.

This is a conservative initial policy, not a copy of Strata's implementation. It does not implement SSD expert streaming, PLE tables, or predictive scheduling. Allocating small per-operation graphs and copying activations may make it slower than baseline; measure before adoption.

## Preserve native paths

Dense operations, GPU-resident expert tensors, repacked/noncontiguous weights, multi-token prefill, and Prism Q1_0/Q2_0/PQ2_0/PTQ1_0 tensors are not claimed by this backend. These keep their normal backend selection. Native backend capability checks are required for every claimed quantized matrix primitive. Unsupported shapes are rejected before scheduling. An invalid expert ID fails the operation without indexing outside the weight buffer.

Qwen3-Next and Nemotron-H MoE are regression targets, not yet validated model claims. Qwen3.8 Flash-Next uses `qwen4exp`, which this pinned Prism baseline does not implement. This cache does not supply that missing architecture.

## Build separately from installed engines

Use the `gears/v2-moe-runtime` checkout, never an existing Qwen Chat installation directory.

```sh
cmake -S . -B build-gears -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_GEARS_RUNTIME=ON -DLLAMA_GEARS_INSPECT=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-gears --config Release --parallel 4
ctest --test-dir build-gears -C Release -R llama-gears --output-on-failure
```

The runtime option is OFF by default. Only load the plugin built together with the executable and all its GGML libraries. Do not mix binaries from stock llama.cpp, another Prism release or another Gears checkout. The GGML plugin ABI version alone is not a guarantee of tensor-layout compatibility.

## Opt in for a hardware experiment

In a fresh PowerShell process, point to the actual build output (generator-dependent):

```powershell
$env:GGML_BACKEND_PATH = (Resolve-Path 'build-gears/bin/Release/ggml-gears.dll').Path
$env:LLAMA_GEARS_CACHE_MIB = '256'
$env:LLAMA_GEARS_HEADROOM_MIB = '256'
$env:LLAMA_GEARS_DEVICE = 'CUDA0'
```

On Linux the plugin is normally `build-gears/bin/libggml-gears.so`; set `GGML_BACKEND_PATH` to its absolute path. Launch the matching llama-server or llama-bench with an existing MoE GGUF and deliberate CPU expert placement. Inspect that binary's help for `--cpu-moe`, `--n-cpu-moe`, and `--no-repack` variants before choosing a profile. Host experts must use the standard layout; repacked host buffers are intentionally excluded. The runtime does not move all weights into RAM for you or select safe context/offload values automatically.

A 256 MiB cache and 256 MiB reserve are initial test settings, not universal safe settings. The reserve applies to cache admission, not the entire inference working set. In-flight workspace or model allocation can still fail. The cache capacity is per inference context, not process-global; multiple contexts multiply the budget. Unset `GGML_BACKEND_PATH` or use a fresh process with `LLAMA_GEARS_CACHE_MIB=0` to restore native-only execution. Configuration is fixed when the backend is first registered.

## Validation and observability

The standalone cache tests use owned memory and run under AddressSanitizer/UndefinedBehaviorSanitizer. The native runtime test loads this plugin, uses GGML's real scheduler and two real CPU backends, compares F32/Q4_0/Q2_K outputs against the original `ggml_mul_mat_id`, exercises cold/mixed/hot paths and eviction, and rejects invalid IDs. The second CPU stands in for the device so this test does not establish CUDA correctness, concurrency or speed.

The registry exposes `ggml_gears_get_stats` with executed-node, device-lane, CPU-lane, upload, failed-upload, eviction and resident-byte counters. The test-only initializer is compiled only when `LLAMA_BUILD_TESTS=ON`. GPU model acceptance still requires CUDA build, sanitizer checks, actual model output comparison, baseline throughput, peak RAM/VRAM, cancellation and context lifecycle tests.
