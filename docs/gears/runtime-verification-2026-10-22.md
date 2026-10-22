# Gears runtime verification - 2026-10-22

## Exact tested source

- Repository: nawnie/llama-Gears
- Branch: gears/v2-moe-runtime
- Tested implementation commit: `5f5943412e41b3b5a8d4d5934aa5805cf90604c8`
- Base: `2908a236b51a5e69bbba58bc78c656b6db2a964d`
- Preserved Prism branch: `73bfe6cc485554822f80c473a3c0c4f25e80f41b`
- This receipt is a documentation-only successor; it does not change the tested runtime.

## Observed build and test evidence

[Gears runtime CPU verification run 39850388720](https://github.com/nawnie/llama-Gears/actions/runs/39850388720) completed successfully on both matrix platforms. The job logs were retrieved and inspected.

| Environment | Compiler | Native build | CTest | Python benchmark tests |
| --- | --- | --- | --- | --- |
| Ubuntu 26.04.1 | GNU 15.2.0 | PASS | 3/3 PASS | 4/4 PASS |
| Windows Server 2025 | MSVC 19.51.36260.0 | PASS | 3/3 PASS | 4/4 PASS |

Job IDs: Ubuntu `120074645744`; Windows `120074645205`.

Native targets built: `ggml-gears`, `llama-gears-runtime-test`, `llama-gears-runtime-cache-test`, `llama-gears-policy-test`, and their GGML dependencies. This was not a full llama-server build. CUDA was OFF.

Verification commands:

```sh
cmake --build build-gears --config Release --parallel 2 --target ggml-gears llama-gears-runtime-test llama-gears-runtime-cache-test llama-gears-policy-test
ctest --test-dir build-gears -C Release -R llama-gears --output-on-failure
python -m unittest discover -s tools/gears/tests -p test_bench.py -v
```

The runtime test executes the real GGML scheduler, backend allocation, copying and matrix multiplication. It compares F32, Q4_0 and Q2_K routed output against native MUL_MAT_ID, including cold/mixed/hot cache states, repeated expert IDs, eviction and invalid IDs. Two real CPU backends are used; one substitutes for a device. This does not establish CUDA behavior or CPU/GPU concurrency.

The standalone cache resource-ownership test additionally passed locally with GCC 14.2.0: 27 checks, zero failures, exit 0 in Release-style compilation and in AddressSanitizer/UndefinedBehaviorSanitizer compilation with leak checking. This sanitizer result covers the standalone cache, not the full GGML runtime.

The new Gears workflow is green. An inherited build-cann.yml workflow also reported a failure for the same commit; it was not diagnosed or modified in this work. Do not describe repository-wide CI as fully passing.

## Implementation boundaries

The plugin owns real cached expert matrices and dispatches cache-hit device work before CPU miss work. It is opt-in, uses one shared GGML core and does not replace installed engines. Normal dense operations, multi-token prefill, GPU-resident or repacked weights and Prism-specific Q1_0/Q2_0/PQ2_0/PTQ1_0 formats keep native selection. This is not a complete Strata port or an SSD streaming implementation.

## Not yet verified

- CUDA-enabled build and actual NVIDIA execution.
- Whole-model loading and generation for Bonsai, conventional GGUF, Qwen3-Next or the prior Nemotron checkpoint.
- Full-server integration, cancellation and model/context lifecycle under GPU load.
- Throughput, first-token latency, peak VRAM/RAM, CUDA overlap or speedup on Jarvis/Sager.
- Qwen3.8 Flash-Next: its qwen4exp architecture remains absent from this pinned Prism baseline. The cache cannot replace that architecture port.

Next acceptance gate: a matching CUDA build and numerical comparison on Jarvis, then matched native-versus-cache model tests. Keep the original Prism binaries and compare identical model hashes, context, sampler and offload settings. Do not promote the experiment to Qwen Chat production based on CPU CI alone.
