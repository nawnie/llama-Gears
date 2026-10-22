# llama-Gears acceptance protocol (Jarvis + Sager)

Use the same immutable GGUF and prompt set for each engine/profile. This is a procedure, **not a record of completed hardware tests**.

## Machines and models

- Jarvis: RTX 4070 Ti Super 16 GB VRAM, i5-13600KF, 40 GB DDR4. Inspect free VRAM and actual RAM clock before launch.
- Sager: RTX 4070 Laptop 8 GB VRAM, i9-13900HX, 32 GB DDR5. Inspect actual memory channels, speed, PCIe link and thermals before attributing differences to RAM.
- Profiles: (1) normal GGUF, (2) Bonsai PQ2_0/PTQ1_0 or mainline Q2_0 group-64, (3) Qwen3-Next MoE, (4) the previously tried Nemotron-H MoE. These exercise three planned gears; Nemotron is a separate regression model within MoE.
- Capture exact model name, quantization, shard list and SHA256 hashes. A model not fitting physical memory is `unviable`, not a measured performance loss.

## Preflight

Build from pinned Prism source with `-DLLAMA_GEARS_INSPECT=ON`, then use `llama-gears-inspect MODEL.gguf`. Do **not** treat `hybrid-moe-candidate` as proof of expert-cache implementation. The output is metadata analysis only.

Use Prism's documented `llama-bench` commands for real performance. Inspect `llama-bench --help` on the installed build before running. Start with one representative model and no speculative decoding.

```sh
# Baseline full GPU offload when model and runtime fit in VRAM
llama-bench -m MODEL.gguf -ngl 99 -p 512 -n 128 -r 3 -o json

# Hybrid baseline: tune n-gpu-layers conservatively; n-cpu-moe is valid for
# this pinned Prism llama-bench build, but its effect depends on architecture.
llama-bench -m MODEL.gguf -ngl 20 -ncmoe 0 -p 512 -n 128 -r 3 -o json
```

The second command is a **candidate**, not an endorsed optimal configuration: some models have different layer counts or placement constraints. Do not transfer numbers from one GGUF family to another.

## Gates

1. **Load:** All model tensors accepted without unsupported quantization or missing kernel messages; no hard page faults or memory exhaustion.
2. **Correctness:** Run fixed seed and small deterministic prompts with identical sampler settings, compare to the unmodified Prism build and evaluate token/output correctness. Gears preflight must not alter inference output at all.
3. **Throughput:** Capture prefill tok/s and decode tok/s from matched `llama-bench` runs. Report each run; summarize median and spread, not only the fastest.
4. **Memory:** Record peak system RAM, NVIDIA VRAM, context length, KV cache allocation and GPU/CPU placement per run.
5. **Latency:** Measure first-token latency separately through `llama-server`; `llama-bench` does not include all application overhead.
6. **Stability:** Test model unload/reload, error handling, cancellation and 4K/16K contexts where the GGUF supports them.
7. **Regression:** Compare Bonsai PQ2_0/PTQ1_0, conventional GGUF and Nemotron against original Prism binaries. No engine replacement before every gate passes.

## Next integration

Implement an optional MoE expert-cache runtime in a **separate feature branch**, reusing Prism/GGML's scheduler and registering CPU cache-miss processing only after inspecting the upstream `aman/moe-cache-cpu-misses` implementation. Keep the original code path selectable and fail closed on unsupported quantization or devices.

Sources: [Prism llama-bench documentation](https://github.com/PrismML-Eng/llama.cpp/blob/prism/tools/llama-bench/README.md), [upstream MoE cache experimental branch](https://github.com/ggml-org/llama.cpp/tree/aman/moe-cache-cpu-misses), and [Prism sources](https://github.com/PrismML-Eng/llama.cpp).
