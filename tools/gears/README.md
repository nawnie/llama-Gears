# llama-Gears opt-in GGUF preflight

This experimental inspector reads actual GGUF metadata, categorizes a model as `standard`, `prism`, or `hybrid-moe-candidate`, and produces JSON. It does NOT modify model loading, runtime flags, GGML, CUDA kernels, caching or model allocations. The current Prism llama-server remains unchanged.

Build on a real checkout with:

```sh
cmake -S . -B build-gears -DLLAMA_GEARS_INSPECT=ON -DGGML_CUDA=ON
cmake --build build-gears --config Release -j 4
# Run the produced llama-gears-inspect executable on a GGUF path
```

The normal build leaves the tool disabled. `prism` detects private tensor IDs 142 / 143. Conventional Bonsai Q2_0 remains with the standard loader. `hybrid-moe-candidate` requires the exact architecture `qwen3next` or `nemotron_h_moe` and consistent positive expert counts. It does NOT activate a Strata-like MoE cache; that work is still pending.

A metadata decision is not proof of model load, numerical correctness, kernel support or performance. For split GGUFs the inspector sees only the file supplied; llama.cpp validates the complete model.
