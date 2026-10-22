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

## Flash-Next compatibility boundary

Qwen3.8 Flash-Next GGUF uses `general.architecture=qwen4exp`, **not** `qwen3next`. The pinned Prism baseline does not include the `qwen4exp` loader. For `qwen4exp` and `qwen4exp-mtp`, the inspector reports `unverified` and exits with code 4. `invalid` GGUF metadata exits with code 3; metadata-only classifications exit 0. Unknown architectures also fail closed with exit code 4. This tool does not make the model load. Upstream llama.cpp merged Qwen4Exp support in [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742), but porting it into Prism needs a separate compatibility patch.
