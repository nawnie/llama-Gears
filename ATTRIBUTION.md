# llama-Gears attribution and provenance

llama-Gears is a research fork of [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), ultimately derived from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The initial fork inherits the Prism `prism` branch at commit `73bfe6cc485554822f80c473a3c0c4f25e80f41b`.

- **ggml and llama.cpp authors:** Inference implementation, model architecture code, GGUF, CPU/GPU backends, graph scheduler and backend plugin interfaces. Original MIT license, copyright notices and AUTHORS are retained unchanged.
- **PrismML authors and contributors:** Bonsai inference changes, PQ2_0 and PTQ1_0 GGML types, Hadamard operations and CUDA/CPU kernels. Original source history and notices are retained.
- **llama-Gears contributors:** Metadata classification, benchmark runner, and the optional Gears-MoE backend, resource-owning cache and tests, authored for this experimental fork. Runtime additions start from `2908a236b51a5e69bbba58bc78c656b6db2a964d` and reuse the existing GGML APIs and kernels. Meaningful AI assistance is disclosed in the commit history.
- **Strata (Niko1221 and contributors):** MIT-licensed architectural reference for expert placement and hybrid MoE execution. No Strata implementation is copied or integrated here.
- **ik_llama.cpp (ikawrakow and contributors):** MIT-licensed research reference for CPU offloading and low-bit inference. No code has been imported.
- **Upstream llama.cpp MoE-cache contributors:** The `aman/moe-cache-cpu-misses` branch informed the research comparison. This backend uses its own bounded LRU/second-touch policy; it does not incorporate that branch's source code or claim its results.

References inspected include `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend-reg.cpp`, `ggml/src/ggml-backend.cpp`, and `src/llama-context.cpp` at the pinned Gears base. Original MIT terms continue to apply to this repository; existing per-file notices must remain.

Before importing any future kernel or data asset, document source URL, exact commit, paths, modifications and separate licensing requirements. Model checkpoints have independent licenses. No endorsement by upstream authors, model publishers or NVIDIA is implied.
