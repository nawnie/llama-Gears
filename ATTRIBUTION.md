# llama-Gears attribution and provenance

llama-Gears is a research fork of [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), ultimately derived from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The initial fork inherits the Prism `prism` branch at commit `73bfe6cc485554822f80c473a3c0c4f25e80f41b`.

- **ggml and llama.cpp authors:** Inference implementation, model architecture code, GGUF, CPU/GPU backends; original MIT license and copyright notices retained in the repository, including `LICENSE` and `AUTHORS`.
- **PrismML authors and contributors:** Bonsai inference changes, PQ2_0 and PTQ1_0 GGML types, Hadamard operations and CUDA/CPU kernels; original source history and notices retained.
- **llama-Gears contributors:** Metadata-only gear classification and GGUF preflight inspector authored for this experimental branch (2026-10-22).
- **Strata (Niko1221 and contributors):** MIT-licensed architectural reference for expert placement and hybrid MoE execution. No Strata code has been imported.
- **ik_llama.cpp (ikawrakow and contributors):** MIT-licensed research reference for CPU offloading and optimized low-bit inference. No code has been imported.

Keep full applicable original license/copyright notices in redistributions. Before importing a future kernel or data asset, document source URL, exact commit, changed files, modification details and any separate licensing requirements. Model checkpoints have independent licenses. This work is not endorsed by upstream authors or NVIDIA.
