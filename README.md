# llama-Gears

An experimental fork of [PrismML llama.cpp](https://github.com/PrismML-Eng/llama.cpp), derived from [llama.cpp](https://github.com/ggml-org/llama.cpp) and [GGML](https://github.com/ggml-org/ggml). One inference engine, one shared GGML core, optional execution modes.

## Current branch: experimental MoE runtime

`gears/v2-moe-runtime` adds an opt-in `Gears-MoE` backend. Eligible host-resident expert matrices can be cached in bounded device allocations; cache hits are submitted to the device and misses are computed on the CPU using the existing GGML kernels. This is runtime source code, not just a policy simulator, but it is not a verified production engine or a proven performance improvement.

Normal dense inference, Prism/Bonsai-specific formats, GPU-resident tensors and multi-token prefill retain native backend selection. No installed engine is overwritten. The `prism` branch remains the baseline. Qwen3.8 Flash-Next (`qwen4exp`) still requires an architecture port; Qwen3-Next is a different architecture.

Read the [runtime implementation, build steps and limits](tools/gears/runtime/README.md), [metadata inspector and benchmark tools](tools/gears/README.md), and [acceptance protocol](tools/gears/VALIDATION.md). Build and load the optional backend only from the same checkout as the executable and its GGML libraries.

## Attribution

The underlying inference engine, model implementations, tensor library and device kernels are the work of the llama.cpp/GGML authors and PrismML contributors. Their original licenses, authorship and Git history are retained. The Gears backend is an additional experimental module; it does not claim ownership of that inherited work. See [ATTRIBUTION.md](ATTRIBUTION.md), [LICENSE](LICENSE), and [AUTHORS](AUTHORS).

The previous README, including Prism's full inherited documentation, is preserved verbatim in [README.PRISM.md](README.PRISM.md). That archived file includes earlier Gears status notes; this README and the runtime documentation describe the current branch. There is no upstream or NVIDIA endorsement.
