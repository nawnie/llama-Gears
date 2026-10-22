#include "llama_gears/policy.hpp"

#include <algorithm>

namespace llama_gears {
namespace {
bool is_known_architecture(const std::string & arch) {
    static const char * const known[] = {
        "llama", "qwen", "qwen2", "qwen2moe", "qwen3", "qwen3moe",
        "qwen3next", "qwen35", "qwen35moe", "nemotron", "nemotron_h",
        "nemotron_h_moe", "gemma", "gemma2", "gemma3", "gemma4",
        "gpt-oss", "deepseek2", "deepseek32", "deepseek4", "glm4moe",
        "glm-dsa", "kimi-linear", "kimi-k3"
    };
    return std::any_of(std::begin(known), std::end(known), [&](const char * value) {
        return arch == value;
    });
}
bool is_known_tensor_type(int32_t type) {
    switch (type) {
        case 4: case 5: case 31: case 32: case 33:
        case 36: case 37: case 38: return false;
        default: break;
    }
    return (type >= 0 && type <= 42) || type == 142 || type == 143;
}
} // namespace

const char * gear_name(Gear gear) {
    switch (gear) {
        case Gear::Standard: return "standard";
        case Gear::Prism: return "prism";
        case Gear::HybridMoeCandidate: return "hybrid-moe-candidate";
    }
    return "standard";
}
const char * readiness_name(Readiness status) {
    switch (status) {
        case Readiness::MetadataOnly: return "metadata-only";
        case Readiness::Unverified: return "unverified";
        case Readiness::Invalid: return "invalid";
    }
    return "invalid";
}
Decision classify(const ModelMetadata & model) {
    if (model.architecture.empty() || model.tensor_types.empty()) {
        return { Gear::Standard, Readiness::Invalid, "Missing GGUF architecture or tensor metadata" };
    }
    bool prism = false;
    for (int32_t type : model.tensor_types) {
        if (!is_known_tensor_type(type)) {
            return { Gear::Standard, Readiness::Invalid, "Tensor type is not defined by pinned Prism GGML" };
        }
        prism |= type == 142 || type == 143;
    }
    if (!is_known_architecture(model.architecture)) {
        return { Gear::Standard, Readiness::Unverified,
            "Architecture not in Gears preflight allowlist; leave execution to original llama.cpp loader" };
    }
    if (prism) {
        return { Gear::Prism, Readiness::MetadataOnly,
            "Prism-private GGUF tensor type detected; use original Prism kernels" };
    }
    const bool targeted_moe = model.architecture == "qwen3next" ||
                              model.architecture == "nemotron_h_moe";
    if (targeted_moe && model.expert_count > 0 &&
        model.active_experts > 0 && model.active_experts <= model.expert_count) {
        return { Gear::HybridMoeCandidate, Readiness::MetadataOnly,
            "MoE profile identified; experimental expert cache is not enabled" };
    }
    if (targeted_moe) {
        return { Gear::Standard, Readiness::Unverified,
            "Target MoE lacks complete expert topology; retain baseline execution" };
    }
    return { Gear::Standard, Readiness::MetadataOnly,
        "Known architecture; original llama.cpp loader remains authoritative" };
}
} // namespace llama_gears
