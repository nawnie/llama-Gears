#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llama_gears {

enum class Gear { Standard, Prism, HybridMoeCandidate };
enum class Readiness { MetadataOnly, Unverified, Invalid };

struct ModelMetadata {
    std::string architecture;
    std::vector<int32_t> tensor_types;
    uint64_t expert_count = 0;
    uint64_t active_experts = 0;
};

struct Decision {
    Gear gear = Gear::Standard;
    Readiness readiness = Readiness::Unverified;
    std::string reason;
};

Decision classify(const ModelMetadata & model);
const char * gear_name(Gear gear);
const char * readiness_name(Readiness readiness);
} // namespace llama_gears
