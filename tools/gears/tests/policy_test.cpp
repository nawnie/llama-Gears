#include "llama_gears/policy.hpp"
#include <cstdlib>
#include <iostream>
using namespace llama_gears;
int main() {
    int failures = 0;
    const auto test = [&](const ModelMetadata & m, Gear g, Readiness s) {
        const Decision d = classify(m);
        if (d.gear != g || d.readiness != s) {
            std::cerr << "Routing mismatch for architecture: " << m.architecture << "\n";
            ++failures;
        }
    };
    test({"llama", {12}, 0, 0}, Gear::Standard, Readiness::MetadataOnly);
    test({"qwen35", {142}, 0, 0}, Gear::Prism, Readiness::MetadataOnly);
    test({"qwen35moe", {143}, 0, 0}, Gear::Prism, Readiness::MetadataOnly);
    test({"qwen3next", {42}, 512, 8}, Gear::HybridMoeCandidate, Readiness::MetadataOnly);
    test({"nemotron_h_moe", {10}, 128, 4}, Gear::HybridMoeCandidate, Readiness::MetadataOnly);
    test({"nemotron_h_moe", {10}, 0, 0}, Gear::Standard, Readiness::Unverified);
    test({"qwen3next", {42}, 12, 13}, Gear::Standard, Readiness::Unverified);
    test({"unknown-future", {12}, 0, 0}, Gear::Standard, Readiness::Unverified);
    test({"llama", {999}, 0, 0}, Gear::Standard, Readiness::Invalid);
    test({"llama", {43}, 0, 0}, Gear::Standard, Readiness::Invalid);
    test({"llama", {4}, 0, 0}, Gear::Standard, Readiness::Invalid);
    test({"", {12}, 0, 0}, Gear::Standard, Readiness::Invalid);
    test({"llama", {}, 0, 0}, Gear::Standard, Readiness::Invalid);
    test({"nemotron", {12}, 0, 0}, Gear::Standard, Readiness::MetadataOnly);
    test({"qwen35moe", {12}, 200, 8}, Gear::Standard, Readiness::MetadataOnly);
    if (failures) return EXIT_FAILURE;
    std::cout << "15 metadata routing tests passed\n";
    return EXIT_SUCCESS;
}
