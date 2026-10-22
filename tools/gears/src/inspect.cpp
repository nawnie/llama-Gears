#include "llama_gears/policy.hpp"
#include "gguf.h"

#include <cstdio>
#include <cstdint>
#include <string>

namespace {
std::string string_value(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) return {};
    const char * value = gguf_get_val_str(ctx, id);
    return value ? value : "";
}
uint64_t unsigned_value(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return 0;
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
        default: return 0;
    }
}
std::string escape_json(const std::string & s) {
    std::string result;
    for (unsigned char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    result += buffer;
                } else result += static_cast<char>(c);
        }
    }
    return result;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: llama-gears-inspect <model.gguf>\n");
        return 2;
    }
    gguf_context * ctx = gguf_init_from_file(argv[1], { true, nullptr });
    if (!ctx) {
        std::fprintf(stderr, "Failed to parse GGUF metadata\n");
        return 2;
    }
    llama_gears::ModelMetadata metadata;
    metadata.architecture = string_value(ctx, "general.architecture");
    metadata.expert_count = unsigned_value(ctx, metadata.architecture + ".expert_count");
    metadata.active_experts = unsigned_value(ctx, metadata.architecture + ".expert_used_count");
    const int64_t count = gguf_get_n_tensors(ctx);
    if (count > 0) metadata.tensor_types.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        metadata.tensor_types.push_back(static_cast<int32_t>(gguf_get_tensor_type(ctx, i)));
    }
    const llama_gears::Decision result = llama_gears::classify(metadata);
    std::printf("{\"architecture\":\"%s\",\"gear\":\"%s\",\"readiness\":\"%s\","
                "\"reason\":\"%s\",\"tensor_count\":%lld,\"expert_count\":%llu,"
                "\"active_experts\":%llu,\"runtime_changed\":false}\n",
        escape_json(metadata.architecture).c_str(), llama_gears::gear_name(result.gear),
        llama_gears::readiness_name(result.readiness), escape_json(result.reason).c_str(),
        static_cast<long long>(count),
        static_cast<unsigned long long>(metadata.expert_count),
        static_cast<unsigned long long>(metadata.active_experts));
    gguf_free(ctx);
    return result.readiness == llama_gears::Readiness::Invalid ? 3 : 0;
}
