#pragma once

#include "ggml-backend.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_gears_stats {
    uint64_t nodes;
    uint64_t gpu_lanes;
    uint64_t cpu_lanes;
    uint64_t uploads;
    uint64_t upload_failures;
    uint64_t evictions;
    uint64_t resident_bytes;
};

typedef bool (*ggml_gears_get_stats_fn)(ggml_backend_t, struct ggml_gears_stats *);
// Exposed through the registry only when the runtime test target is enabled.
typedef ggml_backend_t (*ggml_gears_test_init_fn)(ggml_backend_dev_t, size_t);

#ifdef __cplusplus
}
#endif
