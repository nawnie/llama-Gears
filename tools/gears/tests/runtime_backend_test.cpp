#include "ggml-gears.h"
#include "ggml.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct ctx_delete { void operator()(ggml_context * p) const { if (p) ggml_free(p); } };
struct buf_delete { void operator()(ggml_backend_buffer * p) const { if (p) ggml_backend_buffer_free(p); } };
struct be_delete { void operator()(ggml_backend * p) const { if (p) ggml_backend_free(p); } };
struct sch_delete { void operator()(ggml_backend_sched * p) const { if (p) ggml_backend_sched_free(p); } };
using ctx_ptr = std::unique_ptr<ggml_context, ctx_delete>;
using buf_ptr = std::unique_ptr<ggml_backend_buffer, buf_delete>;
using be_ptr = std::unique_ptr<ggml_backend, be_delete>;
using sch_ptr = std::unique_ptr<ggml_backend_sched, sch_delete>;
int checks = 0;
void require(bool condition, const char * message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
ctx_ptr context() {
    return ctx_ptr(ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true}));
}

void exercise(ggml_backend_t gears, ggml_backend_t reference, ggml_gears_get_stats_fn stats_fn, ggml_type type, size_t budget) {
    constexpr int64_t width = 256, rows = 16, experts = 4, selected = 2;
    auto weights_ctx = context();
    require(weights_ctx != nullptr, "weight context allocation");
    auto * w = ggml_new_tensor_3d(weights_ctx.get(), type, width, rows, experts);
    buf_ptr weights(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), reference));
    require(weights != nullptr, "weight buffer allocation");
    ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> values(static_cast<size_t>(width * rows * experts));
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 31.0f;
    std::vector<unsigned char> packed(ggml_nbytes(w));
    const size_t written = ggml_quantize_chunk(type, values.data(), packed.data(), 0, rows * experts, width, nullptr);
    require(written == packed.size(), "quantized fixture has exact GGML tensor size");
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());

    for (int input_lanes : {1, 2}) {
        auto reference_ctx = context();
        auto * rx = ggml_new_tensor_3d(reference_ctx.get(), GGML_TYPE_F32, width, input_lanes, 1);
        auto * ri = ggml_new_tensor_2d(reference_ctx.get(), GGML_TYPE_I32, selected, 1);
        auto * ry = ggml_mul_mat_id(reference_ctx.get(), w, rx, ri);
        auto * rg = ggml_new_graph_custom(reference_ctx.get(), 128, false);
        ggml_build_forward_expand(rg, ry);
        buf_ptr rb(ggml_backend_alloc_ctx_tensors(reference_ctx.get(), reference));
        require(rb != nullptr, "reference buffer allocation");

        auto candidate_ctx = context();
        auto * x = ggml_new_tensor_3d(candidate_ctx.get(), GGML_TYPE_F32, width, input_lanes, 1);
        auto * ids = ggml_new_tensor_2d(candidate_ctx.get(), GGML_TYPE_I32, selected, 1);
        ggml_set_input(x);
        ggml_set_input(ids);
        auto * y = ggml_mul_mat_id(candidate_ctx.get(), w, x, ids);
        auto * graph = ggml_new_graph_custom(candidate_ctx.get(), 128, false);
        ggml_build_forward_expand(graph, y);
        ggml_backend_t backends[] = {gears, reference};
        sch_ptr scheduler(ggml_backend_sched_new(backends, nullptr, 2, 128, false, false));
        require(scheduler != nullptr, "scheduler initialization");
        require(ggml_backend_sched_alloc_graph(scheduler.get(), graph), "scheduler allocation");
        require(ggml_backend_sched_get_tensor_backend(scheduler.get(), y) == gears, "scheduler selected actual Gears backend");
        std::vector<float> input(static_cast<size_t>(width * input_lanes));
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 13.0f;
        ggml_backend_tensor_set(rx, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        for (const std::vector<int32_t> route : {std::vector<int32_t>{0, 2}, {1, 3}, {0, 0}}) {
            ggml_backend_tensor_set(ri, route.data(), 0, route.size() * sizeof(int32_t));
            ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
            require(ggml_backend_graph_compute(reference, rg) == GGML_STATUS_SUCCESS, "reference compute");
            std::vector<float> expected(static_cast<size_t>(rows * selected)), result(expected.size());
            ggml_backend_tensor_get(ry, expected.data(), 0, expected.size() * sizeof(float));
            for (int round = 0; round < 4; ++round) {
                require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS, "cold, mixed or hot compute");
                ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
                for (size_t i = 0; i < result.size(); ++i) {
                    require(std::isfinite(result[i]) && std::fabs(result[i] - expected[i]) <= 1e-4f * (1.0f + std::fabs(expected[i])), "numeric parity with GGML MUL_MAT_ID");
                }
            }
        }
        const int32_t invalid[] = {0, 4};
        ggml_backend_tensor_set(ids, invalid, 0, sizeof(invalid));
        require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_FAILED, "invalid expert ID rejected without abort");
    }

    auto probe_ctx = context();
    auto * x = ggml_new_tensor_3d(probe_ctx.get(), GGML_TYPE_F32, width, 1, 2);
    auto * ids = ggml_new_tensor_2d(probe_ctx.get(), GGML_TYPE_I32, selected, 2);
    auto * prefill = ggml_mul_mat_id(probe_ctx.get(), w, x, ids);
    require(!ggml_backend_supports_op(gears, prefill), "multi-token prefill stays on native path");
    auto * dense_w = ggml_new_tensor_2d(probe_ctx.get(), GGML_TYPE_F32, width, rows);
    auto * dense_x = ggml_new_tensor_2d(probe_ctx.get(), GGML_TYPE_F32, width, 1);
    auto * dense = ggml_mul_mat(probe_ctx.get(), dense_w, dense_x);
    require(!ggml_backend_supports_op(gears, dense), "dense model path unchanged");
    ggml_gears_stats stats = {};
    require(stats_fn(gears, &stats), "runtime statistics available");
    require(stats.gpu_lanes > 0 && stats.cpu_lanes > 0 && stats.uploads > 0, "both compute paths and real uploads executed");
    require(stats.evictions > 0, "working-set changes evict real backend allocations");
    require(stats.resident_bytes <= budget, "device cache obeys configured capacity");
    std::printf("runtime %s: nodes=%llu device_lanes=%llu cpu_lanes=%llu uploads=%llu evictions=%llu resident_bytes=%llu\n",
        ggml_type_name(type), (unsigned long long) stats.nodes, (unsigned long long) stats.gpu_lanes, (unsigned long long) stats.cpu_lanes,
        (unsigned long long) stats.uploads, (unsigned long long) stats.evictions, (unsigned long long) stats.resident_bytes);
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: runtime-test <ggml-gears plugin>\n"); return 2; }
    try {
        ggml_backend_load_all();
        auto * reg = ggml_backend_load(argv[1]);
        require(reg != nullptr, "runtime plugin loads");
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(cpu_dev != nullptr, "real GGML CPU backend present");
        auto init = reinterpret_cast<ggml_gears_test_init_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_gears_test_init"));
        auto stats = reinterpret_cast<ggml_gears_get_stats_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_gears_get_stats"));
        require(init && stats, "test and statistics API");
        be_ptr cpu(ggml_backend_dev_init(cpu_dev, nullptr));
        require(cpu != nullptr, "reference compute backend initializes");
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_dev), "ggml_backend_set_n_threads"));
        if (set_threads) set_threads(cpu.get(), 2);
        auto set_gears_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(cpu_dev));
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q2_K}) {
            const size_t budget = 2 * (ggml_row_size(type, 256) * 16 + alignment);
            be_ptr candidate(init(cpu_dev, budget));
            require(candidate != nullptr, "candidate compute backend initializes");
            if (set_gears_threads) set_gears_threads(candidate.get(), 2);
            exercise(candidate.get(), cpu.get(), stats, type, budget);
        }
        ggml_quantize_free();
        std::printf("%d checks passed using two real GGML CPU backends; CUDA performance NOT tested\n", checks);
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL after %d checks: %s\n", checks, error.what());
        return EXIT_FAILURE;
    }
}
