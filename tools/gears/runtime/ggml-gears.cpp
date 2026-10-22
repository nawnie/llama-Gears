#include "ggml-gears.h"
#include "ggml-backend-impl.h"
#include "cache.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using llama_gears::expert_key;
constexpr size_t mib = 1024 * 1024;
constexpr int64_t max_lanes = 64;
static_assert(GGML_BACKEND_API_VERSION == 2, "Revalidate Gears against the new backend ABI");
static ggml_guid gears_guid = {0x95,0xd1,0x27,0x32,0xf3,0x26,0x4c,0xa2,0x8b,0x19,0xda,0x0e,0x46,0x21,0x9a,0x11};

struct context_delete { void operator()(ggml_context * p) const { if (p) ggml_free(p); } };
struct buffer_delete { void operator()(ggml_backend_buffer * p) const { if (p) ggml_backend_buffer_free(p); } };
struct backend_delete { void operator()(ggml_backend * p) const { if (p) ggml_backend_free(p); } };
using context_ptr = std::unique_ptr<ggml_context, context_delete>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_delete>;
using backend_ptr = std::unique_ptr<ggml_backend, backend_delete>;

size_t checked_mul(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) throw std::overflow_error("Gears size overflow");
    return a * b;
}

size_t env_size(const char * name, size_t fallback, size_t max) {
    const char * value = std::getenv(name);
    if (!value) return fallback;
    if (!*value || *value == '-') throw std::invalid_argument(std::string("invalid ") + name);
    errno = 0;
    char * end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (errno || !end || *end || parsed > max) throw std::invalid_argument(std::string("invalid ") + name);
    return static_cast<size_t>(parsed);
}

struct device_config {
    ggml_backend_dev_t gpu = nullptr;
    ggml_backend_dev_t cpu = nullptr;
    size_t budget = 0;
    size_t headroom = 256 * mib;
};

struct device_weight {
    context_ptr context;
    buffer_ptr buffer;
    ggml_tensor * tensor = nullptr;
    size_t bytes() const { return buffer ? ggml_backend_buffer_get_size(buffer.get()) : 0; }
};

struct workset {
    context_ptr context;
    buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> outputs;
    std::vector<int64_t> lanes;

    explicit workset(size_t n) {
        const size_t graph_size = 8 * n + 32;
        const size_t overhead = (4 * n + 8) * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false);
        context.reset(ggml_init({overhead, nullptr, true}));
        if (!context) throw std::bad_alloc();
        graph = ggml_new_graph_custom(context.get(), graph_size, false);
    }

    void append(ggml_tensor * weight, int64_t width, int64_t lane) {
        auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, 1);
        auto * output = ggml_mul_mat(context.get(), weight, input);
        ggml_build_forward_expand(graph, output);
        inputs.push_back(input);
        outputs.push_back(output);
        lanes.push_back(lane);
    }

    bool allocate(ggml_backend_t backend) {
        if (lanes.empty()) return true;
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        return buffer != nullptr;
    }

    void set_inputs(const std::vector<float> & values, int64_t width, int64_t input_lanes) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto offset = static_cast<size_t>((lanes[i] % input_lanes) * width);
            ggml_backend_tensor_set(inputs[i], values.data() + offset, 0, static_cast<size_t>(width) * sizeof(float));
        }
    }

    void get_outputs(std::vector<float> & values, int64_t rows) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto offset = static_cast<size_t>(lanes[i] * rows);
            ggml_backend_tensor_get(outputs[i], values.data() + offset, 0, static_cast<size_t>(rows) * sizeof(float));
        }
    }
};

struct state {
    device_config config;
    ggml_backend_device device = {};
    backend_ptr gpu;
    backend_ptr cpu;
    llama_gears::expert_cache<device_weight> cache;
    ggml_gears_stats stats = {};
    ggml_abort_callback abort_callback = nullptr;
    void * abort_data = nullptr;
    std::mutex mutex;

    explicit state(const device_config & cfg) : config(cfg), cache(cfg.budget) {
        gpu.reset(ggml_backend_dev_init(cfg.gpu, nullptr));
        cpu.reset(ggml_backend_dev_init(cfg.cpu, nullptr));
        if (!gpu || !cpu) throw std::runtime_error("Gears could not initialize compute streams");
    }
    ~state() {
        if (gpu) ggml_backend_synchronize(gpu.get());
        if (cpu) ggml_backend_synchronize(cpu.get());
        cache.clear();
    }
};

struct sync_guard {
    ggml_backend_t gpu;
    ggml_backend_t cpu;
    ~sync_guard() {
        ggml_backend_synchronize(gpu);
        ggml_backend_synchronize(cpu);
    }
};

const char * backend_name(ggml_backend_t) { return "Gears-MoE"; }
const char * device_name(ggml_backend_dev_t) { return "Gears-MoE"; }
const char * device_description(ggml_backend_dev_t) { return "Experimental decode expert cache with CPU misses"; }

enum ggml_backend_dev_type device_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }
void device_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = 0; *total = 0; }
void device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    *props = {};
    props->name = device_name(dev);
    props->description = device_description(dev);
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

ggml_backend_buffer_type_t device_buft(ggml_backend_dev_t dev) {
    auto * cfg = static_cast<device_config *>(dev->context);
    return ggml_backend_dev_buffer_type(cfg->cpu);
}

bool device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto * cfg = static_cast<device_config *>(dev->context);
    return ggml_backend_buft_is_host(buft) && ggml_backend_dev_supports_buft(cfg->cpu, buft);
}

bool shape_supported(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT_ID || op->type != GGML_TYPE_F32) return false;
    const auto * w = op->src[0];
    const auto * x = op->src[1];
    const auto * ids = op->src[2];
    if (!w || !x || !ids || !w->buffer || w->view_src || w->extra) return false;
    if (!ggml_backend_buffer_is_host(w->buffer) || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) return false;
    // Keep Prism-specific transforms on their original path.
    if (w->type == GGML_TYPE_PQ2_0 || w->type == GGML_TYPE_PTQ1_0 ||
        w->type == GGML_TYPE_Q1_0 || w->type == GGML_TYPE_Q2_0) return false;
    if (x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(ids) || !ggml_is_contiguous(op)) return false;
    if (w->ne[0] <= 0 || w->ne[1] <= 0 || w->ne[2] <= 1 || w->ne[3] != 1) return false;
    if (ids->ne[0] <= 0 || ids->ne[0] > max_lanes || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1) return false;
    if (x->ne[0] != w->ne[0] || x->ne[2] != 1 || x->ne[3] != 1) return false;
    if (x->ne[1] != 1 && x->ne[1] != ids->ne[0]) return false;
    return op->ne[0] == w->ne[1] && op->ne[1] == ids->ne[0] && op->ne[2] == 1 && op->ne[3] == 1;
}

// Probe the exact primitive used by each lane, not only the routed operator.
bool device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    if (!shape_supported(op)) return false;
    auto * cfg = static_cast<device_config *>(dev->context);
    ggml_tensor w = *op->src[0];
    ggml_tensor x = *op->src[1];
    ggml_tensor y = *op;
    w.ne[2] = w.ne[3] = 1;
    w.nb[3] = w.nb[2];
    w.op = GGML_OP_NONE;
    w.buffer = nullptr;
    w.data = nullptr;
    w.view_src = nullptr;
    w.extra = nullptr;
    std::fill(std::begin(w.src), std::end(w.src), nullptr);
    x.ne[1] = x.ne[2] = x.ne[3] = 1;
    x.nb[2] = x.nb[3] = x.nb[1];
    x.op = GGML_OP_NONE;
    x.buffer = nullptr;
    x.data = nullptr;
    x.view_src = nullptr;
    x.extra = nullptr;
    std::fill(std::begin(x.src), std::end(x.src), nullptr);
    y.ne[1] = y.ne[2] = y.ne[3] = 1;
    y.nb[2] = y.nb[3] = y.nb[1];
    y.op = GGML_OP_MUL_MAT;
    y.buffer = nullptr;
    y.data = nullptr;
    y.view_src = nullptr;
    y.extra = nullptr;
    std::fill(std::begin(y.src), std::end(y.src), nullptr);
    std::fill(std::begin(y.op_params), std::end(y.op_params), 0);
    y.src[0] = &w;
    y.src[1] = &x;
    return ggml_backend_dev_supports_op(cfg->cpu, &y) && ggml_backend_dev_supports_op(cfg->gpu, &y);
}

expert_key key_for(const ggml_tensor * w, int32_t expert) {
    return {reinterpret_cast<uintptr_t>(w), reinterpret_cast<uintptr_t>(w->data), expert,
            static_cast<int32_t>(w->type), w->ne[0], w->ne[1]};
}

void admit(state & s, const expert_key & key, ggml_tensor * w, int32_t expert) {
    auto resource = std::make_unique<device_weight>();
    resource->context.reset(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}));
    if (!resource->context) throw std::bad_alloc();
    resource->tensor = ggml_new_tensor_2d(resource->context.get(), w->type, w->ne[0], w->ne[1]);
    auto * buft = ggml_backend_get_default_buffer_type(s.gpu.get());
    const size_t raw = ggml_backend_buft_get_alloc_size(buft, resource->tensor);
    const size_t align = ggml_backend_buft_get_alignment(buft);
    if (!align || raw > std::numeric_limits<size_t>::max() - align) return;
    const size_t required = raw + align;
    const auto before = s.cache.size();
    if (!s.cache.make_room(required)) return;
    s.stats.evictions += before - s.cache.size();
    if (ggml_backend_dev_type(s.config.gpu) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(s.config.gpu, &free, &total);
        if (required > free || s.config.headroom > free - required) return;
    }
    resource->buffer.reset(ggml_backend_alloc_ctx_tensors(resource->context.get(), s.gpu.get()));
    if (!resource->buffer) { ++s.stats.upload_failures; return; }
    ggml_backend_buffer_set_usage(resource->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t offset = checked_mul(static_cast<size_t>(expert), w->nb[2]);
    const size_t bytes = ggml_nbytes(resource->tensor);
    const size_t available = ggml_nbytes(w);
    if (offset > available || bytes > available - offset) throw std::runtime_error("Gears expert span outside weight tensor");
    ggml_backend_tensor_set(resource->tensor, static_cast<const char *>(w->data) + offset, 0, bytes);
    const auto resident_before = s.cache.size();
    if (s.cache.insert(key, std::move(resource))) {
        ++s.stats.uploads;
        s.stats.evictions += resident_before + 1 - s.cache.size();
    } else {
        ++s.stats.upload_failures;
    }
}

ggml_status compute_node(state & s, ggml_tensor * op) {
    if (!device_supports_op(&s.device, op)) return GGML_STATUS_FAILED;
    if (s.abort_callback && s.abort_callback(s.abort_data)) return GGML_STATUS_ABORTED;
    auto * w = op->src[0];
    auto * x = op->src[1];
    const int64_t lanes = op->src[2]->ne[0];
    const int64_t width = w->ne[0];
    const int64_t rows = w->ne[1];
    if (!w->data || !x->data || !op->src[2]->data || !op->data) return GGML_STATUS_FAILED;
    std::vector<int32_t> ids(static_cast<size_t>(lanes));
    ggml_backend_tensor_get(op->src[2], ids.data(), 0, ids.size() * sizeof(int32_t));
    for (int32_t expert : ids) if (expert < 0 || expert >= w->ne[2]) return GGML_STATUS_FAILED;
    const size_t input_elements = checked_mul(static_cast<size_t>(width), static_cast<size_t>(x->ne[1]));
    const size_t output_elements = checked_mul(static_cast<size_t>(rows), static_cast<size_t>(lanes));
    checked_mul(input_elements, sizeof(float));
    checked_mul(output_elements, sizeof(float));
    std::vector<float> input(input_elements);
    std::vector<float> output(output_elements);
    ggml_backend_tensor_get(x, input.data(), 0, input.size() * sizeof(float));

    std::map<int32_t, device_weight *> selected;
    std::vector<int32_t> admit_next;
    for (int32_t expert : ids) {
        if (selected.count(expert)) continue;
        const auto key = key_for(w, expert);
        auto * cached = s.cache.find(key);
        selected.emplace(expert, cached);
        if (!cached && s.cache.observe_miss(key)) admit_next.push_back(expert);
    }

    {
        workset gpu(static_cast<size_t>(lanes));
        workset cpu(static_cast<size_t>(lanes));
        for (int64_t lane = 0; lane < lanes; ++lane) {
            if (auto * cached = selected.at(ids[static_cast<size_t>(lane)])) {
                gpu.append(cached->tensor, width, lane);
            } else {
                auto * view = ggml_view_2d(cpu.context.get(), w, width, rows, w->nb[1],
                    checked_mul(static_cast<size_t>(ids[static_cast<size_t>(lane)]), w->nb[2]));
                cpu.append(view, width, lane);
            }
        }
        if (!gpu.allocate(s.gpu.get())) return GGML_STATUS_ALLOC_FAILED;
        if (!cpu.allocate(s.cpu.get())) return GGML_STATUS_ALLOC_FAILED;
        sync_guard complete{s.gpu.get(), s.cpu.get()};
        gpu.set_inputs(input, width, x->ne[1]);
        cpu.set_inputs(input, width, x->ne[1]);
        if (!gpu.lanes.empty()) {
            const auto status = ggml_backend_graph_compute_async(s.gpu.get(), gpu.graph);
            if (status != GGML_STATUS_SUCCESS) return status;
        }
        if (!cpu.lanes.empty()) {
            const auto status = ggml_backend_graph_compute(s.cpu.get(), cpu.graph);
            if (status != GGML_STATUS_SUCCESS) return status;
        }
        ggml_backend_synchronize(s.gpu.get());
        gpu.get_outputs(output, rows);
        cpu.get_outputs(output, rows);
        s.stats.gpu_lanes += gpu.lanes.size();
        s.stats.cpu_lanes += cpu.lanes.size();
    }
    ggml_backend_tensor_set(op, output.data(), 0, output.size() * sizeof(float));
    ++s.stats.nodes;

    // At most one second-touch miss is uploaded after both streams finish.
    if (!admit_next.empty() && s.config.budget) {
        try {
            admit(s, key_for(w, admit_next.front()), w, admit_next.front());
        } catch (const std::bad_alloc &) {
            ++s.stats.upload_failures;
        }
    }
    return GGML_STATUS_SUCCESS;
}

ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto & s = *static_cast<state *>(backend->context);
    const std::lock_guard<std::mutex> lock(s.mutex);
    try {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
            const auto status = compute_node(s, node);
            if (status != GGML_STATUS_SUCCESS) return status;
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        return GGML_STATUS_ALLOC_FAILED;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Gears-MoE: %s\n", error.what());
        return GGML_STATUS_FAILED;
    }
}

void backend_free(ggml_backend_t backend) {
    delete static_cast<state *>(backend->context);
    delete backend;
}
void backend_sync(ggml_backend_t backend) {
    auto & s = *static_cast<state *>(backend->context);
    const std::lock_guard<std::mutex> lock(s.mutex);
    ggml_backend_synchronize(s.gpu.get());
    ggml_backend_synchronize(s.cpu.get());
}

void set_threads(ggml_backend_t backend, int threads) {
    auto & s = *static_cast<state *>(backend->context);
    const std::lock_guard<std::mutex> lock(s.mutex);
    auto * reg = ggml_backend_dev_backend_reg(s.config.cpu);
    auto set = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (set) {
        set(s.cpu.get(), threads);
        if (ggml_backend_dev_type(s.config.gpu) == GGML_BACKEND_DEVICE_TYPE_CPU) set(s.gpu.get(), threads);
    }
}
void set_abort(ggml_backend_t backend, ggml_abort_callback callback, void * data) {
    auto & s = *static_cast<state *>(backend->context);
    const std::lock_guard<std::mutex> lock(s.mutex);
    s.abort_callback = callback;
    s.abort_data = data;
    auto * reg = ggml_backend_dev_backend_reg(s.config.cpu);
    auto set = reinterpret_cast<ggml_backend_set_abort_callback_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback"));
    if (set) set(s.cpu.get(), callback, data);
}
bool get_stats(ggml_backend_t backend, ggml_gears_stats * out) {
    if (!backend || !out || !ggml_guid_matches(ggml_backend_guid(backend), &gears_guid)) return false;
    auto & s = *static_cast<state *>(backend->context);
    const std::lock_guard<std::mutex> lock(s.mutex);
    *out = s.stats;
    out->resident_bytes = s.cache.bytes();
    return true;
}

ggml_backend_t init_backend(ggml_backend_dev_t, const char *);

ggml_backend_device_i device_interface() {
    ggml_backend_device_i iface = {};
    iface.get_name = device_name;
    iface.get_description = device_description;
    iface.get_memory = device_memory;
    iface.get_type = device_type;
    iface.get_props = device_props;
    iface.init_backend = init_backend;
    iface.get_buffer_type = device_buft;
    iface.supports_op = device_supports_op;
    iface.supports_buft = device_supports_buft;
    return iface;
}

ggml_backend_t make_backend(const device_config & config, ggml_backend_reg_t reg) {
    try {
        auto s = std::make_unique<state>(config);
        s->device.iface = device_interface();
        s->device.reg = reg;
        s->device.context = &s->config;
        ggml_backend_i iface = {};
        iface.get_name = backend_name;
        iface.free = backend_free;
        iface.synchronize = backend_sync;
        iface.graph_compute = graph_compute;
        auto result = std::make_unique<ggml_backend>();
        result->guid = &gears_guid;
        result->iface = iface;
        result->device = &s->device;
        result->context = s.release();
        return result.release();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Gears-MoE initialization: %s\n", error.what());
        return nullptr;
    }
}

ggml_backend_t init_backend(ggml_backend_dev_t dev, const char *) {
    return make_backend(*static_cast<device_config *>(dev->context), dev->reg);
}

struct registry_state {
    std::mutex mutex;
    device_config config;
    ggml_backend_device device = {};
    ggml_backend_reg registry = {};
    bool configured = false;
};

registry_state & registry_storage() { static registry_state storage; return storage; }
const char * registry_name(ggml_backend_reg_t) { return "Gears-MoE"; }

size_t device_count(ggml_backend_reg_t reg) {
    auto & s = *static_cast<registry_state *>(reg->context);
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (s.configured) return s.config.gpu && s.config.cpu && s.config.budget ? 1 : 0;
    s.configured = true;
    try {
        s.config.budget = checked_mul(env_size("LLAMA_GEARS_CACHE_MIB", 256, 1048576), mib);
        s.config.headroom = checked_mul(env_size("LLAMA_GEARS_HEADROOM_MIB", 256, 1048576), mib);
        const char * selected = std::getenv("LLAMA_GEARS_DEVICE");
        s.config.cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            auto * parent = ggml_backend_dev_backend_reg(dev);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && parent &&
                std::strcmp(ggml_backend_reg_name(parent), "CUDA") == 0 &&
                (!selected || std::strcmp(ggml_backend_dev_name(dev), selected) == 0)) {
                s.config.gpu = dev;
                break;
            }
        }
        s.device.iface = device_interface();
        s.device.reg = reg;
        s.device.context = &s.config;
        if (!s.config.cpu || !s.config.gpu) std::fprintf(stderr, "Gears-MoE disabled: CPU and CUDA devices are required\n");
    } catch (const std::exception & error) {
        s.config.budget = 0;
        std::fprintf(stderr, "Gears-MoE disabled: %s\n", error.what());
    }
    return s.config.gpu && s.config.cpu && s.config.budget ? 1 : 0;
}
ggml_backend_dev_t get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0 || !device_count(reg)) return nullptr;
    return &static_cast<registry_state *>(reg->context)->device;
}

#ifdef LLAMA_GEARS_RUNTIME_TESTING
ggml_backend_t test_init(ggml_backend_dev_t cpu, size_t budget) {
    if (!cpu || ggml_backend_dev_type(cpu) != GGML_BACKEND_DEVICE_TYPE_CPU) return nullptr;
    device_config config{cpu, cpu, budget, 0};
    return make_backend(config, &registry_storage().registry);
}
#endif

void * get_proc(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) return reinterpret_cast<void *>(set_threads);
    if (std::strcmp(name, "ggml_backend_set_abort_callback") == 0) return reinterpret_cast<void *>(set_abort);
    if (std::strcmp(name, "ggml_gears_get_stats") == 0) return reinterpret_cast<void *>(get_stats);
#ifdef LLAMA_GEARS_RUNTIME_TESTING
    if (std::strcmp(name, "ggml_gears_test_init") == 0) return reinterpret_cast<void *>(test_init);
#endif
    return nullptr;
}

ggml_backend_reg_t gears_reg() {
    auto & s = registry_storage();
    static std::once_flag init;
    std::call_once(init, [&s] {
        s.registry.api_version = GGML_BACKEND_API_VERSION;
        s.registry.iface.get_name = registry_name;
        s.registry.iface.get_device_count = device_count;
        s.registry.iface.get_device = get_device;
        s.registry.iface.get_proc_address = get_proc;
        s.registry.context = &s;
    });
    return &s.registry;
}

} // namespace

GGML_BACKEND_DL_IMPL(gears_reg)
