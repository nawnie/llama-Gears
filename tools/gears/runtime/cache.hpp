#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <iterator>
#include <map>
#include <memory>
#include <tuple>

namespace llama_gears {

using expert_key = std::tuple<uintptr_t, uintptr_t, int64_t, int32_t, int64_t, int64_t>;

template<class Resource> class expert_cache {
    struct entry {
        expert_key key;
        std::unique_ptr<Resource> resource;
    };
    using entries = std::list<entry>;
    using history = std::list<expert_key>;

    const size_t capacity_;
    const size_t history_limit_;
    size_t used_ = 0;
    entries lru_;
    std::map<expert_key, typename entries::iterator> resident_;
    history ghosts_;
    std::map<expert_key, typename history::iterator> seen_;

public:
    explicit expert_cache(size_t capacity, size_t history_limit = 4096)
        : capacity_(capacity), history_limit_(history_limit) {}
    expert_cache(const expert_cache &) = delete;
    expert_cache & operator=(const expert_cache &) = delete;

    Resource * find(const expert_key & key) {
        const auto found = resident_.find(key);
        if (found == resident_.end()) return nullptr;
        lru_.splice(lru_.end(), lru_, found->second);
        return found->second->resource.get();
    }

    bool observe_miss(const expert_key & key) {
        const auto found = seen_.find(key);
        if (found != seen_.end()) {
            ghosts_.splice(ghosts_.end(), ghosts_, found->second);
            return true;
        }
        if (!history_limit_) return false;
        ghosts_.push_back(key);
        try {
            seen_.emplace(key, std::prev(ghosts_.end()));
        } catch (...) {
            ghosts_.pop_back();
            throw;
        }
        if (ghosts_.size() > history_limit_) {
            seen_.erase(ghosts_.front());
            ghosts_.pop_front();
        }
        return false;
    }

    // Call after all users of resident pointers have completed.
    bool make_room(size_t bytes) {
        if (!bytes || bytes > capacity_) return false;
        while (used_ > capacity_ - bytes) {
            const auto oldest = lru_.begin();
            used_ -= oldest->resource->bytes();
            resident_.erase(oldest->key);
            lru_.erase(oldest);
        }
        return true;
    }

    bool insert(const expert_key & key, std::unique_ptr<Resource> resource) {
        if (!resource || !resource->bytes() || resource->bytes() > capacity_) return false;
        const auto found = resident_.find(key);
        if (found != resident_.end()) return false;
        const size_t size = resource->bytes();
        if (!make_room(size)) return false;
        lru_.push_back({key, std::move(resource)});
        try {
            resident_.emplace(key, std::prev(lru_.end()));
        } catch (...) {
            lru_.pop_back();
            throw;
        }
        used_ += size;
        const auto ghost = seen_.find(key);
        if (ghost != seen_.end()) {
            ghosts_.erase(ghost->second);
            seen_.erase(ghost);
        }
        return true;
    }

    size_t bytes() const { return used_; }
    size_t size() const { return resident_.size(); }

    void clear() {
        resident_.clear();
        lru_.clear();
        seen_.clear();
        ghosts_.clear();
        used_ = 0;
    }
};

} // namespace llama_gears
