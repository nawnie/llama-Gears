#include "cache.hpp"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

struct allocation {
    static int alive;
    std::vector<unsigned char> payload;
    explicit allocation(size_t bytes) : payload(bytes, 7) { ++alive; }
    ~allocation() { --alive; }
    size_t bytes() const { return payload.size(); }
};
int allocation::alive = 0;

int main() {
    int failures = 0;
    int checks = 0;
    auto check = [&](bool condition, const char * message) {
        ++checks;
        if (!condition) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
    };
    using llama_gears::expert_key;
    const expert_key a{1, 10, 0, 0, 4, 4}, b{1, 10, 1, 0, 4, 4};
    const expert_key c{2, 20, 0, 0, 4, 4}, changed_data{1, 11, 0, 0, 4, 4};
    {
        llama_gears::expert_cache<allocation> cache(128, 2);
        check(!cache.observe_miss(a), "first use stays on CPU");
        check(cache.observe_miss(a), "second use is eligible for admission");
        check(cache.make_room(64), "64-byte resident fits");
        check(cache.insert(a, std::make_unique<allocation>(64)), "insert real allocation");
        check(cache.bytes() == 64 && cache.size() == 1, "resident accounting");
        check(cache.find(a) && cache.find(a)->payload[0] == 7, "resident bytes accessible");
        check(!cache.find(changed_data), "a changed host address is a different key");
        check(cache.insert(b, std::make_unique<allocation>(64)), "two experts fit");
        check(cache.find(a) != nullptr, "touch the first expert");
        check(cache.make_room(64), "evict before device allocation");
        check(cache.find(b) == nullptr && cache.find(a), "evict least recently used expert");
        check(allocation::alive == 1, "eviction frees owned memory");
        check(cache.insert(c, std::make_unique<allocation>(64)), "insert after eviction");
        check(cache.bytes() == 128, "cache obeys byte budget");
        check(!cache.make_room(129), "oversized allocation refused");
        check(!cache.insert(b, std::make_unique<allocation>(129)), "oversized resource rejected");
        check(cache.bytes() == 128 && allocation::alive == 2, "rejection frees candidate without losing residents");
        cache.clear();
        check(cache.bytes() == 0 && cache.size() == 0 && allocation::alive == 0, "clear releases every allocation");
        check(!cache.observe_miss(a), "clear resets admission history");
        check(!cache.observe_miss(b), "bounded history second entry");
        check(!cache.observe_miss(c), "bounded history third entry");
        check(!cache.observe_miss(a), "old ghost entry was evicted");
    }
    check(allocation::alive == 0, "destruction leaks no allocations");
    {
        llama_gears::expert_cache<allocation> cache(0);
        check(!cache.make_room(1), "zero cache leaves work on CPU");
        check(!cache.insert(a, std::make_unique<allocation>(1)), "zero cache refuses weights");
        check(!cache.insert(a, std::make_unique<allocation>(0)), "zero-byte resource rejected");
    }
    check(allocation::alive == 0, "zero cache releases rejected allocations");
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
