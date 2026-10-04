#if defined(__PROSPERO__)
#include <ps5platform/shm.h>
#include <signal.h>
#include <sched.h>
#include <atomic>
#include <mutex>

namespace {

constexpr size_t kPs5PtChunk = 64 * 1024;
constexpr size_t kPs5PtPool = 16 * 1024 * 1024;
constexpr uint8_t kPs5PtEmpty = 0;
constexpr uint8_t kPs5PtZero = 1;
constexpr uint8_t kPs5PtMoving = 2;
constexpr uint8_t kPs5PtBacked = 3;

struct Ps5PageTable {
    uintptr_t base = 0;
    size_t bytes = 0;
    std::atomic<uint8_t>* state = nullptr;
};

std::mutex g_ps5_pt_mutex;
Ps5PageTable g_ps5_pt[8];
std::atomic<int> g_ps5_pt_count{0};
struct ps5_shm g_ps5_zero{};
struct ps5_shm g_ps5_pool{};
size_t g_ps5_pool_used = kPs5PtPool;
std::atomic<uint64_t> g_ps5_pt_backed{0};
std::atomic<uint64_t> g_ps5_pt_zero_faults{0};
struct sigaction g_ps5_old_segv{};
struct sigaction g_ps5_old_bus{};

bool ps5_zero_object(struct ps5_shm* shm, size_t bytes) {
    if (ps5_shm_create(bytes, shm) != 0) return false;
    void* scratch = nullptr;
    if (ps5_shm_map(shm, 0, bytes, nullptr, PS5_SHM_READ | PS5_SHM_WRITE, 0, &scratch) != 0) {
        ps5_shm_destroy(shm);
        return false;
    }
    std::memset(scratch, 0, bytes);
    ps5_shm_unmap(scratch, bytes, 0);
    return true;
}

Ps5PageTable* ps5_pt_find(uintptr_t addr) {
    int count = g_ps5_pt_count.load(std::memory_order_acquire);
    for (int i = 0; i < count; i++) {
        Ps5PageTable& t = g_ps5_pt[i];
        if (addr >= t.base && addr < t.base + t.bytes) return &t;
    }
    return nullptr;
}

bool ps5_pt_back_locked(Ps5PageTable& t, size_t chunk) {
    std::atomic<uint8_t>& state = t.state[chunk];
    uint8_t current = state.load(std::memory_order_acquire);
    if (current == kPs5PtBacked) return true;
    if (g_ps5_pool_used + kPs5PtChunk > kPs5PtPool) {
        struct ps5_shm next{};
        if (!ps5_zero_object(&next, kPs5PtPool)) return false;
        g_ps5_pool = next;
        g_ps5_pool_used = 0;
    }
    void* address = reinterpret_cast<void*>(t.base + chunk * kPs5PtChunk);
    if (current == kPs5PtZero) {
        state.store(kPs5PtMoving, std::memory_order_release);
        ps5_shm_unmap(address, kPs5PtChunk, PS5_SHM_KEEP_RESERVED);
    } else {
        state.store(kPs5PtMoving, std::memory_order_release);
    }
    void* view = nullptr;
    int rc = ps5_shm_map(&g_ps5_pool, g_ps5_pool_used, kPs5PtChunk, address, PS5_SHM_READ | PS5_SHM_WRITE,
                         PS5_SHM_FIXED | PS5_SHM_KEEP_RESERVED, &view);
    if (rc != 0) {
        std::fprintf(stderr, "dynarmic-ps5: page table chunk map failed rc=%#x\n", rc);
        state.store(kPs5PtEmpty, std::memory_order_release);
        return false;
    }
    g_ps5_pool_used += kPs5PtChunk;
    g_ps5_pt_backed.fetch_add(1, std::memory_order_relaxed);
    state.store(kPs5PtBacked, std::memory_order_release);
    return true;
}

void ps5_pt_sigaction(int sig, siginfo_t* info, void* context);

struct Ps5VirtualQueryInfo {
    uintptr_t start;
    uintptr_t end;
    int64_t offset;
    int protection;
    int memory_type;
    uint8_t bits;
    char name[32];
};

extern "C" int sceKernelVirtualQuery(const void* addr, int flags, Ps5VirtualQueryInfo* info, size_t size);

bool ps5_pt_usable(uintptr_t at) {
    Ps5VirtualQueryInfo info{};
    return sceKernelVirtualQuery(reinterpret_cast<const void*>(at), 0, &info, sizeof(info)) == 0 && info.start <= at &&
           at < info.end;
}

bool ps5_pt_reserve(size_t bytes, void** out) {
    void* rejected[64];
    int count = 0;
    bool found = false;
    for (int k = -1; k < 60 && !found; k++) {
        uintptr_t hint = k < 0 ? 0 : 0x1000000000ull + static_cast<uintptr_t>(k) * 0x400000000ull;
        void* base = nullptr;
        if (ps5_vrange_reserve(bytes, reinterpret_cast<void*>(hint), 1 << 21, &base) != 0) continue;
        uintptr_t at = reinterpret_cast<uintptr_t>(base);
        if (ps5_pt_usable(at) && ps5_pt_usable(at + bytes / 2) && ps5_pt_usable(at + bytes - kPs5PtChunk)) {
            *out = base;
            found = true;
        } else if (count < 64) {
            rejected[count++] = base;
        } else {
            ps5_vrange_release(base, bytes);
        }
    }
    if (count > 0) {
        std::fprintf(stderr, "dynarmic-ps5: skipped %d unusable page table reservations (first %p)\n", count, rejected[0]);
    }
    for (int i = 0; i < count; i++) ps5_vrange_release(rejected[i], bytes);
    return found;
}

bool ps5_pt_install_handler() {
    struct sigaction sa{};
    sa.sa_sigaction = &ps5_pt_sigaction;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    return sigaction(SIGSEGV, &sa, &g_ps5_old_segv) == 0 && sigaction(SIGBUS, &sa, &g_ps5_old_bus) == 0;
}

}

extern "C" bool dynarmic_ps5_page_table_fault(uintptr_t addr) {
    Ps5PageTable* t = ps5_pt_find(addr);
    if (!t) return false;
    size_t chunk = (addr - t->base) / kPs5PtChunk;
    std::atomic<uint8_t>& state = t->state[chunk];
    for (;;) {
        uint8_t current = state.load(std::memory_order_acquire);
        if (current == kPs5PtBacked || current == kPs5PtZero) return true;
        if (current == kPs5PtMoving) {
            sched_yield();
            continue;
        }
        uint8_t expected = kPs5PtEmpty;
        if (!state.compare_exchange_strong(expected, kPs5PtMoving, std::memory_order_acq_rel)) continue;
        void* view = nullptr;
        int rc = ps5_shm_map(&g_ps5_zero, 0, kPs5PtChunk, reinterpret_cast<void*>(t->base + chunk * kPs5PtChunk),
                             PS5_SHM_READ, PS5_SHM_FIXED | PS5_SHM_KEEP_RESERVED, &view);
        if (rc != 0) {
            state.store(kPs5PtEmpty, std::memory_order_release);
            return false;
        }
        g_ps5_pt_zero_faults.fetch_add(1, std::memory_order_relaxed);
        state.store(kPs5PtZero, std::memory_order_release);
        return true;
    }
}

namespace {

void ps5_pt_sigaction(int sig, siginfo_t* info, void* context) {
    if (dynarmic_ps5_page_table_fault(reinterpret_cast<uintptr_t>(info->si_addr))) return;
    struct sigaction* old = sig == SIGBUS ? &g_ps5_old_bus : &g_ps5_old_segv;
    if ((old->sa_flags & SA_SIGINFO) && old->sa_sigaction) {
        old->sa_sigaction(sig, info, context);
        return;
    }
    if (old->sa_handler != SIG_DFL && old->sa_handler != SIG_IGN && old->sa_handler) {
        old->sa_handler(sig);
        return;
    }
    sigaction(sig, old, nullptr);
}

}

static void** ps5_page_table_create(size_t bytes) {
    std::lock_guard<std::mutex> guard(g_ps5_pt_mutex);
    int count = g_ps5_pt_count.load(std::memory_order_acquire);
    if (count >= 8) return nullptr;
    if (count == 0) {
        if (!ps5_zero_object(&g_ps5_zero, kPs5PtChunk)) return nullptr;
        if (!ps5_pt_install_handler()) return nullptr;
    }
    void* base = nullptr;
    if (!ps5_pt_reserve(bytes, &base)) return nullptr;
    Ps5PageTable& t = g_ps5_pt[count];
    t.base = reinterpret_cast<uintptr_t>(base);
    t.bytes = bytes;
    t.state = new std::atomic<uint8_t>[bytes / kPs5PtChunk]();
    g_ps5_pt_count.store(count + 1, std::memory_order_release);
    std::fprintf(stderr, "dynarmic-ps5: page table %#zx bytes reserved at %p (lazy %zu KiB chunks)\n", bytes, base,
                 kPs5PtChunk / 1024);
    return static_cast<void**>(base);
}

static inline bool page_table_prepare(void** table, u64 idx, bool value) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(&table[idx]);
    Ps5PageTable* t = ps5_pt_find(addr);
    if (!t) return true;
    size_t chunk = (addr - t->base) / kPs5PtChunk;
    uint8_t state = t->state[chunk].load(std::memory_order_acquire);
    if (state == kPs5PtBacked) return true;
    if (!value) return false;
    std::lock_guard<std::mutex> guard(g_ps5_pt_mutex);
    return ps5_pt_back_locked(*t, chunk);
}

extern "C" void dynarmic_ps5_page_table_stats(uint64_t* backed_chunks, uint64_t* zero_faults) {
    if (backed_chunks) *backed_chunks = g_ps5_pt_backed.load(std::memory_order_relaxed);
    if (zero_faults) *zero_faults = g_ps5_pt_zero_faults.load(std::memory_order_relaxed);
}
#else
static inline bool page_table_prepare(void**, u64, bool) { return true; }
#endif

static inline void page_table_store(void** table, u64 idx, void* value) {
    if (page_table_prepare(table, idx, value != nullptr)) table[idx] = value;
}
