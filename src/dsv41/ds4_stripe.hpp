// src/dsv41/ds4_stripe.hpp - RAM striped over a machine's memory regions.
// An EPYC 7002 with six DIMMs has no six-way interleave: its physical memory is three regions of two channels each, about 40 GB/s
// a region whatever the number of threads, and a stream that stays in a region gets no more.  Memory taken in turn from every
// region reads at the sum (measured on 3 x 32 GiB: 41 -> 95 GB/s).  The kernel gives no way to ask for a region, so a pool of huge
// pages is taken, sorted by their frames (/proc/self/pagemap: CAP_SYS_ADMIN) and moved, a region at a time, to consecutive addresses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#endif

namespace strata::ds4::stripe {

#if defined(__linux__)
inline uint64_t mem_room() {   // what can be taken: the RAM available and, in a container, what its limit leaves
    uint64_t av = 0;
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[128];
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemAvailable: %llu kB", (unsigned long long*) &av) == 1) {
                av <<= 10;
                break;
            }
        std::fclose(f);
    }
    unsigned long long lim = 0, cur = 0;
    if (std::FILE* f = std::fopen("/sys/fs/cgroup/memory.max", "r")) {
        const bool got = std::fscanf(f, "%llu", &lim) == 1;
        std::fclose(f);
        if (got) {
            if (std::FILE* c = std::fopen("/sys/fs/cgroup/memory.current", "r")) {
                if (std::fscanf(c, "%llu", &cur) != 1) cur = 0;
                std::fclose(c);
            }
            if (lim > cur) av = std::min<uint64_t>(av, lim - cur);
        }
    }
    return av;
}

/// `bytes` of memory, 2 MiB at a time from each region in turn.  bounds_gib: where the regions end, but the last.  Locked, so nothing
/// swaps it.  Null when the RAM is not there (`report` says why); without the frames (no CAP_SYS_ADMIN) the memory comes in the
/// kernel's order and `report` says that too.
inline uint8_t* alloc(size_t bytes, const std::vector<double>& bounds_gib, std::string& report) {
    constexpr size_t H = (size_t) 2 << 20;
    const size_t n = (bytes + H - 1) / H;
    const uint64_t keep = (uint64_t) 2 << 30, room = mem_room();
    if (room < n * H + keep) {
        report = "only " + std::to_string(room >> 30) + " GiB of RAM to take from";
        return nullptr;
    }
    // (some spare, so that the regions can come out even)
    const size_t pool = std::min<size_t>(n + n / 5, (size_t) ((room - keep) / H));
    char* raw = (char*) mmap(nullptr, (pool + 1) * H, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == (char*) MAP_FAILED) {
        report = "mmap of the pool failed";
        return nullptr;
    }
    char* m = (char*) (((uintptr_t) raw + H - 1) & ~(uintptr_t) (H - 1));
    madvise(m, pool * H, MADV_HUGEPAGE);
    {
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 16; ++t)
            th.emplace_back([&] {
                for (size_t i; (i = next.fetch_add(1)) < pool;)
                    for (size_t o = 0; o < H; o += 4096) *(volatile char*) (m + i * H + o) = 0;
            });
        for (auto& t : th) t.join();
    }
    const int nr = (int) bounds_gib.size() + 1;
    std::vector<std::vector<uint32_t>> reg((size_t) nr);
    std::vector<uint32_t> mixed;
    const int fd = open("/proc/self/pagemap", O_RDONLY);
    bool frames = false;
    for (size_t i = 0; i < pool; ++i) {
        uint64_t e[2] = {0, 0};
        if (fd >= 0) {
            const off_t at = (off_t) ((uintptr_t) (m + i * H) / 4096) * 8;
            if (pread(fd, e, 16, at) != 16) e[0] = e[1] = 0;
        }
        const uint64_t f0 = e[0] & (((uint64_t) 1 << 55) - 1), f1 = e[1] & (((uint64_t) 1 << 55) - 1);
        frames = frames || f0 != 0;
        if (f0 == 0 || f1 != f0 + 1) {   // (no frame, or small pages from all over)
            mixed.push_back((uint32_t) i);
            continue;
        }
        const double gib = (double) f0 * 4096.0 / (double) ((uint64_t) 1 << 30);
        int r = 0;
        while (r < nr - 1 && gib >= bounds_gib[(size_t) r]) ++r;
        reg[(size_t) r].push_back((uint32_t) i);
    }
    if (fd >= 0) close(fd);
    char* dst = (char*) mmap(nullptr, (n + 1) * H, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (dst == (char*) MAP_FAILED) {
        munmap(raw, (pool + 1) * H);
        report = "mmap of the stripes failed";
        return nullptr;
    }
    char* v = (char*) (((uintptr_t) dst + H - 1) & ~(uintptr_t) (H - 1));
    std::vector<size_t> at((size_t) nr, 0), used((size_t) nr, 0);
    size_t mx = 0, nm = 0, done = 0;
    bool ok = true;
    for (int r = 0; done < n && ok;) {   // a page of each region in turn; a region run out: small pages, then the others
        size_t src = (size_t) -1;
        if (at[(size_t) r] < reg[(size_t) r].size()) {
            src = reg[(size_t) r][at[(size_t) r]++];
            ++used[(size_t) r];
        } else if (mx < mixed.size()) {
            src = mixed[mx++];
            ++nm;
        }
        r = (r + 1) % nr;
        if (src == (size_t) -1) {
            bool any = false;
            for (int q = 0; q < nr; ++q) any = any || at[(size_t) q] < reg[(size_t) q].size();
            if (!any) ok = false;
            continue;
        }
        ok = mremap(m + src * H, H, H, MREMAP_MAYMOVE | MREMAP_FIXED, v + done * H) != MAP_FAILED;
        ++done;
    }
    if (!ok) {
        report = "moving the pages failed";
        return nullptr;
    }
    for (int r = 0; r < nr; ++r)
        for (size_t i = at[(size_t) r]; i < reg[(size_t) r].size(); ++i) munmap(m + (size_t) reg[(size_t) r][i] * H, H);
    for (size_t i = mx; i < mixed.size(); ++i) munmap(m + (size_t) mixed[i] * H, H);
    const bool locked = mlock(v, n * H) == 0;
    char b[192];
    std::string parts;
    for (int r = 0; r < nr; ++r) parts += (r ? " + " : "") + std::to_string(used[(size_t) r] * 2 >> 10);
    std::snprintf(b, sizeof b, "%s GiB in huge pages by region, %zu GiB in small pages%s%s", parts.c_str(), nm * 2 >> 10,
                  frames ? "" : " (no frames: CAP_SYS_ADMIN is missing, not striped)", locked ? "" : " (not locked)");
    report = b;
    return (uint8_t*) v;
}
#else
inline uint8_t* alloc(size_t, const std::vector<double>&, std::string& report) {
    report = "Linux only";
    return nullptr;
}
#endif

}  // namespace strata::ds4::stripe
