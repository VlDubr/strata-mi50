#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/progress.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <vector>

namespace strata::core::conversation_detail {

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
///
/// The copies run as shader blits, not SDMA (HSA_ENABLE_SDMA=0), so the blit touches a plain host endpoint
/// through a device mapping that dies with the pages.  A host endpoint freed before its blit runs - a free on
/// another thread, or a torn-down turn whose buffer the sticky fault outlives - faults the GPU, and the sticky
/// error surfaces many calls later, naming whichever copy polled next (2026-10-08/10-11 T5810: WRITE page
/// faults inside one 2 MiB span of heap, surfacing at a checkpoint save and at KV streaming).  Plain host
/// endpoints therefore stage through pinned bounce scratch the driver maps for the process lifetime; no blit
/// ever touches pages a free could take away.  Pinned or registered host buffers (the pool aliases, the pack
/// arena) and device allocations keep the direct copy.  An endpoint the runtime cannot classify keeps the old
/// cudaMemcpyDefault behaviour.

inline constexpr size_t kCopyBounceBytes = 4 << 20;   // pinned staging per thread, sized for one segment visit

/// A device allocation is `device`; `accessible` also covers every host buffer the device can address for the
/// allocation's lifetime (pinned staging, the registered pack arena, the mapped pool aliases) - a blit may
/// touch those directly because their mapping cannot be taken away under it.
inline cudaError_t endpoint_kind(const void* p, bool& device, bool& accessible) {
    cudaPointerAttributes a{};
    const cudaError_t e = cudaPointerGetAttributes(&a, const_cast<void*>(p));
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
        return e;
    }
    device = a.type == cudaMemoryTypeDevice;
    accessible = device || a.devicePointer != nullptr;
    return cudaSuccess;
}

/// private_stream: this thread's stream on the current device, created on first use.
inline cudaError_t private_stream(cudaStream_t& s) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    s = streams[(size_t) dev];
    return cudaSuccess;
}

/// `queued`: the copy is only queued on the private stream, and the caller synchronizes the device (or orders a
/// later launch after it with order_after_queued) before anything reads the target or writes the source.
/// A leg that stages through bounce scratch ignores `queued` and waits for the slice it queued: the scratch is
/// this thread's, and returning early would let the next copy overwrite bytes the queue has not read.
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes, bool queued = false) {
    if (!bytes) return cudaSuccess;
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;

    bool src_device = false, src_accessible = false, dst_device = false, dst_accessible = false;
    if (endpoint_kind(src, src_device, src_accessible) != cudaSuccess ||
        endpoint_kind(dst, dst_device, dst_accessible) != cudaSuccess) {
        // An endpoint the runtime refuses to classify: the behaviour before the bounce existed.
        e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
        if (e != cudaSuccess || queued) return e;
        e = cudaStreamSynchronize(s);
        if (e == cudaSuccess) core::progress_beat();   // #2049: a parked conversation is many of these
        return e;
    }
    if (!src_device && !dst_device) {
        // Host to host, through an alias or not: the CPU moves the bytes, the GPU nothing.
        std::memmove(dst, src, bytes);
        core::progress_beat();   // #2049: pageable parked state copies at page-in speed and stays silent
        return cudaSuccess;
    }
    if (src_device && dst_device) {
        e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
        if (e != cudaSuccess || queued) return e;
        e = cudaStreamSynchronize(s);
        if (e == cudaSuccess) core::progress_beat();
        return e;
    }
    if (src_device ? dst_accessible : src_accessible) {
        // The device blit only touches the endpoint whose mapping is stable for the allocation's lifetime.
        e = cudaMemcpyAsync(dst, src, bytes, src_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess || queued) return e;
        e = cudaStreamSynchronize(s);
        if (e == cudaSuccess) core::progress_beat();
        return e;
    }
    // One endpoint is plain (not device-accessible) host memory: stage the plain leg through pinned bounce
    // scratch so no blit ever writes or reads pages a free could take away while it runs.
    struct Bounce { uint8_t* p; };
    thread_local Bounce* bounce = nullptr;   // never freed: process lifetime, like the streams
    if (bounce == nullptr) {
        void* p = nullptr;
        const cudaError_t e0 = cudaHostAlloc(&p, kCopyBounceBytes, cudaHostAllocDefault);
        if (e0 != cudaSuccess) return e0;
        bounce = new Bounce{static_cast<uint8_t*>(p)};
    }
    const bool to_host = src_device;   // device to plain host; the other plain case is plain host to device
    for (size_t off = 0; off < bytes; off += kCopyBounceBytes) {
        const size_t n = std::min(kCopyBounceBytes, bytes - off);
        if (to_host) {
            e = cudaMemcpyAsync(bounce->p, static_cast<const uint8_t*>(src) + off, n, cudaMemcpyDeviceToHost, s);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s);
            if (e == cudaSuccess) std::memcpy(static_cast<uint8_t*>(dst) + off, bounce->p, n);
        } else {
            std::memcpy(bounce->p, static_cast<const uint8_t*>(src) + off, n);
            e = cudaMemcpyAsync(static_cast<uint8_t*>(dst) + off, bounce->p, n, cudaMemcpyHostToDevice, s);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        }
        if (e != cudaSuccess) return e;
        core::progress_beat();   // #2049: each slice that lands is progress, as upstream's plain copy reports
    }
    return cudaSuccess;
}

/// #2049 #2065 #1971: a host-to-host copy that tells the watchdog it is moving.  Parked state (the running
/// checkpoints, the K/V segments) sits in ordinary pageable memory for the whole session; when the OS has written
/// it to the pagefile, copying it back runs at the drive's page-in speed (measured in the reports: ~190 hard faults
/// a second, under 1 MB/s) and one memcpy of it stays silent past the 60 s limit, so the watchdog ended an engine
/// that was only slow.  One beat per 1 MiB slice keeps a copy that moves alive and still lets a real stall show.
/// STRATA_TEST_SLOW_COPY_US=N (tests only): every 1 MiB slice of a checkpoint copy takes N microseconds more, as when
/// the OS has paged the checkpoints out and each slice is read back from the drive.  Unset: no delay.
inline long test_slow_copy_us() {
    static const long v = [] { const char* e = std::getenv("STRATA_TEST_SLOW_COPY_US"); return e ? std::atol(e) : 0L; }();
    return v;
}

template<class T>
inline std::vector<T> copy_beating(const std::vector<T>& src) {
    std::vector<T> out;
    out.reserve(src.size());
    const size_t step = std::max<size_t>(1, ((size_t) 1 << 20) / sizeof(T));
    for (size_t at = 0; at < src.size(); at += step) {
        out.insert(out.end(), src.begin() + (std::ptrdiff_t) at,
                   src.begin() + (std::ptrdiff_t) std::min(src.size(), at + step));
        if (const long us = test_slow_copy_us(); us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
        core::progress_beat();
    }
    return out;
}

inline ConversationCheckpoint copy_checkpoint_beating(const ConversationCheckpoint& c) {
    ConversationCheckpoint o;
    o.ids = copy_beating(c.ids);
    o.imgs = copy_beating(c.imgs);
    o.gdn = copy_beating(c.gdn);
    o.ple = copy_beating(c.ple);
    o.tails = copy_beating(c.tails);
    o.dead = copy_beating(c.dead);
    o.block_pos = copy_beating(c.block_pos);
    o.used = c.used;
    o.pinned = c.pinned;
    o.stage_parts.reserve(c.stage_parts.size());
    for (const auto& part : c.stage_parts) o.stage_parts.push_back(copy_checkpoint_beating(part));
    return o;
}

inline std::vector<ConversationCheckpoint> copy_checkpoints_beating(const std::vector<ConversationCheckpoint>& cs) {
    std::vector<ConversationCheckpoint> out;
    out.reserve(cs.size());
    for (const auto& c : cs) out.push_back(copy_checkpoint_beating(c));
    return out;
}

/// Work queued on the legacy stream after this call starts after every copy this thread queued on the current
/// device's private stream (an event the legacy stream waits for; no host wait).
inline cudaError_t order_after_queued() {
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    int dev = 0;
    e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaEvent_t> events;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= events.size()) events.resize((size_t) dev + 1, nullptr);    // never destroyed: process lifetime
    if (!events[(size_t) dev]) {
        e = cudaEventCreateWithFlags(&events[(size_t) dev], cudaEventDisableTiming);
        if (e != cudaSuccess) { events[(size_t) dev] = nullptr; return e; }
    }
    e = cudaEventRecord(events[(size_t) dev], s);
    if (e != cudaSuccess) return e;
    return cudaStreamWaitEvent(nullptr, events[(size_t) dev], 0);
}

} // namespace strata::core::conversation_detail
