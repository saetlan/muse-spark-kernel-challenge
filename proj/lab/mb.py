"""CUDA microbenchmarks (from PE3; its log was lost to FlashInfer's autotuner output): one-shot /
looped 16-B nc loads, bulk L2 prefetch, cp.async.bulk rings, empty-kernel cost, fixed-order DSMEM
reduction across 2/4/8-CTA clusters, PRMT/LOP3 issue rates. Sections: sec_mb_build, sec_mb_basics,
sec_mb_stream, sec_mb_cluster (see lab/e3.py for the CUDA source notes)."""

import time

MB_SIZES = (("8.4MB", 1024 * 4096 * 2), ("16.8MB", 8192 * 1024 * 2), ("37.7MB", 2304 * 8192 * 2),
            ("67.1MB", 4096 * 8192 * 2))

MB_SRC = {
    "core": r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

__device__ __forceinline__ void ld_nc(const uint4* p, unsigned (&r)[4]) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}

// Each CTA streams its contiguous slice; D 16-B loads in flight per thread per step.
template <int D>
__global__ void __launch_bounds__(512) mb_ldg(const uint4* __restrict__ w, long long per_cta, unsigned* sink, int prefetch) {
    const uint4* base = w + (long long)blockIdx.x * per_cta;
    if (prefetch && threadIdx.x == 0) {
        long long bytes = per_cta * 16;
        for (long long off = 0; off < bytes; off += 32768) {
            long long sz = bytes - off < 32768 ? bytes - off : 32768;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"((const char*)base + off), "r"((unsigned)sz) : "memory");
        }
    }
    unsigned acc = 0;
    const long long step = (long long)blockDim.x * D;
    for (long long i = threadIdx.x; i < per_cta; i += step) {
        unsigned r[D][4];
#pragma unroll
        for (int d = 0; d < D; ++d) {
            long long j = i + (long long)d * blockDim.x;
            if (j < per_cta) {
                ld_nc(base + j, r[d]);
            } else {
                r[d][0] = r[d][1] = r[d][2] = r[d][3] = 0u;
            }
        }
#pragma unroll
        for (int d = 0; d < D; ++d) acc ^= r[d][0] ^ r[d][1] ^ r[d][2] ^ r[d][3];
    }
    if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;
}

__global__ void mb_empty(unsigned* sink) {
    if (threadIdx.x == 100000u) sink[0] = 1u;
}

// 8 independent chains per thread of PRMT (op 0) or LOP3 (op 1).
__global__ void __launch_bounds__(256) mb_int(unsigned* sink, int iters, int op) {
    unsigned a0 = threadIdx.x, a1 = a0 * 3u + 1u, a2 = a0 ^ 0x5a5a5a5au, a3 = a0 + 7u;
    unsigned a4 = a0 * 5u, a5 = ~a0, a6 = a0 << 3, a7 = a0 >> 1;
    const unsigned k = 0x76543210u ^ (unsigned)iters;
    const unsigned k2 = 0x80808080u ^ (unsigned)(iters >> 20);
    if (op == 0) {
        for (int i = 0; i < iters; ++i) {
            asm("prmt.b32 %0, %0, %8, 0x5140;\nprmt.b32 %1, %1, %8, 0x7362;\nprmt.b32 %2, %2, %8, 0x5140;\nprmt.b32 %3, %3, %8, 0x7362;\n"
                "prmt.b32 %4, %4, %8, 0x5140;\nprmt.b32 %5, %5, %8, 0x7362;\nprmt.b32 %6, %6, %8, 0x5140;\nprmt.b32 %7, %7, %8, 0x7362;"
                : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5), "+r"(a6), "+r"(a7) : "r"(k), "r"(k2));
        }
    } else {
        for (int i = 0; i < iters; ++i) {
            asm("lop3.b32 %0, %0, %8, %9, 0xEA;\nlop3.b32 %1, %1, %8, %9, 0xEA;\nlop3.b32 %2, %2, %8, %9, 0xEA;\n"
                "lop3.b32 %3, %3, %8, %9, 0xEA;\nlop3.b32 %4, %4, %8, %9, 0xEA;\nlop3.b32 %5, %5, %8, %9, 0xEA;\n"
                "lop3.b32 %6, %6, %8, %9, 0xEA;\nlop3.b32 %7, %7, %8, %9, 0xEA;"
                : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5), "+r"(a6), "+r"(a7) : "r"(k), "r"(k2));
        }
    }
    unsigned acc = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
    if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;
}

void mb_run_ldg(torch::Tensor w, int64_t per_cta, int64_t ctas, int64_t threads, int64_t d, int64_t prefetch, torch::Tensor sink) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint4* p = reinterpret_cast<const uint4*>(w.data_ptr());
    unsigned* s = reinterpret_cast<unsigned*>(sink.data_ptr());
    const int pf = (int)prefetch;
    switch (d) {
        case 2: mb_ldg<2><<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(p, per_cta, s, pf); break;
        case 4: mb_ldg<4><<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(p, per_cta, s, pf); break;
        case 8: mb_ldg<8><<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(p, per_cta, s, pf); break;
        case 16: mb_ldg<16><<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(p, per_cta, s, pf); break;
        default: TORCH_CHECK(false, "unsupported D");
    }
}

void mb_run_empty(int64_t ctas, int64_t threads, torch::Tensor sink) {
    auto stream = at::cuda::getCurrentCUDAStream();
    mb_empty<<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(reinterpret_cast<unsigned*>(sink.data_ptr()));
}

void mb_run_int(int64_t ctas, int64_t iters, int64_t op, torch::Tensor sink) {
    auto stream = at::cuda::getCurrentCUDAStream();
    mb_int<<<(unsigned)ctas, 256, 0, stream>>>(reinterpret_cast<unsigned*>(sink.data_ptr()), (int)iters, (int)op);
}
""",
    "bulk": r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

__device__ __forceinline__ void ld_nc(const uint4* p, unsigned (&r)[4]) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}

__device__ __forceinline__ void bulk_issue(unsigned bar, unsigned dst, const char* src, unsigned bytes) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n"
                 :: "r"(bar), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// cp.async.bulk ring: S stages of CHUNK bytes; one thread issues, all threads touch each stage.
template <int S>
__global__ void __launch_bounds__(256) mb_bulk(const uint4* __restrict__ w, long long per_cta, int chunk, unsigned* sink) {
    extern __shared__ __align__(128) unsigned char ring[];
    __shared__ __align__(8) unsigned long long bars[S];
    const char* base = (const char*)(w + (long long)blockIdx.x * per_cta);
    const int nchunks = (int)(per_cta * 16 / chunk);
    const unsigned bar0 = smem_addr(bars);
    const unsigned ring0 = smem_addr(ring);
    if (threadIdx.x == 0) {
        for (int s = 0; s < S; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar0 + 8u * s) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int c = 0; c < S && c < nchunks; ++c)
            bulk_issue(bar0 + 8u * c, ring0 + (unsigned)(c * chunk), base + (long long)c * chunk, (unsigned)chunk);
    }
    unsigned acc = 0;
    for (int c = 0; c < nchunks; ++c) {
        const int s = c % S;
        mbar_wait(bar0 + 8u * s, (unsigned)((c / S) & 1));
        const uint4* st = (const uint4*)(ring + (size_t)s * chunk);
        for (int t = threadIdx.x * 8; t < chunk / 16; t += blockDim.x * 8) acc ^= st[t].x;
        __syncthreads();
        if (threadIdx.x == 0 && c + S < nchunks)
            bulk_issue(bar0 + 8u * s, ring0 + (unsigned)(s * chunk), base + (long long)(c + S) * chunk, (unsigned)chunk);
    }
    if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;
}

void mb_run_bulk(torch::Tensor w, int64_t per_cta, int64_t ctas, int64_t stages, int64_t chunk, torch::Tensor sink) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint4* p = reinterpret_cast<const uint4*>(w.data_ptr());
    unsigned* s = reinterpret_cast<unsigned*>(sink.data_ptr());
    const size_t smem = (size_t)stages * (size_t)chunk;
    TORCH_CHECK(per_cta * 16 % chunk == 0 && chunk % 16 == 0 && smem <= 200 * 1024, "bad bulk config");
    if (stages == 4) {
        mb_bulk<4><<<(unsigned)ctas, 256, smem, stream>>>(p, per_cta, (int)chunk, s);
    } else if (stages == 6) {
        mb_bulk<6><<<(unsigned)ctas, 256, smem, stream>>>(p, per_cta, (int)chunk, s);
    } else {
        TORCH_CHECK(stages == 8, "unsupported stages");
        mb_bulk<8><<<(unsigned)ctas, 256, smem, stream>>>(p, per_cta, (int)chunk, s);
    }
}

void mb_bulk_attrs() {
    const int big = 200 * 1024;
    cudaFuncSetAttribute(mb_bulk<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, big);
    cudaFuncSetAttribute(mb_bulk<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, big);
    cudaFuncSetAttribute(mb_bulk<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, big);
}
""",
    "cluster": r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

__device__ __forceinline__ void ld_nc(const uint4* p, unsigned (&r)[4]) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}

__device__ __forceinline__ void bulk_issue(unsigned bar, unsigned dst, const char* src, unsigned bytes) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n"
                 :: "r"(bar), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// Each CTA holds an n_per-float partial; CTA r of a C-CTA cluster sums rows [r*n/C, (r+1)*n/C)
// over all peers in rank order (deterministic) and writes them out.
template <int C>
__global__ void __cluster_dims__(C, 1, 1) __launch_bounds__(256) mb_cluster(const float* __restrict__ in, float* __restrict__ out, int n_per) {
    extern __shared__ __align__(16) float part[];
    for (int i = threadIdx.x; i < n_per; i += blockDim.x) part[i] = in[(long long)blockIdx.x * n_per + i];
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int share = n_per / C;
    const unsigned local = smem_addr(part);
    for (int i = (int)rank * share + threadIdx.x; i < ((int)rank + 1) * share; i += blockDim.x) {
        float s = 0.f;
#pragma unroll
        for (int r = 0; r < C; ++r) {
            unsigned remote;
            float v;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local + 4u * (unsigned)i), "r"(r));
            asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
            s += v;
        }
        out[(long long)(blockIdx.x / C) * n_per + i] = s;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

void mb_run_cluster(torch::Tensor in, torch::Tensor out, int64_t ctas, int64_t c, int64_t n_per) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* i = reinterpret_cast<const float*>(in.data_ptr());
    float* o = reinterpret_cast<float*>(out.data_ptr());
    const size_t smem = (size_t)n_per * 4;
    TORCH_CHECK(ctas % c == 0 && n_per % c == 0 && smem <= 48 * 1024, "bad cluster config");
    if (c == 2) mb_cluster<2><<<(unsigned)ctas, 256, smem, stream>>>(i, o, (int)n_per);
    else if (c == 4) mb_cluster<4><<<(unsigned)ctas, 256, smem, stream>>>(i, o, (int)n_per);
    else { TORCH_CHECK(c == 8, "unsupported cluster"); mb_cluster<8><<<(unsigned)ctas, 256, smem, stream>>>(i, o, (int)n_per); }
}

""",
}

MB_CPP = {
    "core": ['void mb_run_ldg(torch::Tensor w, int64_t per_cta, int64_t ctas, int64_t threads, int64_t d, int64_t prefetch, torch::Tensor sink);', 'void mb_run_empty(int64_t ctas, int64_t threads, torch::Tensor sink);', 'void mb_run_int(int64_t ctas, int64_t iters, int64_t op, torch::Tensor sink);'],
    "bulk": ['void mb_run_bulk(torch::Tensor w, int64_t per_cta, int64_t ctas, int64_t stages, int64_t chunk, torch::Tensor sink);', 'void mb_bulk_attrs();'],
    "cluster": ['void mb_run_cluster(torch::Tensor in, torch::Tensor out, int64_t ctas, int64_t c, int64_t n_per);'],
}


def mb_build(part):
    import tempfile
    from torch.utils.cpp_extension import load_inline
    decls = MB_CPP[part]
    names = [d.split("(")[0].split()[-1] for d in decls]
    return load_inline(
        name=f"okc_proj_mb_{part}", cpp_sources="\n".join(decls), cuda_sources=MB_SRC[part], functions=names,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix=f"okc_proj_mb_{part}_"), verbose=False)


def _unused_env(lab):
    import importlib.metadata as md
    import importlib.util as iu
    parts = [f"torch {torch.__version__}", f"triton {triton.__version__}"]
    for dist in ("flashinfer-python", "flashinfer", "deep_gemm", "deep-gemm", "nvidia-cutlass-dsl", "tilelang",
                 "cuda-python", "nvidia-cudnn-frontend", "hpc-ops"):
        try:
            parts.append(f"{dist} {md.version(dist)}")
        except Exception:
            pass
    for mod in ("hpc", "deep_gemm", "flashinfer", "cudnn", "triton.experimental.gluon",
                "triton.experimental.gluon.language.nvidia.hopper", "cutlass", "quack"):
        try:
            parts.append(f"{mod}:{'y' if iu.find_spec(mod) else 'n'}")
        except Exception as exc:
            parts.append(f"{mod}:{type(exc).__name__}")
    props = torch.cuda.get_device_properties(0)
    parts.append(f"L2 {getattr(props, 'L2_cache_size', -1) / 2**20:.0f} MiB, sms {props.multi_processor_count}")
    log("env: " + ", ".join(parts))
    try:
        import deep_gemm
        log(f"deep_gemm: bf16_gemm_nt {hasattr(deep_gemm, 'bf16_gemm_nt')}, "
            f"deterministic {deep_gemm.get_deterministic_algorithms()}")
    except Exception as exc:
        log(f"deep_gemm: {type(exc).__name__}: {_error_text(exc, 200)}")


class _MB:
    def __init__(self):
        self.mods = {}
        self.empty_us = None


def sec_mb_build(lab, mb, part):
    t = time.monotonic()
    mod = mb_build(part)
    if part == "bulk":
        mod.mb_bulk_attrs()
    mb.mods[part] = mod
    log(f"mb: nvcc build {part} {time.monotonic() - t:.1f}s")


def sec_mb_basics(lab, mb):
    m = mb.mods["core"]
    sink = torch.zeros(4096, dtype=torch.int32, device="cuda")
    e1 = lab.time_graph(lambda: m.mb_run_empty(1, 32, sink))
    e132 = lab.time_graph(lambda: m.mb_run_empty(132, 512, sink))
    e132w = lab.time_graph(lambda: m.mb_run_empty(132, 512, sink), cold=False)
    two = lab.time_graph(lambda: (m.mb_run_empty(132, 512, sink), m.mb_run_empty(132, 512, sink)))
    mb.empty_us = e132
    log(f"mb empty kernel (graph replay, raw): 1 CTA {e1:.2f} us, 132 CTAs {e132:.2f} cold / {e132w:.2f} warm, "
        f"2 kernels {two:.2f} us")
    for op, name in ((0, "prmt"), (1, "lop3")):
        iters = 4096
        us = lab.time_graph(lambda: m.mb_run_int(132 * 8, iters, op, sink), cold=False)
        ops = 132 * 8 * 256 * iters * 8
        per_clk_sm = ops / (max(us - e132w, 1e-3) * 1e-6) / 132 / 1.98e9
        log(f"mb {name}: {per_clk_sm:.1f} thread-ops per clock per SM ({us:.1f} us)")


def sec_mb_stream(lab, mb):
    m = mb.mods["core"]
    mbulk = mb.mods.get("bulk")
    sink = torch.zeros(4096, dtype=torch.int32, device="cuda")
    buf = torch.randint(-2**31, 2**31 - 1, (MB_SIZES[-1][1] // 4,), dtype=torch.int32, device="cuda")
    base = mb.empty_us or 0.0
    for name, nbytes in MB_SIZES:
        rows = []
        best = (1e9, "")
        for ctas in (132, 264, 528):
            per = nbytes // 16 // ctas
            for threads in (256, 512):
                for d in (2, 4, 8, 16):
                    if threads * d > 4 * per + threads:  # mostly idle threads
                        continue
                    us = lab.time_graph(lambda: m.mb_run_ldg(buf, per, ctas, threads, d, 0, sink))
                    rows.append(f"c{ctas}t{threads}d{d}:{us:.2f}")
                    if us < best[0]:
                        best = (us, f"ldg c{ctas} t{threads} d{d}")
        for ctas in (132, 264):
            per = nbytes // 16 // ctas
            for threads, d in ((512, 4), (512, 8)):
                us = lab.time_graph(lambda: m.mb_run_ldg(buf, per, ctas, threads, d, 1, sink))
                rows.append(f"pf c{ctas}t{threads}d{d}:{us:.2f}")
                if us < best[0]:
                    best = (us, f"pf+ldg c{ctas} t{threads} d{d}")
        for ctas in ((132, 264) if mbulk is not None else ()):
            for stages, chunk in ((4, 16384), (8, 16384), (4, 32768), (6, 32768)):
                per = (nbytes // ctas // chunk) * chunk // 16
                if per == 0:
                    continue
                us = lab.time_graph(lambda: mbulk.mb_run_bulk(buf, per, ctas, stages, chunk, sink))
                eff_bytes = per * 16 * ctas
                rows.append(f"bulk c{ctas}s{stages}k{chunk // 1024}:{us:.2f}")
                if us * nbytes / eff_bytes < best[0]:
                    best = (us * nbytes / eff_bytes, f"bulk c{ctas} s{stages} {chunk // 1024}K")
        tb = nbytes / ((best[0] - base + 1.1) * 1e-6) / 1e12 if best[0] > base else 0.0
        log(f"mb read {name}: best {best[1]} {best[0]:.2f} us raw (~{best[0] - base + 1.1:.2f} us judge-equivalent, "
            f"{tb:.2f} TB/s); " + " ".join(rows))


def sec_mb_cluster(lab, mb):
    m = mb.mods["cluster"]
    n_per = 4096
    for c in (2, 4, 8):
        ctas = c * (132 // c)
        src = torch.randn(ctas * n_per, device="cuda")
        out = torch.empty((ctas // c) * n_per, device="cuda")
        m.mb_run_cluster(src, out, ctas, c, n_per)
        torch.cuda.synchronize()
        ref = src.view(ctas // c, c, n_per).sum(1).flatten()
        err = float((out - ref).abs().max())
        us = lab.time_graph(lambda: m.mb_run_cluster(src, out, ctas, c, n_per))
        log(f"mb cluster {c}: {ctas} CTAs, 16 KB partial each, max err {err:.1e}, {us:.2f} us raw")
