"""TIRx GEMM tutorial steps 1--3, with explicit full-matrix adapters.

Source: https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_basics/index.html
Public tutorial code, adapted for this independent benchmark experiment.
Upstream revision: ebccca2e5675966f68fb3d4880d4448194bd638d.

All builders compute D = A @ B.T, with fp16 inputs/output and fp32 reduction.
Shapes must be divisible by (128, 128, 64), respectively.

The original v1 computes only (M, N, K) = (128, 128, 64).  Our v1 adapter
serially traverses every output tile in ONE CTA.  It issues each 64-wide MMA
with accum=False, reads that partial sum back from TMEM, and sums the partial
results in fp32 thread registers.  The register reduction and outer tile loops
are benchmark plumbing, NOT optimizations or code claimed by tutorial step 1.

The original v2 computes one 128x128 output tile with a K loop.  Our v2 adapter
again uses ONE CTA to traverse all output tiles serially, but retains the
tutorial's fp32 K accumulation in TMEM.  Thus the v1-to-v2 comparison includes
eliminating the adapter's repeated TMEM reads and register reductions.

v3 follows the tutorial's multi-CTA implementation directly.  Native single-
tile v1/v2 timings and adapted full-matrix timings are different experiments;
the latter must not be presented as unchanged source examples.
"""

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg


def _check_shape(M, N, K):
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 64:
        raise ValueError("M/N must be positive multiples of 128; K of 64")


def hgemm_v1(M, N, K):
    """One CTA, independent K=64 MMAs plus fp32 register reduction adapter."""
    _check_shape(M, N, K)
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dsum = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        # Full-matrix adapter: the original step 1 has neither outer tile
        # loops nor a K reduction.  Only one CTA executes these serial loops.
        for mt in T.serial(M // BLK_M):
            for nt in T.serial(N // BLK_N):
                m_st = T.meta_var(mt * BLK_M)
                n_st = T.meta_var(nt * BLK_N)
                for j in T.unroll(BLK_N):
                    Dsum[j] = T.float32(0)
                for kt in T.serial(K_TILES):
                    Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, kt*BLK_K:(kt+1)*BLK_K])
                    Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, kt*BLK_K:(kt+1)*BLK_K])
                    T.cuda.cta_sync()
                    if warp_id == 0:
                        if T.ptx.elect_sync():
                            Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                          accum=False, dispatch="tcgen05", cta_group=1)
                            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
                    T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                    phase_mma ^= 1
                    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
                    T.ptx.tcgen05.wait.ld()
                    for j in T.unroll(BLK_N):
                        Dsum[j] = Dsum[j] + Dreg[j]
                    # All TMEM readers finish before the next independent MMA
                    # can overwrite the same TMEM columns.
                    T.cuda.cta_sync()
                Tx.cast(Dreg_f16[:], Dsum[:])
                m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
                Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])
                T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


def hgemm_v2(M, N, K):
    """One CTA serially covers M/N tiles, accumulating the K loop in TMEM."""
    _check_shape(M, N, K)
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        # Full-matrix adapter: serialize output tiles in the original single
        # CTA.  Within each tile the tutorial's K loop is unchanged.
        for mt in T.serial(M // BLK_M):
            for nt in T.serial(N // BLK_N):
                m_st = T.meta_var(mt * BLK_M)
                n_st = T.meta_var(nt * BLK_N)
                for kt in T.serial(K_TILES):
                    Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, kt*BLK_K:(kt+1)*BLK_K])
                    Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, kt*BLK_K:(kt+1)*BLK_K])
                    T.cuda.cta_sync()
                    if warp_id == 0:
                        if T.ptx.elect_sync():
                            Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                          accum=(kt != 0), dispatch="tcgen05", cta_group=1)
                            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
                    T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                    phase_mma ^= 1

                Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
                T.ptx.tcgen05.wait.ld()
                Tx.cast(Dreg_f16[:], Dreg[:])
                m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
                Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])
                T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


def hgemm_v3(M, N, K):
    """Tutorial step 3: a 2D CTA grid, one CTA per 128x128 output tile."""
    _check_shape(M, N, K)
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        for kt in T.serial(K_TILES):
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, kt*BLK_K:(kt+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, kt*BLK_K:(kt+1)*BLK_K])
            T.cuda.cta_sync()
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(kt != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
