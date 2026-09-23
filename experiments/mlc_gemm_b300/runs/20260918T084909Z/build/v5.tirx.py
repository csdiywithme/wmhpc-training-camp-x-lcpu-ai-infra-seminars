# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func
def kernel(A: T.Buffer((4096, 4096), "float16"), B: T.Buffer((4096, 4096), "float16"), D: T.Buffer((4096, 4096), "float16")):
    T.attr({"tirx.device_entry": T.bool(True)})
    bx, by = T.cta_id([32, 32])
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])
    pool_buf = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    tmem_addr = T.decl_scalar(T.uint32, data=pool_buf.data, elem_offset=0, scope="shared.dyn")
    tma_bar = T.decl_buffer((2,), "uint64", data=pool_buf.data, elem_offset=1, scope="shared.dyn", align=8)
    mma_bar = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=3, scope="shared.dyn", align=8)
    Asmem = T.decl_buffer((2, 128, 64), "float16", data=pool_buf.data, elem_offset=512, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[16384:1])))
    Bsmem = T.decl_buffer((2, 128, 64), "float16", data=pool_buf.data, elem_offset=16896, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[16384:1])))
    Dsmem = T.decl_buffer((128, 128), "float16", data=pool_buf.data, elem_offset=33280, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(128, 2, 64):(64, 8192, 1)])))
    with T.attr({"tirx.dyn_smem_bytes": T.int64(99328)}):
        T.evaluate(0)
    if warp_id == 0:
        if lane_id == 0:
            T.ptx.mbarrier.init(T.address_of(mma_bar[0]), 1)
            for s in range(2):
                T.ptx.mbarrier.init(T.address_of(tma_bar[s]), 1)
    if warp_id == 0:
        T.ptx.tcgen05.alloc(T.address_of(tmem_addr), 512, 1)
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()
    tmem = T.decl_buffer((128, 512), scope="tmem", layout=T.TileLayout(T.S[(128, 512):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=tmem_addr[0])
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0
    if warp_id * 32 + lane_id == 0:
        for s in range(2):
            T.tile.copy_async(Asmem[s, 0:128, 0:64], A[bx * 128:bx * 128 + 128, s * 64:s * 64 + 64], dispatch="tma_auto", cta_group=1, mbar=T.address_of(tma_bar[s]))
            T.tile.copy_async(Bsmem[s, 0:128, 0:64], B[by * 128:by * 128 + 128, s * 64:s * 64 + 64], dispatch="tma_auto", cta_group=1, mbar=T.address_of(tma_bar[s]))
            T.ptx.mbarrier.arrive.expect_tx(T.address_of(tma_bar[s]), 32768, "", "", "shared", 0, 0)
    for k in range(64):
        stage: T.int32 = k % 2
        T.ptx.mbarrier.try_wait(T.address_of(tma_bar[stage]), phase_tma)
        if warp_id * 32 + lane_id == 0:
            T.tile.gemm_async(tmem[0:128, 0:128], Asmem[stage, 0:128, 0:64], Bsmem[stage, 0:128, 0:64], False, False, k != 0, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(T.address_of(mma_bar[0]), 1, 0)
        T.ptx.mbarrier.try_wait(T.address_of(mma_bar[0]), phase_mma)
        phase_mma = T.bitwise_xor(phase_mma, 1)
        next_k: T.int32 = k + 2
        if next_k < 64:
            if warp_id * 32 + lane_id == 0:
                T.tile.copy_async(Asmem[stage, 0:128, 0:64], A[bx * 128:bx * 128 + 128, next_k * 64:next_k * 64 + 64], dispatch="tma_auto", cta_group=1, mbar=T.address_of(tma_bar[stage]))
                T.tile.copy_async(Bsmem[stage, 0:128, 0:64], B[by * 128:by * 128 + 128, next_k * 64:next_k * 64 + 64], dispatch="tma_auto", cta_group=1, mbar=T.address_of(tma_bar[stage]))
                T.ptx.mbarrier.arrive.expect_tx(T.address_of(tma_bar[stage]), 32768, "", "", "shared", 0, 0)
        if stage == 1:
            phase_tma = T.bitwise_xor(phase_tma, 1)
    Dreg = T.alloc_local((128,))
    Dreg_f16 = T.alloc_local((128,), "float16")
    Dreg_wg = Dreg.view(128, 128, layout=T.TileLayout(T.S[(128, 128):(1 @ Axis.tid_in_wg, 1)]))
    T.wg.copy_async(Dreg_wg[0:128, 0:128], tmem[0:128, 0:128])
    T.ptx.tcgen05.wait.ld()
    T.cuda.cta_sync()
    T.tile.cast(Dreg_f16[0:128], Dreg[0:128])
    T.tile.copy(Dsmem[warp_id * 32 + lane_id, 0:128], Dreg_f16[0:128])
    T.ptx.fence.proxy_async("shared::cta")
    T.cuda.warpgroup_sync(10)
    if warp_id * 32 + lane_id == 0:
        T.tile.copy_async(D[bx * 128:bx * 128 + 128, by * 128:by * 128 + 128], Dsmem[0:128, 0:128], dispatch="tma_auto")
        T.ptx.cp_async.bulk.commit_group()
        T.ptx.cp_async.bulk.wait_group(0, T.bool(True))
    T.cuda.warpgroup_sync(10)
    T.cuda.cta_sync()
    if warp_id == 0:
        T.ptx.tcgen05.relinquish_alloc_permit(1)
        T.ptx.tcgen05.dealloc(tmem_addr, 512, 1)
