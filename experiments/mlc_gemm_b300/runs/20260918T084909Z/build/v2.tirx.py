# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func
def kernel(A: T.Buffer((4096, 4096), "float16"), B: T.Buffer((4096, 4096), "float16"), D: T.Buffer((4096, 4096), "float16")):
    T.attr({"tirx.device_entry": T.bool(True)})
    bx = T.cta_id([1])
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])
    pool_buf = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    tmem_addr = T.decl_scalar(T.uint32, data=pool_buf.data, elem_offset=0, scope="shared.dyn")
    mma_bar = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=1, scope="shared.dyn", align=8)
    Asmem = T.decl_buffer((128, 64), "float16", data=pool_buf.data, elem_offset=512, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[8192:1])))
    Bsmem = T.decl_buffer((128, 64), "float16", data=pool_buf.data, elem_offset=8704, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[8192:1])))
    with T.attr({"tirx.dyn_smem_bytes": T.int64(33792)}):
        T.evaluate(0)
    if warp_id == 0:
        if lane_id == 0:
            T.ptx.mbarrier.init(T.address_of(mma_bar[0]), 1)
        T.ptx.tcgen05.alloc(T.address_of(tmem_addr), 512, 1)
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()
    tmem = T.decl_buffer((128, 512), scope="tmem", layout=T.TileLayout(T.S[(128, 512):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=tmem_addr[0])
    phase_mma: T.int32 = 0
    Dreg = T.alloc_local((128,))
    Dreg_f16 = T.alloc_local((128,), "float16")
    Dreg_wg = Dreg.view(128, 128, layout=T.TileLayout(T.S[(128, 128):(1 @ Axis.tid_in_wg, 1)]))
    for mt, nt in T.grid(32, 32):
        for kt in range(64):
            T.cta.copy(Asmem[0:128, 0:64], A[mt * 128:mt * 128 + 128, kt * 64:kt * 64 + 64])
            T.cta.copy(Bsmem[0:128, 0:64], B[nt * 128:nt * 128 + 128, kt * 64:kt * 64 + 64])
            T.cuda.cta_sync()
            if warp_id == 0:
                if T.ptx.elect_sync():
                    T.tile.gemm_async(tmem[0:128, 0:128], Asmem[0:128, 0:64], Bsmem[0:128, 0:64], False, False, kt != 0, dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(T.address_of(mma_bar[0]), 1, 0)
            T.ptx.mbarrier.try_wait(T.address_of(mma_bar[0]), phase_mma)
            phase_mma = T.bitwise_xor(phase_mma, 1)
        T.wg.copy_async(Dreg_wg[0:128, 0:128], tmem[0:128, 0:128])
        T.ptx.tcgen05.wait.ld()
        T.tile.cast(Dreg_f16[0:128], Dreg[0:128])
        T.tile.copy(D[mt * 128 + warp_id * 32 + lane_id, nt * 128:nt * 128 + 128], Dreg_f16[0:128])
        T.cuda.cta_sync()
    if warp_id == 0:
        T.ptx.tcgen05.relinquish_alloc_permit(1)
        T.ptx.tcgen05.dealloc(tmem_addr, 512, 1)
