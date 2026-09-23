# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func
def kernel(A: T.Buffer((2048, 4096), "float16"), B: T.Buffer((9216, 4096), "float16"), D: T.Buffer((2048, 9216), "float16")):
    T.attr({"tirx.device_entry": T.bool(True)})
    bx = T.cta_id([148])
    cbx, cby = T.cta_id_in_cluster([2, 1])
    wg_id = T.warpgroup_id([3])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])
    pool_buf = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    tmem_addr = T.decl_scalar(T.uint32, data=pool_buf.data, elem_offset=0, scope="shared.dyn")
    buffer = T.decl_buffer((4,), "uint64", data=pool_buf.data, elem_offset=1, scope="shared.dyn", align=8)
    buffer_1 = T.decl_buffer((4,), "uint64", data=pool_buf.data, elem_offset=5, scope="shared.dyn", align=8)
    buffer_2 = T.decl_buffer((2,), "uint64", data=pool_buf.data, elem_offset=9, scope="shared.dyn", align=8)
    ld2mma_buf = T.decl_buffer((2,), "uint64", data=pool_buf.data, elem_offset=11, scope="shared.dyn", align=8)
    Asmem = T.decl_buffer((4, 2, 128, 64), "float16", data=pool_buf.data, elem_offset=512, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[65536:1])))
    Bsmem = T.decl_buffer((4, 128, 64), "float16", data=pool_buf.data, elem_offset=66048, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[32768:1])))
    Dsmem = T.decl_buffer((2, 128, 64), "float16", data=pool_buf.data, elem_offset=98816, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[16384:1])))
    if T.cuda.thread_rank() == 0:
        for i in T.unroll(4):
            T.ptx.mbarrier.init(T.address_of(buffer[i]), 1)
    if T.cuda.thread_rank() == 0:
        for i in T.unroll(4):
            T.ptx.mbarrier.init(T.address_of(buffer_1[i]), 2)
    if T.cuda.thread_rank() == 0:
        for i in T.unroll(2):
            T.ptx.mbarrier.init(T.address_of(buffer_2[i]), 1)
    if T.cuda.thread_rank() == 0:
        for i in T.unroll(2):
            T.ptx.mbarrier.init(T.address_of(ld2mma_buf[i]), 256)
    with T.attr({"tirx.dyn_smem_bytes": T.int64(230400)}):
        T.evaluate(0)
    if wg_id == 0:
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), 512, 2)
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()
    tmem = T.decl_buffer((128, 512), scope="tmem", layout=T.TileLayout(T.S[(128, 512):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=tmem_addr[0])
    buffer_3: T.int32
    buffer_4: T.int32
    buffer_5: T.int32
    tile_scheduler_tile_count: T.int32
    buffer_5 = bx // 2
    tile_scheduler_tile_count = 0
    rem: T.let[T.int32] = bx // 2
    tile_row: T.let[T.int32] = rem % 4
    tile_col: T.let[T.int32] = rem // 4
    buffer_3 = tile_row
    buffer_4 = tile_col
    remote_mbar_ptr: T.let[T.handle("uint64", "shared")] = T.reinterpret(T.handle("uint64", "shared").ty, T.ptx.mapa(T.address_of(buffer[0]), 0, "", "u64", "uint64"))
    buffer_6 = T.decl_buffer((4,), "uint64", data=remote_mbar_ptr, scope="shared")
    remote_mbar_ptr_1: T.let[T.handle("uint64", "shared")] = T.reinterpret(T.handle("uint64", "shared").ty, T.ptx.mapa(T.address_of(ld2mma_buf[0]), 0, "", "u64", "uint64"))
    buffer_7 = T.decl_buffer((2,), "uint64", data=remote_mbar_ptr_1, scope="shared")
    if wg_id == 2:
        if warp_id == 3:
            tma_ps_stage: T.int32
            tma_ps_phase: T.int32
            tma_ps_stage = 0
            tma_ps_phase = 1
            if T.filter(lane_id, T.ptx.elect_sync()):
                while buffer_5 < 144:
                    for k in range(64):
                        T.ptx.mbarrier.try_wait(T.address_of(buffer_1[tma_ps_stage]), T.bitwise_xor(tma_ps_phase, 0))
                        T.tile.copy_async(Asmem[tma_ps_stage, 0, 0:128, 0:64], A[(buffer_3 * 2 * 2 + cbx) * 128:(buffer_3 * 2 * 2 + cbx) * 128 + 128, k * 64:k * 64 + 64], dispatch="tma_auto", cta_group=2, mbar=T.address_of(buffer_6[tma_ps_stage]))
                        T.tile.copy_async(Asmem[tma_ps_stage, 1, 0:128, 0:64], A[(buffer_3 * 2 * 2 + cbx) * 128 + 256:(buffer_3 * 2 * 2 + cbx) * 128 + 256 + 128, k * 64:k * 64 + 64], dispatch="tma_auto", cta_group=2, mbar=T.address_of(buffer_6[tma_ps_stage]))
                        T.tile.copy_async(Bsmem[tma_ps_stage, 0:128, 0:64], B[(buffer_4 * 2 + cbx) * 128:(buffer_4 * 2 + cbx) * 128 + 128, k * 64:k * 64 + 64], dispatch="tma_auto", cta_group=2, mbar=T.address_of(buffer_6[tma_ps_stage]))
                        if cbx == 0:
                            actual_pred: T.bool = T.bool(True)
                            T.ptx.mbarrier.arrive.expect_tx(T.address_of(buffer[tma_ps_stage]), 98304, 0, actual_pred, "", "", "shared::cluster", 1, 1)
                        tma_ps_stage = tma_ps_stage + 1
                        if tma_ps_stage == 4:
                            tma_ps_stage = 0
                            tma_ps_phase = T.bitwise_xor(tma_ps_phase, 1)
                    buffer_5 = buffer_5 + 74
                    tile_scheduler_tile_count = tile_scheduler_tile_count + 1
                    rem_1: T.let[T.int32] = buffer_5
                    tile_row_1: T.let[T.int32] = rem_1 % 4
                    tile_col_1: T.let[T.int32] = rem_1 // 4
                    buffer_3 = tile_row_1
                    buffer_4 = tile_col_1
        else:
            if warp_id < 2:
                mma_ps_stage: T.int32
                mma_ps_phase: T.int32
                mma_ps_stage = 0
                mma_ps_phase = 0
                ld_ps_stage: T.int32
                ld_ps_phase: T.int32
                ld_ps_stage = 0
                ld_ps_phase = 1
                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while buffer_5 < 144:
                            T.ptx.mbarrier.try_wait(T.address_of(ld2mma_buf[warp_id]), T.bitwise_xor(ld_ps_phase, 0))
                            ld_ps_phase = T.bitwise_xor(ld_ps_phase, 1)
                            for k in range(64):
                                T.ptx.mbarrier.try_wait(T.address_of(buffer[mma_ps_stage]), T.bitwise_xor(mma_ps_phase, 0))
                                T.tile.gemm_async(tmem[0:128, warp_id * 256:warp_id * 256 + 256], Asmem[mma_ps_stage, warp_id, 0:128, 0:64], Bsmem[mma_ps_stage, 0:128, 0:64], False, False, k != 0, dispatch="tcgen05", cta_group=2)
                                T.ptx.tcgen05.commit(T.address_of(buffer_1[mma_ps_stage]), 2, 3)
                                mma_ps_stage = mma_ps_stage + 1
                                if mma_ps_stage == 4:
                                    mma_ps_stage = 0
                                    mma_ps_phase = T.bitwise_xor(mma_ps_phase, 1)
                            T.ptx.tcgen05.commit(T.address_of(buffer_2[warp_id]), 2, 3)
                            buffer_5 = buffer_5 + 74
                            tile_scheduler_tile_count = tile_scheduler_tile_count + 1
                            rem_1: T.let[T.int32] = buffer_5
                            tile_row_1: T.let[T.int32] = rem_1 % 4
                            tile_col_1: T.let[T.int32] = rem_1 // 4
                            buffer_3 = tile_row_1
                            buffer_4 = tile_col_1
    else:
        if wg_id < 2:
            wb_ps_stage: T.int32
            wb_ps_phase: T.int32
            wb_ps_stage = 0
            wb_ps_phase = 0
            reg_f16 = T.alloc_local((64,), "float16")
            while buffer_5 < 144:
                T.ptx.mbarrier.try_wait(T.address_of(buffer_2[wg_id]), T.bitwise_xor(wb_ps_phase, 0))
                wb_ps_phase = T.bitwise_xor(wb_ps_phase, 1)
                T.ptx.tcgen05.fence.after_thread_sync()
                for i in T.unroll(4):
                    reg = T.alloc_local((64,))
                    reg_wg = reg.view(128, 64, layout=T.TileLayout(T.S[(128, 64):(1 @ Axis.tid_in_wg, 1)]))
                    T.wg.copy_async(reg_wg[0:128, 0:64], tmem[0:128, wg_id * 256 + i * 64:wg_id * 256 + i * 64 + 64])
                    T.ptx.tcgen05.wait.ld()
                    T.tile.cast(reg_f16[0:64], reg[0:64])
                    T.tile.copy(Dsmem[wg_id, warp_id * 32 + lane_id, 0:64], reg_f16[0:64])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if lane_id == 0:
                            T.tile.copy_async(D[(buffer_3 * 2 * 2 + wg_id * 2 + cbx) * 128:(buffer_3 * 2 * 2 + wg_id * 2 + cbx) * 128 + 128, buffer_4 * 256 + i * 64:buffer_4 * 256 + i * 64 + 64], Dsmem[wg_id, 0:128, 0:64], dispatch="tma_auto")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0, T.bool(True))
                    T.cuda.warpgroup_sync(wg_id + 10)
                actual_pred: T.bool = T.bool(True)
                T.ptx.mbarrier.arrive(T.address_of(ld2mma_buf[wg_id]), 0, actual_pred, "", "", "shared::cluster", 0, 1, 1)
                buffer_5 = buffer_5 + 74
                tile_scheduler_tile_count = tile_scheduler_tile_count + 1
                rem_1: T.let[T.int32] = buffer_5
                tile_row_1: T.let[T.int32] = rem_1 % 4
                tile_col_1: T.let[T.int32] = rem_1 // 4
                buffer_3 = tile_row_1
                buffer_4 = tile_col_1
    T.cuda.cluster_sync()
    if warp_id == 0:
        T.ptx.tcgen05.relinquish_alloc_permit(2)
        T.ptx.tcgen05.dealloc(tmem_addr, 512, 2)
