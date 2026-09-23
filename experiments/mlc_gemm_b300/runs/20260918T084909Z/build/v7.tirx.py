# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func
def kernel(A: T.Buffer((4096, 4096), "float16"), B: T.Buffer((4096, 4096), "float16"), D: T.Buffer((4096, 4096), "float16")):
    T.attr({"tirx.device_entry": T.bool(True)})
    bx = T.cta_id([148])
    wg_id = T.warpgroup_id([2])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])
    pool_buf = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    tmem_addr = T.decl_scalar(T.uint32, data=pool_buf.data, elem_offset=0, scope="shared.dyn")
    buffer = T.decl_buffer((2,), "uint64", data=pool_buf.data, elem_offset=1, scope="shared.dyn", align=8)
    buffer_1 = T.decl_buffer((2,), "uint64", data=pool_buf.data, elem_offset=3, scope="shared.dyn", align=8)
    buffer_2 = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=5, scope="shared.dyn", align=8)
    ld2mma_buf = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=6, scope="shared.dyn", align=8)
    Asmem = T.decl_buffer((2, 128, 64), "float16", data=pool_buf.data, elem_offset=512, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[16384:1])))
    Bsmem = T.decl_buffer((2, 128, 64), "float16", data=pool_buf.data, elem_offset=16896, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[16384:1])))
    Dsmem = T.decl_buffer((128, 128), "float16", data=pool_buf.data, elem_offset=33280, scope="shared.dyn", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(128, 2, 64):(64, 8192, 1)])))
    if T.cuda.thread_rank() == 0:
        for i in T.unroll(2):
            T.ptx.mbarrier.init(T.address_of(buffer[i]), 1)
    if T.cuda.thread_rank() == 0:
        for i in T.unroll(2):
            T.ptx.mbarrier.init(T.address_of(buffer_1[i]), 1)
    if T.cuda.thread_rank() == 0:
        for i in T.unroll(1):
            T.ptx.mbarrier.init(T.address_of(buffer_2[i]), 1)
    if T.cuda.thread_rank() == 0:
        for i in T.unroll(1):
            T.ptx.mbarrier.init(T.address_of(ld2mma_buf[i]), 128)
    with T.attr({"tirx.dyn_smem_bytes": T.int64(99328)}):
        T.evaluate(0)
    if wg_id == 0:
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), 512, 1)
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()
    tmem = T.decl_buffer((128, 512), scope="tmem", layout=T.TileLayout(T.S[(128, 512):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=tmem_addr[0])
    buffer_3: T.int32
    buffer_4: T.int32
    buffer_5: T.int32
    tile_scheduler_tile_count: T.int32
    buffer_5 = bx
    tile_scheduler_tile_count = 0
    if T.bitwise_and(T.bool(True), bx < 1024):
        group_id: T.let[T.int32] = bx // 256
        within_group: T.let[T.int32] = bx % 256
        tile_row: T.let[T.int32] = group_id * 8 + within_group % 8
        tile_col: T.let[T.int32] = within_group // 8
        buffer_3 = tile_row
        buffer_4 = tile_col
    else:
        buffer_3 = 0
        buffer_4 = 0
    if wg_id == 1:
        if warp_id == 3:
            tma_ps_stage: T.int32
            tma_ps_phase: T.int32
            tma_ps_stage = 0
            tma_ps_phase = 1
            if T.filter(lane_id, T.ptx.elect_sync()):
                while buffer_5 < 1024:
                    for k in range(64):
                        T.ptx.mbarrier.try_wait(T.address_of(buffer_1[tma_ps_stage]), T.bitwise_xor(tma_ps_phase, 0))
                        T.tile.copy_async(Asmem[tma_ps_stage, 0:128, 0:64], A[buffer_3 * 128:buffer_3 * 128 + 128, k * 64:k * 64 + 64], dispatch="tma_auto", cta_group=1, mbar=T.address_of(buffer[tma_ps_stage]))
                        T.tile.copy_async(Bsmem[tma_ps_stage, 0:128, 0:64], B[buffer_4 * 128:buffer_4 * 128 + 128, k * 64:k * 64 + 64], dispatch="tma_auto", cta_group=1, mbar=T.address_of(buffer[tma_ps_stage]))
                        T.ptx.mbarrier.arrive.expect_tx(T.address_of(buffer[tma_ps_stage]), 32768, "", "", "shared", 0, 0)
                        tma_ps_stage = tma_ps_stage + 1
                        if tma_ps_stage == 2:
                            tma_ps_stage = 0
                            tma_ps_phase = T.bitwise_xor(tma_ps_phase, 1)
                    buffer_5 = buffer_5 + 148
                    tile_scheduler_tile_count = tile_scheduler_tile_count + 1
                    if T.bitwise_and(T.bool(True), buffer_5 < 1024):
                        group_id: T.let[T.int32] = buffer_5 // 256
                        within_group: T.let[T.int32] = buffer_5 % 256
                        tile_row: T.let[T.int32] = group_id * 8 + within_group % 8
                        tile_col: T.let[T.int32] = within_group // 8
                        buffer_3 = tile_row
                        buffer_4 = tile_col
                    else:
                        buffer_3 = 0
                        buffer_4 = 0
        else:
            if warp_id == 0:
                mma_ps_stage: T.int32
                mma_ps_phase: T.int32
                mma_ps_stage = 0
                mma_ps_phase = 0
                ld_ps_stage: T.int32
                ld_ps_phase: T.int32
                ld_ps_stage = 0
                ld_ps_phase = 1
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while buffer_5 < 1024:
                        T.ptx.mbarrier.try_wait(T.address_of(ld2mma_buf[ld_ps_stage]), T.bitwise_xor(ld_ps_phase, 0))
                        ld_ps_phase = T.bitwise_xor(ld_ps_phase, 1)
                        for k in range(64):
                            T.ptx.mbarrier.try_wait(T.address_of(buffer[mma_ps_stage]), T.bitwise_xor(mma_ps_phase, 0))
                            T.tile.gemm_async(tmem[0:128, 0:128], Asmem[mma_ps_stage, 0:128, 0:64], Bsmem[mma_ps_stage, 0:128, 0:64], False, False, k != 0, dispatch="tcgen05", cta_group=1)
                            T.ptx.tcgen05.commit(T.address_of(buffer_1[mma_ps_stage]), 1, 0)
                            mma_ps_stage = mma_ps_stage + 1
                            if mma_ps_stage == 2:
                                mma_ps_stage = 0
                                mma_ps_phase = T.bitwise_xor(mma_ps_phase, 1)
                        T.ptx.tcgen05.commit(T.address_of(buffer_2[0]), 1, 0)
                        buffer_5 = buffer_5 + 148
                        tile_scheduler_tile_count = tile_scheduler_tile_count + 1
                        if T.bitwise_and(T.bool(True), buffer_5 < 1024):
                            group_id: T.let[T.int32] = buffer_5 // 256
                            within_group: T.let[T.int32] = buffer_5 % 256
                            tile_row: T.let[T.int32] = group_id * 8 + within_group % 8
                            tile_col: T.let[T.int32] = within_group // 8
                            buffer_3 = tile_row
                            buffer_4 = tile_col
                        else:
                            buffer_3 = 0
                            buffer_4 = 0
    else:
        if wg_id == 0:
            wb_ps_stage: T.int32
            wb_ps_phase: T.int32
            wb_ps_stage = 0
            wb_ps_phase = 0
            reg_f16 = T.alloc_local((128,), "float16")
            while buffer_5 < 1024:
                T.ptx.mbarrier.try_wait(T.address_of(buffer_2[wb_ps_stage]), T.bitwise_xor(wb_ps_phase, 0))
                wb_ps_phase = T.bitwise_xor(wb_ps_phase, 1)
                T.ptx.tcgen05.fence.after_thread_sync()
                reg = T.alloc_local((128,))
                reg_wg = reg.view(128, 128, layout=T.TileLayout(T.S[(128, 128):(1 @ Axis.tid_in_wg, 1)]))
                T.wg.copy_async(reg_wg[0:128, 0:128], tmem[0:128, 0:128])
                T.ptx.tcgen05.wait.ld()
                T.ptx.mbarrier.arrive(T.address_of(ld2mma_buf[0]), "", "", "shared", 0, 0, 0)
                T.tile.cast(reg_f16[0:128], reg[0:128])
                T.tile.copy(Dsmem[warp_id * 32 + lane_id, 0:128], reg_f16[0:128])
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    if lane_id == 0:
                        T.tile.copy_async(D[buffer_3 * 128:buffer_3 * 128 + 128, buffer_4 * 128:buffer_4 * 128 + 128], Dsmem[0:128, 0:128], dispatch="tma_auto")
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0, T.bool(True))
                T.cuda.warpgroup_sync(10)
                buffer_5 = buffer_5 + 148
                tile_scheduler_tile_count = tile_scheduler_tile_count + 1
                if T.bitwise_and(T.bool(True), buffer_5 < 1024):
                    group_id: T.let[T.int32] = buffer_5 // 256
                    within_group: T.let[T.int32] = buffer_5 % 256
                    tile_row: T.let[T.int32] = group_id * 8 + within_group % 8
                    tile_col: T.let[T.int32] = within_group // 8
                    buffer_3 = tile_row
                    buffer_4 = tile_col
                else:
                    buffer_3 = 0
                    buffer_4 = 0
    T.cuda.cta_sync()
    if warp_id == 0:
        T.ptx.tcgen05.relinquish_alloc_permit(1)
        T.ptx.tcgen05.dealloc(tmem_addr, 512, 1)
