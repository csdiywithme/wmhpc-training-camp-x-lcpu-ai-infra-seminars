"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T


def make_softmax(M, N):
    # 逻辑行宽和物理线程数是两回事。至少补到一个 warp 的宽度，
    # 再向上取 2 的幂，方便布局推断；例如 N=1000 时 block_N=1024。
    block_N = max(32, 1 << (N - 1).bit_length())
    threads = min(256, block_N)

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Y: T.Tensor((M, N), "float32"),
    ):
        # 总共 M 个 block，每个 block 独立处理一整行。
        with T.Kernel(M, threads=threads) as row:
            # 整个 block 协作持有 block_N 个值，不是每线程各持有这么多。
            values = T.alloc_fragment((block_N,), "float32")
            row_max = T.alloc_fragment((1,), "float32")
            row_sum = T.alloc_fragment((1,), "float32")

            # 1. 读入一行；越界列补 -inf，不参与最大值竞争。
            for j in T.Parallel(block_N):
                values[j] = T.if_then_else(j < N, X[row, j], -T.infinity("float32"))

            # 2. 沿唯一的维度归约，结果写到 row_max[0]。
            # clear=True 自动用 -inf 初始化归约输出。
            # 线程内局部归约、线程间交换与同步由编译器安排。
            T.reduce_max(values, row_max, dim=0, clear=True)

            # 3. 先减整行最大值再取 exp，避免大正数导致 exp 溢出。
            # 对有限输入，补齐列此时变成 exp(-inf)=0。
            for j in T.Parallel(block_N):
                values[j] = T.exp(values[j] - row_max[0])

            # 4. 求分母；补齐列的 0 不影响求和。
            T.reduce_sum(values, row_sum, dim=0, clear=True)

            # 5. 归一化，只写回实际存在的 N 列。
            for j in T.Parallel(block_N):
                if j < N:
                    Y[row, j] = values[j] / row_sum[0]

    return main


@lru_cache(maxsize=128)
def _compiled_softmax(M, N, device_index):
    # 按形状和 GPU 缓存编译结果；out_idx=[1] 指定第二个参数 Y 是输出。
    with torch.cuda.device(device_index):
        return tilelang.compile(make_softmax(M, N), out_idx=[1])


def softmax(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("softmax expects a 2D tensor")
    if x.dtype != torch.float32 or not x.is_cuda:
        raise ValueError("softmax expects a float32 CUDA tensor")

    M, N = x.shape
    if not 1 <= N <= 4096:
        raise ValueError("softmax requires 1 <= N <= 4096")
    if M == 0:
        return torch.empty_like(x)

    # kernel 声明的是紧密排列的二维 Tensor；先处理可能不连续的输入。
    with torch.cuda.device(x.device):
        kernel = _compiled_softmax(M, N, x.device.index)
        return kernel(x.contiguous())
