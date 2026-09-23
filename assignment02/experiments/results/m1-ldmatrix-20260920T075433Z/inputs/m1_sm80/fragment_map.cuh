#pragma once

// A/B 映射来自学员已通过 1.1 判测的实现；D 映射由助教协助修正。
#ifdef __CUDACC__
#define FRAGMENT_HD __host__ __device__
#else
#define FRAGMENT_HD
#endif

FRAGMENT_HD inline int a_row_of(int lane, int i) {
    return (lane >> 2) | ((i & 0x4) << 1);
}
FRAGMENT_HD inline int a_col_of(int lane, int i) {
    return ((lane & 0x3) << 2) | (i & 0x3) | ((i & 0x8) << 1);
}
FRAGMENT_HD inline int b_row_of(int lane, int i) {
    return (i & 0x3) | ((lane & 0x3) << 2) | ((i & 0x4) << 2);
}
FRAGMENT_HD inline int b_col_of(int lane, int /* i */) {
    return lane >> 2;
}
// 每个 lane 的 d0/d1 属于上半行，d2/d3 属于下半行。
FRAGMENT_HD inline int d_row_of(int lane, int i) {
    return (lane >> 2) + (i >= 2 ? 8 : 0);
}
FRAGMENT_HD inline int d_col_of(int lane, int i) {
    return (lane & 3) * 2 + (i & 1);
}

#undef FRAGMENT_HD
