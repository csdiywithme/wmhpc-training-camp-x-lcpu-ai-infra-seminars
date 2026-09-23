"""CPU arithmetic checks for the C1 theoretical analysis (standard library only).

This is neither a FlashKDA implementation nor a GPU benchmark. It checks the
exact-arithmetic chunk identity on a small example, accounting formulas, and
an isolated BF16 state-storage counterexample. It does not emulate NVIDIA MMA.
Run: python3 theory_checks.py
"""

import json
import math
import random
import struct


def transpose(a):
    return [list(row) for row in zip(*a)]


def mm(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def add(a, b, sign=1):
    return [[x + sign * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def max_error(a, b):
    return max(abs(x - y) for ar, br in zip(a, b) for x, y in zip(ar, br))


def bf16_rne(x):
    """Round a finite normal float32 to BF16, round to nearest, ties to even."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def check_chunk_identity():
    rng = random.Random(20260912)
    c, d = 16, 4

    def matrix(m, n):
        return [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(m)]

    def normalize(rows):
        return [[x / math.sqrt(sum(y * y for y in row)) for x in row] for row in rows]

    q, k, v, s0 = normalize(matrix(c, d)), normalize(matrix(c, d)), matrix(c, d), matrix(d, d)
    g = [[-rng.uniform(0.001, 0.15) for _ in range(d)] for _ in range(c)]
    beta = [rng.uniform(0.1, 0.9) for _ in range(c)]
    state = [row[:] for row in s0]
    recurrent_out = []
    for i in range(c):
        state = [[math.exp(g[i][a]) * x for x in row] for a, row in enumerate(state)]
        prediction = mm([k[i]], state)[0]
        delta = [beta[i] * (x - y) for x, y in zip(v[i], prediction)]
        state = add(state, [[ka * dv for dv in delta] for ka in k[i]])
        recurrent_out.append(mm([q[i]], state)[0])

    prefix = [[sum(g[t][a] for t in range(i + 1)) for a in range(d)] for i in range(c)]
    kd = [[k[i][a] * math.exp(prefix[i][a]) for a in range(d)] for i in range(c)]
    ki = [[k[i][a] * math.exp(-prefix[i][a]) for a in range(d)] for i in range(c)]
    qd = [[q[i][a] * math.exp(prefix[i][a]) for a in range(d)] for i in range(c)]
    kr = [[k[i][a] * math.exp(prefix[-1][a] - prefix[i][a]) for a in range(d)] for i in range(c)]
    gram = mm(kd, transpose(ki))
    l = [[beta[i] * gram[i][j] if i > j else 0.0 for j in range(c)] for i in range(c)]
    ident = [[float(i == j) for j in range(c)] for i in range(c)]
    inverse = add(ident, l, -1)
    power = l
    for _ in range(int(math.log2(c)) - 1):
        power = mm(power, power)
        inverse = add(inverse, mm(inverse, power))
    residual = max_error(mm(add(ident, l), inverse), ident)
    rhs = [[beta[i] * x for x in row] for i, row in enumerate(add(v, mm(kd, s0), -1))]
    u = mm(inverse, rhs)
    m = mm(qd, transpose(ki))
    m = [[x if i >= j else 0.0 for j, x in enumerate(row)] for i, row in enumerate(m)]
    chunk_out = add(mm(qd, s0), mm(m, u))
    chunk_state = add([[math.exp(prefix[-1][a]) * x for x in row] for a, row in enumerate(s0)], mm(transpose(kr), u))
    result = dict(inverse_residual_max=residual, output_max_abs=max_error(recurrent_out, chunk_out),
                  state_max_abs=max_error(state, chunk_state))
    assert max(result.values()) < 1e-12, result
    return result


def accounting(c, d=128):
    stages = int(math.log2(c)) - 1
    inverse_flops = 4 * stages * c ** 3
    k1_flops = 4 * c * c * d + inverse_flops
    k2_flops = 6 * c * d * d + 4 * c * c * d
    workspace_bytes = 6 * c * d + 4 * d + 4 * c * c
    # Leading logical bytes, beta counted once in each main kernel; excludes
    # state I/O, beta transpose, gate parameter reads, alignment, and cache effects.
    logical_bytes = 10 * c * d + 4 * c + 2 * workspace_bytes
    input_stage_bytes = 8 * c * d + 128 + 4 * d + 4 * c * c
    # Hypothetically extend the current aligned 3-input/2-output storage scheme;
    # this is a capacity budget, not a claim the fixed-C16 source supports C32/64.
    shared_capacity_bytes = 2 * d * d + max(3 * input_stage_bytes + 4 * c * d, 4 * d * d)
    return dict(chunk=c, exp_min=math.exp(-5 * c), exp_inverse_max=math.exp(5 * c),
                neumann_gemms=2 * stages, neumann_max_power_entry=math.comb(c - 2, c // 2 - 1),
                neumann_partial_p8_corner_abs=math.comb(c - 3, 6),
                inverse_flops=inverse_flops, k1_flops=k1_flops, k2_flops=k2_flops,
                total_flops=k1_flops + k2_flops, flops_per_token=(k1_flops + k2_flops) / c,
                workspace_bytes_per_chunk=workspace_bytes, logical_bytes_per_token=logical_bytes / c,
                logical_arithmetic_intensity=(k1_flops + k2_flops) / logical_bytes,
                state_smem_read_write_bytes_per_token=6 * d * d / c,
                hypothetical_shared_capacity_bytes_before_barriers=shared_capacity_bytes)


def state_storage_counterexample():
    c, length, log_gate = 16, 65536, -1e-4
    decay = math.exp(c * log_gate)
    stored = 1.0
    for _ in range(length // c):
        stored = bf16_rne(stored * decay)
    exact = math.exp(length * log_gate)
    assert stored == 1.0
    return dict(chunk=c, tokens=length, log_gate_per_token=log_gate,
                exact_chunk_decay=decay, rounded_chunk_decay=bf16_rne(decay),
                exact_final_state=exact, bf16_stored_final_state=stored,
                absolute_error=abs(stored - exact),
                scope="isolated mathematical state-storage recurrence; no CUDA kernel run")


if __name__ == "__main__":
    print(json.dumps(dict(scope="CPU theoretical checks; not GPU measurements",
                          chunk_identity=check_chunk_identity(),
                          accounting=[accounting(c) for c in (16, 32, 64)],
                          bf16_state_counterexample=state_storage_counterexample()), indent=2))
