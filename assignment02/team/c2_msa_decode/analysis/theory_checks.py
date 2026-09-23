"""C2 CPU checks: attention identities, indexing, scale semantics and accounting.

Standard library only. These checks do not implement or benchmark a GPU kernel,
do not emulate Triton MMA rounding, and do not profile the C2 baseline.
"""

import json
import math
import random


def softmax_attention(logits, values):
    if not logits:
        return None
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    denominator = sum(weights)
    out = [sum(w * row[d] for w, row in zip(weights, values)) / denominator
           for d in range(len(values[0]))]
    log2_lse = (maximum + math.log(denominator)) / math.log(2)
    return out, log2_lse


def merge_normalized(partials, dim):
    """Exact-algebra LSE merge; an empty partial is represented by None."""
    nonempty = [p for p in partials if p is not None]
    if not nonempty:
        return None  # Explicitly undefined attention, not a numerical NaN result.
    largest = max(p[1] for p in nonempty)
    weights = [2 ** (p[1] - largest) for p in nonempty]
    denominator = sum(weights)
    return [sum(w * p[0][d] for w, p in zip(weights, nonempty)) / denominator
            for d in range(dim)]


def maxerr(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def check_merge():
    rng = random.Random(20260913)
    logits = [rng.uniform(-30, 30) for _ in range(53)]
    values = [[rng.uniform(-2, 2) for _ in range(5)] for _ in logits]
    gold = softmax_attention(logits, values)[0]
    errors = []
    for chunks in (1, 2, 4, 8, 16, 64):
        width = math.ceil(len(logits) / chunks)
        partials = [softmax_attention(logits[i * width:(i + 1) * width],
                                      values[i * width:(i + 1) * width])
                    for i in range(chunks)]
        errors.append(maxerr(gold, merge_normalized(partials, 5)))
    assert max(errors) < 1e-12
    logits = [0.0, 0.0, 10.0]
    values = [[0.0], [0.0], [1.0]]
    correct = softmax_attention(logits, values)[0][0]
    assert abs(correct - 0.5) > 0.49
    return dict(partition_max_abs_error=max(errors),
                unequal_partition_correct_output=correct, incorrect_unweighted_average=0.5,
                all_empty_contract="undefined; choose/mask a padding convention explicitly")


def check_physical_scale_indexing():
    # Small page=4, D=4 example. Raw values are exactly representable simple
    # numbers; the check isolates indexing and dequantization algebra, not FP8 encoding.
    rng = random.Random(11)
    pages, page_size, dim = 3, 4, 4
    physical_k = [[[rng.choice((-2.0, -1.0, 0.0, 1.0, 2.0)) for _ in range(dim)]
                   for _ in range(page_size)] for _ in range(pages)]
    physical_v = [[[rng.choice((-2.0, -1.0, 0.0, 1.0, 2.0)) for _ in range(dim)]
                   for _ in range(page_size)] for _ in range(pages)]
    ks = [[0.15 + 0.3 * p + 0.025 * t for t in range(page_size)] for p in range(pages)]
    vs = [[0.25 + 0.2 * p + 0.05 * t for t in range(page_size)] for p in range(pages)]
    block_table, selected, kv_len = [2, 0, 1], [2, 0, 1], 10
    query = [0.5, -0.25, 1.0, -0.5]

    def evaluate(kpool, vpool, kscale, vscale, table, order, wrong_logical_scale=False):
        logits, vals = [], []
        for logical in order:
            physical = table[logical]
            for token in range(page_size):
                if logical * page_size + token >= kv_len:
                    continue
                scale_page = logical if wrong_logical_scale else physical
                key = [x * kscale[scale_page][token] for x in kpool[physical][token]]
                val = [x * vscale[scale_page][token] for x in vpool[physical][token]]
                logits.append(sum(x * y for x, y in zip(query, key)) / math.sqrt(dim))
                vals.append(val)
        return softmax_attention(logits, vals)[0]

    gold = evaluate(physical_k, physical_v, ks, vs, block_table, selected)
    wrong = evaluate(physical_k, physical_v, ks, vs, block_table, selected, True)
    order_changed = evaluate(physical_k, physical_v, ks, vs, block_table, selected[::-1])
    # New physical page i contains old physical page permutation[i].
    permutation = [1, 2, 0]
    reverse = {old: new for new, old in enumerate(permutation)}
    permuted = evaluate([physical_k[i] for i in permutation], [physical_v[i] for i in permutation],
                        [ks[i] for i in permutation], [vs[i] for i in permutation],
                        [reverse[i] for i in block_table], selected)
    assert maxerr(gold, permuted) < 1e-12
    assert maxerr(gold, order_changed) < 1e-12
    assert maxerr(gold, wrong) > 1e-3
    return dict(physical_page_permutation_max_abs_error=maxerr(gold, permuted),
                selected_order_permutation_max_abs_error=maxerr(gold, order_changed),
                incorrect_logical_scale_index_max_abs_error=maxerr(gold, wrong))


def check_scale_movement():
    q, raw_k, raw_v = 1.0, [1.0, 1.0], [[0.0], [1.0]]
    token_ks = [1.0, 2.0]
    gold_k = softmax_attention([q * k * s for k, s in zip(raw_k, token_ks)], raw_v)[0][0]
    wrong_k = softmax_attention(raw_k, raw_v)[0][0]
    token_vs = [1.0, 3.0]
    gold_v = softmax_attention([0.0, 0.0], [[v[0] * s] for v, s in zip(raw_v, token_vs)])[0][0]
    wrong_v = softmax_attention([0.0, 0.0], raw_v)[0][0] * token_vs[0]
    assert gold_k > wrong_k and gold_v != wrong_v
    return dict(per_token_k_scale_correct=gold_k, ignored_k_scale=wrong_k,
                per_token_v_scale_correct=gold_v, incorrect_single_post_scale=wrong_v)


def accounting(batch, kv_heads, kv_bytes, dql=1, group=16, dim=128, topk=16, page=128):
    total_q = batch * dql
    heads = kv_heads * group
    target = max(1, min(topk, 256 // max(1, total_q * kv_heads)))
    splits = 1 << (target.bit_length() - 1)
    length = topk * page
    flops = 4 * total_q * heads * length * dim
    kv_payload = 2 * kv_bytes * total_q * kv_heads * length * dim
    q_payload = 2 * total_q * heads * dim
    partial_payload = splits * total_q * heads * (2 * dim + 4)
    # Shape-level read/write payload model, not physical HBM bytes. q/output
    # are BF16; no inter-query KV reuse, cacheline overfetch, metadata or spills.
    total_payload = kv_payload + splits * q_payload + 2 * partial_payload + q_payload
    scale_payload = 8 * total_q * kv_heads * length if kv_bytes == 1 else 0
    return dict(batch=batch, kv_heads=kv_heads, query_heads=heads, dql=dql, kv_bytes=kv_bytes,
                splits=splits, pages_per_split=math.ceil(topk / splits),
                partial_ctas=total_q * kv_heads * splits, merge_ctas=total_q * heads,
                matmul_flops=flops, kv_payload_bytes=kv_payload,
                workspace_bytes=partial_payload, total_payload_bytes=total_payload,
                request_intensity=flops / total_payload,
                per_physical_token_head_scale_bytes=scale_payload,
                request_intensity_with_token_scales=flops / (total_payload + scale_payload),
                ideal_kv_only_intensity=2 * group / kv_bytes,
                ideal_cold_kv_8TBps_us=kv_payload / 8e12 * 1e6,
                saved_partial_roundtrip_fraction=2 * partial_payload / total_payload)


if __name__ == "__main__":
    print(json.dumps(dict(scope="CPU algebra checks and logical payload estimates; no GPU profiling",
                          merge=check_merge(), physical_scales=check_physical_scale_indexing(),
                          scale_movement=check_scale_movement(),
                          accounting=[accounting(b, hkv, size)
                                      for hkv in (4, 1)
                                      for size in (2, 1)
                                      for b in (1, 4, 8, 16)]), indent=2))
