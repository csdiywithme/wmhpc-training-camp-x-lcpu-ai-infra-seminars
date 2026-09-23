#!/usr/bin/env python3
"""Full KDA correctness and alternating complete-forward timing for nextgen.

Every performance row is gated by a same-input baseline comparison. A separate
K2-only timing is labelled explicitly and never used as a full-forward speedup.
"""
import argparse
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import torch


def load_module(directory):
    directory = Path(directory)
    info = json.loads((directory / "manifest.json").read_text())
    if info["status"] != "built":
        raise RuntimeError("Extension manifest is not a successful build")
    spec = importlib.util.spec_from_file_location(info["module"], directory / info["module_path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, info


def error_metrics(actual, ref):
    a, b = actual.float(), ref.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "same_rounding_pass": False}
    diff = a - b
    denom = float(b.square().mean().sqrt())
    rmse = float(diff.square().mean().sqrt())
    peak = float(b.abs().max())
    max_abs = float(diff.abs().max())
    relative_rmse = rmse / max(denom, 1e-8)
    exact=bool(torch.equal(actual, ref))
    unequal=(actual != ref).flatten().nonzero()
    return {"finite": True, "bitwise_equal": exact,
            "different_elements":int(unequal.numel()),
            "first_difference_flat":int(unequal[0,0]) if unequal.numel() else None,
            "rmse": rmse, "relative_rmse": relative_rmse, "max_abs": max_abs,
            "reference_peak": peak, "same_rounding_pass":exact}


def make_case(heads=4, length=33, batch=1, lengths=None, seed=0,
              state_in=True, state_out=True, state_fp32=False, gate="random"):
    torch.manual_seed(seed)
    if lengths is not None:
        batch, length = 1, sum(lengths)
        cu = torch.tensor([0] + list(__import__("itertools").accumulate(lengths)),
                          device="cuda", dtype=torch.int64)
        n = len(lengths)
    else:
        n, cu = batch, None
    shape = (batch, length, heads, 128)
    q, k, v, g = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    if gate != "random":
        g.fill_(-10 if gate == "weak" else 10)
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(heads, 128, device="cuda", dtype=torch.float32)
    state_dtype = torch.float32 if state_fp32 else torch.bfloat16
    initial = torch.randn(n, heads, 128, 128, device="cuda", dtype=state_dtype) * 0.1 if state_in else None
    return dict(q=q, k=k, v=v, g=g, beta=beta, a_log=a_log, dt_bias=dt_bias,
                initial=initial, cu=cu, n=n, state_out=state_out, state_dtype=state_dtype,
                metadata=dict(heads=heads, length=length, batch=batch, lengths=lengths,
                              seed=seed, state_in=state_in, state_out=state_out,
                              state_fp32=state_fp32, gate=gate))


def old_precision_case(seed, lengths, gate):
    # Reuse the OLD exact input generator, including normalized BF16 q/k,
    # A_log/dt_bias distribution and initial-state magnitude. Do not recreate a
    # superficially similar domain with different random draws.
    from run_experiments import inputs
    args,kw=inputs(4,lengths,seed)
    if gate != "random":
        args[3].fill_(-8 if gate == "weak_decay" else 8)
        kw["dt_bias"].zero_()
    q,k,v,g,beta,_,_=args
    return dict(q=q,k=k,v=v,g=g,beta=beta,a_log=kw["A_log"],dt_bias=kw["dt_bias"],
                initial=kw["initial_state"],cu=kw.get("cu_seqlens"),n=len(lengths),
                state_out=True,state_dtype=torch.bfloat16,
                metadata=dict(domain="old_precision_30",heads=4,length=sum(lengths),batch=1,
                              lengths=list(lengths),seed=seed,state_in=True,state_out=True,
                              state_fp32=False,gate=gate))


def long_prefix_cases(seed, lengths, gates, source_length):
    """Generate ONE longest old-input sequence, then take real shared prefixes.

    This intentionally does not call inputs separately at each length: doing so
    would change k/v/g/beta/state random draws and invalidate a drift comparison.
    """
    from run_experiments import inputs
    args,kw=inputs(4,(source_length,),seed)
    q,k,v,g,beta,_,_=args
    for gate in gates:
        gate_source=g.clone()
        bias=kw["dt_bias"].clone()
        if gate != "random":
            gate_source.fill_(-8 if gate=="weak_decay" else 8)
            bias.zero_()
        for length in lengths:
            yield dict(q=q[:,:length].contiguous(),k=k[:,:length].contiguous(),
                       v=v[:,:length].contiguous(),g=gate_source[:,:length].contiguous(),
                       beta=beta[:,:length].contiguous(),a_log=kw["A_log"],dt_bias=bias,
                       initial=kw["initial_state"],cu=None,n=1,state_out=True,state_dtype=torch.bfloat16,
                       metadata=dict(domain="long_shared_prefix",heads=4,length=length,batch=1,
                                     lengths=[length],seed=seed,state_in=True,state_out=True,
                                     state_fp32=False,gate=gate,source_length=source_length,
                                     prefix_policy="single old inputs() source; all inputs and initial state shared"))


def buffers(module, case):
    b, t, h, _ = case["q"].shape
    return dict(out=torch.full_like(case["q"], float("nan")),
                final=torch.full((case["n"], h, 128, 128), float("nan"), device="cuda",
                                 dtype=case["state_dtype"]) if case["state_out"] else None,
                workspace=torch.empty(module.get_workspace_size(b*t, h, case["n"]),
                                      device="cuda", dtype=torch.uint8))


def call(module, case, buf):
    module.fwd(case["q"], case["k"], case["v"], case["g"], case["beta"],
               128 ** -0.5, buf["out"], buf["workspace"], case["a_log"],
               case["dt_bias"], -5.0, initial_state=case["initial"],
               final_state=buf["final"], cu_seqlens=case["cu"])


def workspace_reference(case, buf):
    """Independent Torch K2 recurrence from actual K1 workspace.

    BF16 mm inputs, FP32 matmul accumulation, explicit original BF16 boundaries;
    state product+sum evaluated in FP64 then rounded to FP32 to reproduce an FMA.
    torch.tanh supplies an independent approximate-beta reference (not bit exact
    tanh.approx). This checks recurrence/layout/rounding, not K1 independently.
    """
    b, t, h, d = case["q"].shape
    n, c = case["n"], 16
    lengths = case["metadata"]["lengths"] or [t]*b
    total_tiles = (b*t+c-1)//c+n if case["cu"] is not None else n*((t+c-1)//c)
    nht = h*total_tiles
    ws = buf["workspace"]
    offsets, offset = {}, 0
    for key, per, dtype, tail in [("kd",4096,torch.bfloat16,(16,128)),
                                  ("qd",4096,torch.bfloat16,(16,128)),
                                  ("kr",4096,torch.bfloat16,(16,128)),
                                  ("gt",512,torch.float32,(128,)),
                                  ("inv",512,torch.bfloat16,(16,16)),
                                  ("mqk",512,torch.bfloat16,(16,16))]:
        offsets[key] = ws[offset:offset+nht*per].view(dtype).view(h,total_tiles,*tail)
        offset += nht*per
    state = case["initial"].to(torch.bfloat16).clone() if case["initial"] is not None else torch.zeros(n,h,d,d,device="cuda",dtype=torch.bfloat16)
    result = torch.full_like(case["q"], float("nan")).reshape(-1,h,d)
    values, betas = case["v"].reshape(-1,h,d), case["beta"].reshape(-1,h)
    bf = lambda x: x.to(torch.bfloat16)
    # FP32 inputs here are exact BF16 expansions; disable TF32 explicitly.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    mm = lambda a, b: torch.matmul(a.float(), b.float())
    try:
        tile_base, start = 0, 0
        for seq, length in enumerate(lengths):
            for j in range((length+c-1)//c):
                size=min(c,length-j*c)
                sl=slice(start+j*c,start+j*c+size)
                for head in range(h):
                    wi=tile_base+j
                    p={key:value[head,wi] for key,value in offsets.items()}
                    vv=torch.zeros(c,d,device="cuda",dtype=torch.bfloat16)
                    vv[:size]=values[sl,head]
                    beta=torch.zeros(c,device="cuda",dtype=torch.bfloat16)
                    beta[:size]=bf(torch.tanh(betas[sl,head].float()*0.5)*0.5+0.5)
                    old=state[seq,head]
                    residual=bf(bf(vv-bf(mm(p["kd"],old.t())))*beta[:,None])
                    u=bf(mm(p["inv"],residual))
                    result[sl,head]=bf(bf(mm(p["qd"],old.t()))+bf(mm(p["mqk"],u)))[:size]
                    delta=mm(p["kr"].t(),u).t()
                    state[seq,head]=bf((old.double()*p["gt"].double()[None,:]+delta.double()).float())
            tile_base += (length+c-1)//c
            start += length
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return result.view_as(case["q"]), state.to(case["state_dtype"])


def official_reference(case):
    sys.path.insert(0,"/opt/FlashKDA/tests")
    from torch_ref import torch_ref
    out=torch.empty_like(case["q"])
    final=torch.empty((case["n"],case["q"].shape[2],128,128),device="cuda",
                      dtype=case["state_dtype"]) if case["state_out"] else None
    torch_ref(case["q"],case["k"],case["v"],case["g"],case["beta"],128**-0.5,
              out,A_log=case["a_log"],dt_bias=case["dt_bias"],lower_bound=-5.0,
              initial_state=case["initial"],final_state=final,cu_seqlens=case["cu"])
    return out,final


def naive_reference(case):
    # Fixed upstream naive explicitly computes in FP32; it is not FP64 gold.
    import torch.nn.functional as F
    try:
        from c1_naive import naive_recurrent_kda
    except ModuleNotFoundError:
        naive_path=Path("/opt/c1_reference/fla_kda_ref/naive.py")
        spec=importlib.util.spec_from_file_location("c1_fixed_naive",naive_path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        naive_recurrent_kda=module.naive_recurrent_kda
    b,t,h,d=case["q"].shape
    q=F.normalize(case["q"].float(),dim=-1).reshape(1,-1,h,d)
    k=F.normalize(case["k"].float(),dim=-1).reshape(1,-1,h,d)
    v=case["v"].float().reshape(1,-1,h,d)
    g=case["g"].float().reshape(1,-1,h,d)
    gate=-5.0*torch.sigmoid(case["a_log"].exp()[None,None,:,None]*(g+case["dt_bias"][None,None]))
    beta=case["beta"].float().reshape(1,-1,h).sigmoid()
    lengths=case["metadata"]["lengths"] or [t]*b
    os,hs=[],[]
    start=0
    for seq,length in enumerate(lengths):
        initial=case["initial"][seq:seq+1].float().transpose(-1,-2).contiguous() if case["initial"] is not None else None
        out,final=naive_recurrent_kda(q[:,start:start+length],k[:,start:start+length],
             v[:,start:start+length],gate[:,start:start+length],beta[:,start:start+length],
             scale=128**-0.5,initial_state=initial,output_final_state=True)
        os.append(out)
        hs.append(final.transpose(-1,-2).contiguous())
        start+=length
    return torch.cat(os,dim=1).view(b,t,h,d),torch.cat(hs,dim=0)


def verify_one(candidate, baseline, case, oracle=False, official=True, naive=True):
    cb, bb = buffers(candidate, case), buffers(baseline, case)
    call(baseline,case,bb)
    torch.cuda.synchronize()
    row={"case":case["metadata"]}
    if official:
        oo,oh=official_reference(case)
        row["baseline_official_output"]=error_metrics(bb["out"],oo)
        if bb["final"] is not None:
            row["baseline_official_state"]=error_metrics(bb["final"],oh)
        baseline_ok=all(row[key]["same_rounding_pass"] for key in row if key.startswith("baseline_official_"))
        if not baseline_ok:
            row.update(qualification="BASELINE_ROUNDING_FAILED",same_rounding_pass=False)
            return row,cb,bb
    call(candidate,case,cb)
    torch.cuda.synchronize()
    row["output"]=error_metrics(cb["out"],bb["out"])
    if cb["final"] is not None:
        row["state"]=error_metrics(cb["final"],bb["final"])
    if official:
        row["candidate_official_output"]=error_metrics(cb["out"],oo)
        if cb["final"] is not None:
            row["candidate_official_state"]=error_metrics(cb["final"],oh)
    if naive:
        no,nh=naive_reference(case)
        row["baseline_naive_output"]=error_metrics(bb["out"],no)
        row["candidate_naive_output"]=error_metrics(cb["out"],no)
        if cb["final"] is not None:
            row["baseline_naive_state"]=error_metrics(bb["final"],nh)
            row["candidate_naive_state"]=error_metrics(cb["final"],nh)
        if case["metadata"].get("domain")=="long_shared_prefix":
            length=case["q"].shape[1]
            row["output_windows_1024"]=[]
            for start in range(0,length,1024):
                stop=min(start+1024,length)
                row["output_windows_1024"].append(dict(start=start,end=stop,
                    candidate_baseline=error_metrics(cb["out"][:,start:stop],bb["out"][:,start:stop]),
                    baseline_naive=error_metrics(bb["out"][:,start:stop],no[:,start:stop]),
                    candidate_naive=error_metrics(cb["out"][:,start:stop],no[:,start:stop])))
            index=row["output"].get("first_difference_flat")
            if index is not None:
                token,index2=divmod(index,case["q"].shape[2]*128)
                head,value=divmod(index2,128)
                row["first_output_difference_position"]={"token":token,"head":head,"value":value}
            else:
                row["first_output_difference_position"]=None
    if oracle:
        out, state=workspace_reference(case,cb)
        row["workspace_oracle_output"]=error_metrics(cb["out"],out)
        if cb["final"] is not None:
            row["workspace_oracle_state"]=error_metrics(cb["final"],state)
    row["finite"]=all(value["finite"] for value in row.values() if isinstance(value,dict) and "finite" in value)
    row["same_rounding_pass"]=row["output"]["same_rounding_pass"] and ("state" not in row or row["state"]["same_rounding_pass"])
    row["qualification"]="SAME_ROUNDING_CASE_PASS" if row["same_rounding_pass"] else "NEW_ROUNDING_RESEARCH"
    if not row["finite"]: row["qualification"]="NONFINITE_FAILED"
    return row,cb,bb


def time_loop(fn, iterations):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations): fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)*1000/iterations


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--baseline-build-dir", type=Path)
    p.add_argument("--mode", choices=["smoke","verify","bench","profile"], default="smoke")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--suite",choices=["quick","formal","long"],default="quick")
    p.add_argument("--lengths",default="8192",help="Comma-separated prefix lengths for verify --suite long")
    p.add_argument("--gates",default="random,weak_decay",help="Gate families for verify --suite long")
    p.add_argument("--seed",type=int,default=907,help="Independent long-prefix seed; old 30-case seeds remain unchanged")
    p.add_argument("--source-length",type=int,default=32768,help="Generate this longest source once so separately run prefixes match")
    p.add_argument("--profile-target",choices=["baseline","tcgen05"],default="tcgen05")
    p.add_argument("--profile-heads",type=int,default=96)
    p.add_argument("--profile-length",type=int,default=8192)
    p.add_argument("--iterations",type=int,default=30,help="Minimum calls per sample; pilot raises this toward 10 ms")
    p.add_argument("--max-iterations",type=int,default=1000)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--qualification-json",type=Path,
                   help="Complete same-build SAME_ROUNDING_PASS verification record")
    p.add_argument("--allow-unqualified-timing",action="store_true",
                   help="Allow explicitly labelled DIAGNOSTIC_UNQUALIFIED timings")
    args=p.parse_args()
    if args.iterations<1 or args.max_iterations<args.iterations:
        p.error("Require 1 <= iterations <= max-iterations")
    if args.suite=="long" and args.mode!="verify":
        p.error("--suite long is only available with --mode verify")
    long_lengths=[int(x) for x in args.lengths.split(",") if x]
    long_gates=[x for x in args.gates.split(",") if x]
    if args.suite=="long":
        if not long_lengths or any(x<1 or x>args.source_length for x in long_lengths):
            p.error("Long prefix lengths must be positive and no greater than --source-length")
        if not long_gates or any(x not in ["random","weak_decay","strong_decay"] for x in long_gates):
            p.error("Long gates must be random,weak_decay,strong_decay")
    args.output.mkdir(parents=True,exist_ok=True)
    candidate,build=load_module(args.build_dir)
    if args.baseline_build_dir:
        baseline,baseline_build=load_module(args.baseline_build_dir)
    else:
        baseline,baseline_build=importlib.import_module("flash_kda_C"),{"module":"flash_kda_C"}
    candidate.set_k1_enabled(True)
    if hasattr(baseline,"set_k1_enabled"): baseline.set_k1_enabled(True)
    report={"status":"running","mode":args.mode,"suite":args.suite,"started_unix":time.time(),
            "build":build,"baseline_build":baseline_build,"argv":sys.argv,
            "environment":{"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability(),
                           "torch":torch.__version__,"cuda":torch.version.cuda,"python":platform.python_version()},
            "same_rounding_gate":"finite and torch.equal for output AND state; baseline official prerequisite",
            "new_rounding_contract":"NONE; numerical metrics are research diagnostics, never old-gate PASS",
            "timing_method":"CUDA events around eager full forward; no workspace allocation; includes beta transpose",
            "rows":[]}
    result_kind="long-verify" if args.suite=="long" else args.mode
    path=args.output/f"{result_kind}-{build['variant']}.json"
    save=lambda:path.write_text(json.dumps(report,indent=2))
    save()
    try:
        if args.mode in ["smoke","verify"]:
            case_factories=[lambda:make_case(heads=2,length=17),
                            lambda:make_case(heads=2,length=33,state_in=False),
                            lambda:make_case(heads=2,lengths=[1,17,33],state_fp32=True)]
            if args.mode=="verify" and args.suite!="long":
                case_factories=[]
                for seed in [0,1]:
                    for lengths in [(16,),(17,),(97,),(17,33,65),(1024,)]:
                        for gate in ["random","weak_decay","strong_decay"]:
                            case_factories.append(lambda seed=seed,lengths=lengths,gate=gate:
                                                  old_precision_case(seed,lengths,gate))
                for fp in [False,True]:
                    for si in [False,True]:
                        for so in [False,True]:
                            case_factories.append(lambda fp=fp,si=si,so=so:
                                  make_case(heads=4,lengths=[17,33,65],state_in=si,state_out=so,state_fp32=fp))
            is_long=args.suite=="long"
            report["expected_cases"]=len(long_lengths)*len(long_gates) if is_long else len(case_factories)
            report["expected_old_precision_cases"]=30 if args.mode=="verify" and not is_long else 0
            report["scope"]="LONG_SHARED_PREFIX_SCOPE" if is_long else "SMOKE_SCOPE_ONLY" if args.mode=="smoke" else "OLD_30_PLUS_8_INTERFACES"
            if is_long:
                report["long_configuration"]={"seed":args.seed,"lengths":long_lengths,"gates":long_gates,"source_length":args.source_length,
                    "official_reference":"Not rerun for long sequences; this result does not replace the old 38-case gate",
                    "mathematical_reference":"fixed independent FP32 naive"}
                cases=long_prefix_cases(args.seed,long_lengths,long_gates,args.source_length)
            else:
                cases=(factory() for factory in case_factories)
            for index,case in enumerate(cases):
                row,cb,bb=verify_one(candidate,baseline,case,
                    oracle=not is_long and (args.mode=="smoke" or index in [0,1,2]),official=not is_long)
                report["rows"].append(row)
                save()
                print(json.dumps(row),flush=True)
                if row["qualification"]=="BASELINE_ROUNDING_FAILED":
                    raise AssertionError(f"baseline official rounding case {index} failed; candidate not eligible")
                if not row["same_rounding_pass"] and not (args.output/"first-difference.pt").exists():
                    torch.save({"case":{key:value.cpu() if isinstance(value,torch.Tensor) else value for key,value in case.items()},
                                "candidate_output":cb["out"].cpu(),"baseline_output":bb["out"].cpu(),
                                "candidate_final":cb["final"].cpu() if cb["final"] is not None else None,
                                "baseline_final":bb["final"].cpu() if bb["final"] is not None else None},
                               args.output/"first-difference.pt")
                if not row["finite"]: raise AssertionError(f"nonfinite correctness case {index}")
            report["completed_cases"]=len(report["rows"])
            report["completed_old_precision_cases"]=sum(r["case"].get("domain")=="old_precision_30" for r in report["rows"])
            exact=all(r["same_rounding_pass"] for r in report["rows"])
            exact_status="LONG_SCOPE_SAME_ROUNDING" if is_long else "SAME_ROUNDING_PASS" if args.mode=="verify" else "SMOKE_SCOPE_SAME_ROUNDING"
            report["status"]=exact_status if exact else "NEW_ROUNDING_RESEARCH"
        elif args.mode=="profile":
            case=make_case(heads=args.profile_heads,length=args.profile_length)
            row,cb,bb=verify_one(candidate,baseline,case,official=False,naive=False)
            report["rows"].append(row)
            if not row["finite"]: raise AssertionError("profile case has nonfinite output")
            module,buf=(baseline,bb) if args.profile_target=="baseline" else (candidate,cb)
            for _ in range(5): call(module,case,buf)
            torch.cuda.synchronize()
            region="c1_nextgen_profile_"+args.profile_target
            report["profile"]={"target":args.profile_target,"nvtx_region":region,
                               "scope":"complete forward; NCU may filter the actual K2 kernel separately",
                               "warmup_calls":5,"profiled_calls":1}
            save()
            torch.cuda.nvtx.range_push(region)
            try:
                call(module,case,buf)
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            row["post_profile_output"]=error_metrics(cb["out"],bb["out"])
            if cb["final"] is not None:
                row["post_profile_state"]=error_metrics(cb["final"],bb["final"])
            report["status"]="PROFILE_DIAGNOSTIC_COMPLETED"
        else:
            prior_qualified=False
            if args.qualification_json:
                prior=json.loads(args.qualification_json.read_text())
                prior_qualified=(prior.get("status")=="SAME_ROUNDING_PASS" and prior.get("completed_old_precision_cases")==30
                                 and prior.get("completed_cases")==prior.get("expected_cases")==38
                                 and prior["build"]["generated_launcher_sha256"]==build["generated_launcher_sha256"]
                                 and prior["build"]["k2_source_sha256"]==build["k2_source_sha256"]
                                 and prior["build"]["variant"]==build["variant"] and prior["build"]["arch"]==build["arch"])
                report["qualification_source"]=str(args.qualification_json)
            if not prior_qualified and not args.allow_unqualified_timing:
                raise RuntimeError("Formal timing requires complete same-build qualification; use explicit --allow-unqualified-timing for research diagnostics")
            report["performance_qualification"]="QUALIFIED_PRIOR_GATE" if prior_qualified else "DIAGNOSTIC_UNQUALIFIED"
            shapes=[dict(heads=h,length=1024) for h in [12,96]]+[dict(heads=96,batch=8,length=128)]
            if args.suite=="formal":
                shapes=[]
                for h in [12,96]:
                    shapes.extend([dict(heads=h,length=8192),dict(heads=h,lengths=[1300,547,2048,963,271,3063]),
                                   dict(heads=h,batch=8,length=1024)])
            for cfg in shapes:
                case=make_case(**cfg)
                row,cb,bb=verify_one(candidate,baseline,case,official=False,naive=False)
                report["rows"].append(row)
                save()
                if not row["finite"]: raise AssertionError("benchmark nonfinite correctness gate failed")
                if not row["same_rounding_pass"]:
                    row["performance_qualification"]="DIAGNOSTIC_UNQUALIFIED"
                    if not args.allow_unqualified_timing: raise AssertionError("benchmark same-rounding gate failed")
                cf,bf=lambda:call(candidate,case,cb),lambda:call(baseline,case,bb)
                for _ in range(20): bf(); cf()
                torch.cuda.synchronize()
                pilot={"baseline_us":time_loop(bf,5),"candidate_us":time_loop(cf,5)}
                fastest=max(min(pilot.values()),0.001)
                chosen=min(args.max_iterations,max(args.iterations,math.ceil(10000/fastest)))
                row["iteration_selection"]={"pilot":pilot,"pilot_iterations":5,"target_interval_us":10000,
                    "minimum":args.iterations,"maximum":args.max_iterations,"chosen":chosen,
                    "rule":"ceil(10000/min(baseline_us,candidate_us)), clamped to [minimum,maximum]",
                    "hit_maximum":chosen==args.max_iterations}
                times={"baseline_full_us":[],"candidate_full_us":[]}
                row["paired_order"]=[]
                for repeat in range(args.repeats):
                    order=[("baseline_full_us",bf),("candidate_full_us",cf)]
                    if repeat%2: order.reverse()
                    row["paired_order"].append([name for name,_ in order])
                    for name,fn in order: times[name].append(time_loop(fn,chosen))
                row["timing"]=times
                row["sample_interval_us"]={name:[x*chosen for x in samples] for name,samples in times.items()}
                row["paired_latency_ratios"]=[b/c for b,c in zip(times["baseline_full_us"],times["candidate_full_us"])]
                row["timing_summary"]={name:{"median":statistics.median(samples),"min":min(samples),"max":max(samples),
                                                   "stdev":statistics.stdev(samples) if len(samples)>1 else 0.0}
                                       for name,samples in times.items()}
                row["post_full_timing_output"]=error_metrics(cb["out"],bb["out"])
                if cb["final"] is not None:
                    row["post_full_timing_state"]=error_metrics(cb["final"],bb["final"])
                if not all(value["finite"] for key,value in row.items() if key.startswith("post_full_timing_")):
                    save()
                    raise AssertionError("timed full-forward output became nonfinite")
                timed_exact=all(value["same_rounding_pass"] for key,value in row.items() if key.startswith("post_full_timing_"))
                qualified=prior_qualified and row["same_rounding_pass"]
                qualified=qualified and timed_exact
                row["performance_qualification"]="QUALIFIED" if qualified else "DIAGNOSTIC_UNQUALIFIED"
                ratio_key="full_speedup" if qualified else "diagnostic_latency_ratio"
                row[ratio_key]=statistics.median(times["baseline_full_us"])/statistics.median(times["candidate_full_us"])
                candidate.set_k1_enabled(False)
                try:
                    row["candidate_k2_only_us"]=[time_loop(cf,chosen) for _ in range(args.repeats)]
                finally: candidate.set_k1_enabled(True)
                if hasattr(baseline,"set_k1_enabled"):
                    baseline.set_k1_enabled(False)
                    try: row["baseline_k2_only_us"]=[time_loop(bf,chosen) for _ in range(args.repeats)]
                    finally: baseline.set_k1_enabled(True)
                row["post_timing_output"]=error_metrics(cb["out"],bb["out"])
                if cb["final"] is not None: row["post_timing_state"]=error_metrics(cb["final"],bb["final"])
                save()
                print(json.dumps(row),flush=True)
            report["status"]="BENCH_COMPLETED_QUALIFIED" if all(r["performance_qualification"]=="QUALIFIED" for r in report["rows"]) else "DIAGNOSTIC_UNQUALIFIED"
    except Exception as exc:
        report.update(status="failed",error=repr(exc))
        raise
    finally:
        report["finished_unix"]=time.time()
        save()
    print(f"RESULT_JSON={path}",flush=True)


if __name__=="__main__":
    main()
