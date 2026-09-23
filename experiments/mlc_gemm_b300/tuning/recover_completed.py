"""Read a completed Modal call's retained output; never starts a function."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import modal

async def main(args):
    if args.call_id:
        call_id = args.call_id
        print("READ_EXISTING_CALL", call_id, flush=True)
        call = modal.FunctionCall.from_id(call_id)
        result = await asyncio.wait_for(call.get.aio(timeout=0), 120)
        hashes = {}
        for name, data in result.pop("artifacts").items():
            p = (args.output / "artifacts" / name).resolve()
            if not p.is_relative_to(args.output.resolve()):
                raise ValueError("Artifact path escapes output")
            if p.exists() and p.read_bytes() != data:
                raise ValueError("Existing artifact differs: " + name)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        (args.output / "gpu.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / "recovery.json").write_text(json.dumps({
            "app_id": args.app, "function_call_id": call_id,
            "method": "Read retained FunctionCall output; no new execution",
            "sha256": hashes}, indent=2) + "\n")
        print("RECOVERED", len(hashes), "artifacts", flush=True)
        return
    raise RuntimeError("Supply an existing call ID from modal app logs --show-function-call-id")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--app", required=True)
    p.add_argument("--call-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
