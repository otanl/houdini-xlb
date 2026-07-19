"""Persistent JSON-lines worker that owns Warp/XLB in the project Python."""

from __future__ import annotations

import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import TextIO

import numpy as np

from .config import XlbConfig
from .core import Solver, analyze_heightmap
from .protocol import READY, RESPONSE


def handle_request(
    request: dict[str, object],
    solver: Solver | None = None,
    solver_signature: str | None = None,
) -> dict[str, object]:
    operation = request.get("op")
    if operation == "health":
        return {"ok": True, "protocol": 1}
    if operation != "analyze":
        raise ValueError(f"unsupported operation {operation!r}")

    input_path = Path(str(request["heightmap_path"])).resolve()
    cache_dir = Path(str(request["cache_dir"])).resolve()
    config = XlbConfig.from_dict(dict(request.get("config", {})))
    heightmap = np.load(input_path, allow_pickle=False)
    with contextlib.redirect_stdout(sys.stderr):
        result = analyze_heightmap(
            heightmap,
            config,
            cache_dir=cache_dir,
            solver=solver,
            solver_signature=solver_signature,
        )
    response = result.metadata()
    response["ok"] = True
    return response


def serve(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    *,
    solver: Solver | None = None,
    solver_signature: str | None = None,
) -> None:
    print(READY, file=output_stream, flush=True)
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        if line == "shutdown":
            break
        try:
            request = json.loads(line)
            response = handle_request(
                request,
                solver=solver,
                solver_signature=solver_signature,
            )
        except Exception as exc:
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        print(RESPONSE + json.dumps(response, ensure_ascii=True), file=output_stream, flush=True)


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
