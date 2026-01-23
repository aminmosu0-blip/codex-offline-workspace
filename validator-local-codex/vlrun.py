#!/usr/bin/env python3
"""vlrun.py: small stdlib client for validator-local.

This avoids curl + manual polling:
- preflight: runs /v1/preflight/from-files
- triad: submits /v1/jobs/from-files and polls /v1/jobs/{id}/summary until completion

No third-party deps (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _json_request(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    req = Request(url, data=data, method=method, headers=headers)
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body.strip() else {}
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}")
    except URLError as e:
        raise RuntimeError(f"Connection error: {e}")


def _tail(base_url: str, job_id: str, name: str, lines: int) -> str:
    url = f"{base_url}/v1/jobs/{job_id}/artifacts/{name}/tail?lines={lines}"
    req = Request(url, method="GET")
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _path(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _common_payload(args) -> dict:
    return {
        "repo_path": _path(args.repo),
        "sha": args.sha,
        "dockerfile_path": _path(args.dockerfile),
        "test_patch_path": _path(args.test_patch),
        "solution_patch_path": _path(args.solution_patch),
        "description_path": _path(args.description) if args.description else None,
    }


def cmd_preflight(args) -> int:
    base = args.base_url.rstrip("/")
    payload = _common_payload(args)
    # drop Nones
    payload = {k: v for k, v in payload.items() if v is not None}
    out = _json_request("POST", f"{base}/v1/preflight/from-files", payload)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out.get("ok") else 2


def cmd_triad(args) -> int:
    base = args.base_url.rstrip("/")
    payload = _common_payload(args)
    payload = {k: v for k, v in payload.items() if v is not None}

    res = _json_request("POST", f"{base}/v1/jobs/from-files", payload)
    job_id = res.get("job_id")
    if not job_id:
        print(json.dumps(res, indent=2, sort_keys=True))
        raise RuntimeError("could not read job_id from response")

    print(f"job_id: {job_id}")

    deadline = time.time() + args.timeout
    last_status = None
    while time.time() < deadline:
        st = _json_request("GET", f"{base}/v1/jobs/{job_id}/summary")
        status = st.get("status")
        phase = st.get("phase")
        code = st.get("code")
        if status != last_status:
            print(f"status={status} phase={phase} code={code}")
            last_status = status
        if status in {"succeeded", "failed"}:
            print(json.dumps(st, indent=2, sort_keys=True))
            if status == "failed":
                for log_name in ("runner.log", "triad.log", "docker_build.log"):
                    try:
                        print("\n--- tail", log_name, "---")
                        print(_tail(base, job_id, log_name, args.tail_lines))
                    except Exception:
                        pass
            return 0 if status == "succeeded" else 3
        time.sleep(args.poll)

    raise RuntimeError(f"timed out after {args.timeout} seconds waiting for job {job_id}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vlrun.py")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--repo", required=True)
        sp.add_argument("--sha", required=True)
        sp.add_argument("--dockerfile", required=True)
        sp.add_argument("--test-patch", required=True)
        sp.add_argument("--solution-patch", required=True)
        sp.add_argument("--description")

    sp1 = sub.add_parser("preflight", help="fast static checks + env validation")
    add_common(sp1)
    sp1.set_defaults(func=cmd_preflight)

    sp2 = sub.add_parser("triad", help="run full triad")
    add_common(sp2)
    sp2.add_argument("--timeout", type=float, default=60 * 30)
    sp2.add_argument("--poll", type=float, default=0.5)
    sp2.add_argument("--tail-lines", type=int, default=160)
    sp2.set_defaults(func=cmd_triad)

    args = p.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
