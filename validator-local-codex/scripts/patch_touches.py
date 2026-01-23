#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

def touches(p: Path) -> list[str]:
    out: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("+++ b/"):
            out.append(line[6:])
    return out

def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print("usage: patch_touches.py /abs/path/to/test.patch [/abs/path/to/solution.patch]")
        raise SystemExit(2)
    for arg in sys.argv[1:]:
        p = Path(arg)
        fs = touches(p)
        print(p.name + " touches:")
        for f in fs:
            print(" - " + f)

if __name__ == "__main__":
    main()
