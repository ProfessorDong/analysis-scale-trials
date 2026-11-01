"""Produce the single-file submission the AOAS template requires.

Inlines every \\input{...} recursively (numbers.tex, pending.tex, the sections,
the supplement tables), drops the pending placeholders, and inlines the
compiled .bbl in place of \\bibliography{...} so the file needs no .bib.

    python3 flatten.py paper        -> submit/paper.tex
    python3 flatten.py supplement   -> submit/supplement.tex
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMIT = HERE / "submit"
INPUT = re.compile(r"\\input\{([^}]+)\}")
IFEXISTS = re.compile(r"\\IfFileExists\{([^}]+)\}\{\\input\{([^}]+)\}\}\{[^}]*\}")


def resolve(name: str) -> Path:
    p = HERE / name
    return p if p.suffix else p.with_suffix(".tex")


def inline(text: str, depth: int = 0) -> str:
    if depth > 8:
        raise RecursionError("input nesting too deep")
    text = IFEXISTS.sub(lambda m: f"\\input{{{m.group(2)}}}" if resolve(m.group(2)).exists() else "", text)

    def rep(m):
        name = m.group(1)
        if name == "pending":
            return "% (placeholders removed)"
        path = resolve(name)
        body = path.read_text()
        return f"% ---- begin {name}\n{inline(body, depth + 1)}\n% ---- end {name}\n"
    return INPUT.sub(rep, text)


def main(stem: str) -> None:
    SUBMIT.mkdir(exist_ok=True)
    text = inline((HERE / f"{stem}.tex").read_text())
    bbl = HERE / f"{stem}.bbl"
    if bbl.exists():
        text = re.sub(r"\\bibliography\{[^}]+\}", lambda m: bbl.read_text(), text)
    out = SUBMIT / f"{stem}.tex"
    out.write_text(text)
    for f in ("imsart.cls", "imsart.sty", "spr-ims-nameyear.bst"):
        shutil.copy(HERE / f, SUBMIT / f)
    figs = HERE / "figures"
    (SUBMIT / "figures").mkdir(exist_ok=True)
    for pdf in figs.glob("fig_*.pdf"):
        shutil.copy(pdf, SUBMIT / "figures" / pdf.name)
    print(f"wrote {out} ({len(text.splitlines())} lines); remaining \\input: {len(INPUT.findall(text))}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "paper")
