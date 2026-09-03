"""Verify the hard-coded Supplementary Note numbers in the manuscript.

The main text and the Supplementary Information are separate documents, so
\\ref cannot reach across them and note numbers have to be written out.  That
invites silent drift: inserting a note renumbers every one after it and
nothing in the build complains.  Each hard-coded number in paper.tex carries
a \\SInote{label} marker naming the note it means; this checks the marker
against the actual section order in si_body.tex.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "NaturePortfolio"


def main() -> int:
    body = (ROOT / "si" / "si_body.tex").read_text()
    order = {lab: i for i, lab in
             enumerate(re.findall(r"^\\section\{.*?\}\\label\{(.+?)\}", body, re.M), 1)}

    paper = (ROOT / "paper.tex").read_text()
    refs = re.findall(r"Supplementary Note~(\d+)\\SInote\{(.+?)\}", paper)
    if not refs:
        print("no hard-coded Supplementary Note references found")
        return 0

    bad = 0
    for num, lab in refs:
        want = order.get(lab)
        if want is None:
            print(f"FAIL  {lab}: no such note in si_body.tex")
            bad += 1
        elif int(num) != want:
            print(f"FAIL  {lab}: paper says Note {num}, SI has it at Note {want}")
            bad += 1
        else:
            print(f"ok    {lab} = Supplementary Note {want}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
