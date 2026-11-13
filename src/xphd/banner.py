"""Internal module of xphd, used by the xphd commands; not run directly."""
from __future__ import annotations

import os
import platform
import random
import sys
import textwrap
import time

ART = {
    'big': "__   _______  _     _____\n\\ \\ / /  __ \\| |   |  __ \\\n \\ V /| |__) | |__ | |  | |\n  > < |  ___/| '_ \\| |  | |\n / . \\| |    | | | | |__| |\n/_/ \\_\\_|    |_| |_|_____/",
    'slant': '   _  __ ____  __    ____\n  | |/ // __ \\/ /_  / __ \\\n  |   // /_/ / __ \\/ / / /\n /   |/ ____/ / / / /_/ /\n/_/|_/_/   /_/ /_/_____/',
    'doom': "__   ________ _    ______\n\\ \\ / /| ___ \\ |   |  _  \\\n \\ V / | |_/ / |__ | | | |\n /   \\ |  __/| '_ \\| | | |\n/ /^\\ \\| |   | | | | |/ /\n\\/   \\/\\_|   |_| |_|___/",
}

TAGLINE = "Exciton-Phonon Dynamics"

DEVELOPERS = (("Rishabh Saraswat", 1), ("Mantasha Ansari", 2),
              ("Rekha Verma", 1), ("Sitangshu Bhattacharya", 2))
AFFILIATIONS = (
    "Nanoscale Electro-thermal Laboratory, Indian Institute of Information "
    "Technology Allahabad, Uttar Pradesh 211015, India",
    "Electronic Structure Theory Group, Department of Electronics and "
    "Communication Engineering, Indian Institute of Information Technology "
    "Allahabad, Uttar Pradesh 211015, India",
)
GROUP = "Developed by the Electronic Structure Theory Group, IIIT Allahabad"
AUTHORS = tuple(name for name, _ in DEVELOPERS)
AFFILIATION = "Indian Institute of Information Technology, Allahabad"

PAPER = ("R. Saraswat, M. Ansari, R. Verma and S. Bhattacharya, "
         "Real-time relaxation dynamics of dark excitons in mirror-symmetric "
         "monolayers, arXiv:2610.12996 (2026)")
REPOSITORY = "https://github.com/xphd-code/xphd"
DOI = "10.5281/zenodo.24184943"
SOFTWARE = (f"xphd, Zenodo, https://doi.org/{DOI}" if DOI else
            f"xphd, GitHub, {REPOSITORY}" if REPOSITORY else "")

ACKNOWLEDGEMENT = (
    "The developers thank the Anusandhan National Research Foundation (ANRF), "
    "India. This work was carried out under grant number CRG/2023/000476. This "
    "work used the computational resources of the PARAM Rudra 20 PF HPC system, "
    "provided by the Centre for Development of Advanced Computing (C-DAC) under "
    "the National Supercomputing Mission (NSM), India, jointly steered by MeitY "
    "and DST.")

WIDTH = 78


def _developer_lines():
    """Developers with [n] against each, then the numbered affiliations."""
    cells = [f"{name} [{n}]" for name, n in DEVELOPERS]
    col = max(len(c) for c in cells) + 6
    out = ["  Developers:"]
    for i in range(0, len(cells), 2):
        out.append(("    " + "".join(c.ljust(col) for c in cells[i:i + 2])).rstrip())
    for n, aff in enumerate(AFFILIATIONS, 1):
        out += textwrap.wrap(aff, WIDTH - 4, initial_indent=f"    [{n}] ",
                             subsequent_indent=" " * (len(f"[{n}] ") + 4))
    return out


def _citation_lines():
    refs = [r for r in (PAPER, SOFTWARE) if r and r.strip()]
    if not refs:
        return []
    out = ["  If you use xphd, please cite:"]
    for r in refs:
        out += textwrap.wrap(r.strip(), WIDTH - 6,
                             initial_indent="    - ", subsequent_indent="      ")
    return out


def _acknowledgement_lines():
    if not (ACKNOWLEDGEMENT and ACKNOWLEDGEMENT.strip()):
        return []
    return ["  Acknowledgements:"] + textwrap.wrap(
        ACKNOWLEDGEMENT.strip(), WIDTH - 4,
        initial_indent="    ", subsequent_indent="    ")


def banner_text(version: str, argv=None, face: str | None = None,
                run_info: bool = True) -> str:
    """The banner as a string; `face` picks the lettering (default: random)."""
    art = ART[face] if face in ART else random.choice(list(ART.values()))
    try:
        import numpy
        npv = numpy.__version__
    except Exception:
        npv = "?"
    lines = [""]
    lines += ["  " + l for l in art.splitlines()]
    lines += ["", f"  Version {version}  --  {TAGLINE}", ""]
    for block in (_developer_lines(), _citation_lines(), _acknowledgement_lines()):
        if block:
            lines += block + [""]
    if run_info:
        lines.append(f"  python {platform.python_version()} | numpy {npv} | "
                     f"{time.strftime('%Y-%m-%d %H:%M')}")
        if argv:
            lines.append("  $ xphd " + " ".join(str(a) for a in argv))
    lines.append("")
    return "\n".join(lines)


def banner(version: str, argv=None, stream=None) -> str:
    """Print the banner unless XPHD_NO_BANNER is set; return its text either way (the run report
    carries the same banner)."""
    text = banner_text(version, argv)
    if not os.environ.get("XPHD_NO_BANNER"):
        print(text, file=stream or sys.stdout, flush=True)
    return text
