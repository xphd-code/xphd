"""Build docs/xphd_run.pdf from docs/source/xphd_run.tex and `xphd <command> --help`."""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
SOURCE = os.path.join(ROOT, "docs", "source", "xphd_run.tex")
TARGET = os.path.join(ROOT, "docs", "xphd_run.pdf")

GROUPS = [
    ("Archives", ["ibz", "dmats", "generate", "check-archive", "verify-archives", "check"]),
    ("Phonons on the fine mesh", ["matdyn-split", "matdyn-merge", "modes-check", "matdyn",
                                  "mode-labels"]),
    ("Linewidths, self-energy and lineshape", ["linewidth", "sweep", "unfold", "selfenergy",
                                               "lineshape"]),
    ("Symmetry", ["parity", "band-parity", "irreps", "selection-rule"]),
    ("Optics", ["dipoles", "chirality"]),
    ("Transport and photoemission", ["bte", "arpes"]),
    ("Parallel runs", ["hpc-info"]),
]
EXTRA_GROUPS = []


def tex(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def xphd(*args):
    env = dict(os.environ, COLUMNS="100", XPHD_REPORT="0",
               PYTHONPATH=os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", "from xphd.cli import main; main()", *args],
                       capture_output=True, text=True, env=env)
    return r.stdout + r.stderr


def commands():
    out = xphd("--help")
    names = re.search(r"\{([a-z0-9,-]+)\}", out).group(1).split(",")
    helps = {}
    for line in out.splitlines():
        if line.startswith("options:"):
            break
        m = re.match(r"\s{4}([a-z0-9-]+)\s{2,}(.*)", line)
        if m:
            helps[m.group(1)] = m.group(2)
            last = m.group(1)
        elif line.startswith(" " * 20) and helps:
            helps[last] += " " + line.strip()
    return names, helps


def repository():
    from xphd.banner import REPOSITORY
    return REPOSITORY


def version():
    from xphd import __version__
    return __version__


def reference(names):
    lines = []
    listed = [c for _, cs in GROUPS + EXTRA_GROUPS for c in cs]
    groups = GROUPS + EXTRA_GROUPS + [("Other", [c for c in names if c not in listed])]
    for title, cs in groups:
        cs = [c for c in cs if c in names]
        if not cs:
            continue
        lines.append(f"\\subsection*{{{title}}}")
        for c in cs:
            lines.append(f"\\paragraph{{xphd {tex(c)}}}\\label{{ref:{c}}}")
            lines.append("\\begin{Verbatim}[fontsize=\\scriptsize,frame=leftline,"
                         "framerule=0.6pt,rulecolor=\\color{accent},xleftmargin=2pt]")
            lines.append(xphd(c, "--help").rstrip())
            lines.append("\\end{Verbatim}")
    return "\n".join(lines) + "\n"


def main():
    names, helps = commands()
    full = False
    build = tempfile.mkdtemp(prefix="xphd_docs_")
    try:
        shutil.copy(SOURCE, build)
        with open(os.path.join(build, "variant.tex"), "w") as f:
            f.write(f"\\newif\\iffull\\full{'true' if full else 'false'}\n"
                    f"\\newcommand{{\\xphdversion}}{{{tex(version())}}}\n"
                    f"\\newcommand{{\\xphdrepo}}{{\\url{{{repository()}}}}}\n")
        with open(os.path.join(build, "reference.tex"), "w") as f:
            f.write(reference(names))
        with open(os.path.join(build, "extra.tex"), "w") as f:
            f.write("")
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                "xphd_run.tex"], cwd=build, capture_output=True, text=True)
            if r.returncode:
                sys.exit(r.stdout[-3000:])
        os.makedirs(os.path.dirname(TARGET), exist_ok=True)
        shutil.copy(os.path.join(build, "xphd_run.pdf"), TARGET)
        print(f"wrote {TARGET} ({'full' if full else 'release'} version, {len(names)} commands)")
    finally:
        shutil.rmtree(build, ignore_errors=True)


if __name__ == "__main__":
    main()
