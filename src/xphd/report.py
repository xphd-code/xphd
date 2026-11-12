"""PDF report of one xphd run: the xphd banner, command, environment, inputs, parameters,
outputs, plots of the main results, and the log."""
import argparse
import contextlib
import datetime
import hashlib
import io
import os
import platform
import re
import sys
import textwrap
import time
from collections.abc import Mapping

import numpy as np

HASH_LIMIT = 64 * 2**20
LOG_LIMIT = 2 * 2**20
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
LOG_PAGES = 12
SCAN_DEPTH = 3
SCAN_LIMIT = 20000
OUTPUT_FLAGS = ("-o", "--out", "--output", "--outdir", "--out-dir", "--prefix")
MAX_PLOTS = 2
BATCH = 4

INK, ACCENT, MUTED, RULE = "#1B2A41", "#1F5F8B", "#5B6573", "#C9D3E0"
CODE_BG, OK, BAD = "#F2F4F7", "#2E7D4F", "#B3261E"
SANS = ("Montserrat", "Segoe UI", "Inter", "Helvetica Neue", "Arial", "Liberation Sans",
        "DejaVu Sans")
MONO = ("Cascadia Mono", "Consolas", "JetBrains Mono", "Menlo", "Liberation Mono",
        "DejaVu Sans Mono")


def enabled(argv):
    """(argv without --no-report, report wanted)."""
    want = os.environ.get("XPHD_REPORT", "1").strip().lower() not in ("0", "no", "off", "false")
    if "--no-report" in argv:
        argv = [x for x in argv if x != "--no-report"]
        want = False
    if not argv or {"-h", "--help", "--version"} & set(argv):
        want = False
    return argv, want


FORMATS = {"pdf": ("pdf",), "txt": ("txt",), "text": ("txt",), "both": ("pdf", "txt")}


def formats(env=None, system=None):
    """The reports to write."""
    env = os.environ if env is None else env
    v = env.get("XPHD_REPORT", "").strip().lower()
    if v in FORMATS:
        return FORMATS[v]
    system = sys.platform if system is None else system
    return ("txt",) if system.startswith("linux") else ("pdf",)


class _Log:
    """The run's output as the terminal finally shows it."""

    def __init__(self):
        self.done, self.n, self.line, self.pos = [], 0, "", 0

    def write(self, s):
        parts = s.split("\n")
        for i, part in enumerate(parts):
            for j, piece in enumerate(part.split("\r")):
                if j:
                    self.pos = 0
                if piece:
                    self.line = (self.line[:self.pos] + piece
                                 + self.line[self.pos + len(piece):])
                    self.pos += len(piece)
            if i < len(parts) - 1:
                if self.n < LOG_LIMIT:
                    line = self.line.rstrip()[:LOG_LIMIT - self.n]
                    self.done.append(line)
                    self.n += len(line) + 1
                self.line, self.pos = "", 0
        return len(s)

    def getvalue(self):
        text = "\n".join(self.done + ([self.line.rstrip()] if self.line.strip() else []))
        return _ANSI.sub("", text)


class _Tee(io.TextIOBase):
    def __init__(self, stream, buf):
        self.stream, self.buf = stream, buf

    def write(self, s):
        self.buf.write(s)
        return self.stream.write(s)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def fileno(self):
        return self.stream.fileno()


class Run:
    """One command: start() before it runs, finish() after, write() last."""

    def __init__(self, argv):
        self.argv = list(argv)
        self.cmd = argv[0]
        self.t0 = time.time()
        self.cwd = os.getcwd()
        self.log = _Log()
        self.namespace = None
        self.status, self.error = "running", ""
        self.banner = None
        self.inputs = self._inputs()
        self._roots = self._find_roots()
        self._before = self._scan()

    def _paths(self):
        out = []
        for i, tok in enumerate(self.argv[1:], 1):
            if tok.startswith("-"):
                continue
            out.append((tok, self.argv[i - 1] if i > 1 else ""))
        return out

    def _inputs(self):
        rows = []
        for tok, flag in self._paths():
            if flag in OUTPUT_FLAGS or not os.path.exists(tok):
                continue
            rows.append(_describe(tok))
        return rows

    @contextlib.contextmanager
    def capture(self):
        """Tee stdout/stderr into the log and keep the parsed options."""
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, self.log), _Tee(old_err, self.log)
        orig = argparse.ArgumentParser.parse_known_args
        run = self

        def parse_known_args(parser, args=None, namespace=None):
            ns, rest = orig(parser, args, namespace)
            if run.namespace is None or len(vars(ns)) > len(vars(run.namespace)):
                run.namespace = ns
            return ns, rest

        orig_error = argparse.ArgumentParser.error

        def error(parser, message):
            run.usage = True
            return orig_error(parser, message)

        argparse.ArgumentParser.parse_known_args = parse_known_args
        argparse.ArgumentParser.error = error
        try:
            yield self
        finally:
            argparse.ArgumentParser.parse_known_args = orig
            argparse.ArgumentParser.error = orig_error
            sys.stdout, sys.stderr = old_out, old_err

    def finish(self, exc=None):
        self.t1 = time.time()
        if exc is None or (isinstance(exc, SystemExit) and exc.code in (0, None)):
            self.status = "completed"
        else:
            self.status = "FAILED"
            if isinstance(exc, SystemExit):
                self.error = str(exc.code)
            else:
                import traceback
                self.error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.outputs = self._outputs()

    def usage_error(self, exc):
        return isinstance(exc, SystemExit) and getattr(self, "usage", False)

    def _find_roots(self):
        roots = {self.cwd}
        for tok, _ in self._paths():
            p = os.path.abspath(tok)
            for d in (p, os.path.dirname(p), os.path.dirname(os.path.dirname(p))):
                if os.path.isdir(d):
                    roots.add(d)
                    break
        return sorted(roots, key=len)

    def _scan(self):
        """{path."""
        seen = {}
        for root in self._roots:
            for path in _walk(root, SCAN_DEPTH):
                if len(seen) >= SCAN_LIMIT:
                    return seen
                if path in seen:
                    continue
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                seen[path] = (st.st_mtime_ns, st.st_size)
        return seen

    def _outputs(self):
        """Files that are new, or changed, since the run started."""
        after = self._scan()
        out = [p for p, sig in after.items()
               if self._before.get(p) != sig and not p.endswith("_report.pdf")]
        out.sort(key=lambda p: after[p][0])
        return out

    def report_path(self):
        """Next to the main output, named after it; for a batch (more than BATCH data files,
        e.g. generate all) or no output, named after the command and the time, in the batch's
        folder or the working directory."""
        data = [p for p in self.outputs if p.endswith((".npz", ".npy"))] or self.outputs
        stamp = datetime.datetime.fromtimestamp(self.t0).strftime("%Y%m%d_%H%M%S")
        name = f"xphd_{self.cmd}_{stamp}_report.pdf"
        if len(data) > BATCH:
            try:
                where = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in data])
            except ValueError:
                where = self.cwd
            return os.path.join(where, name)
        if data:
            main = max(data, key=os.path.getsize)
            return os.path.splitext(main)[0] + "_report.pdf"
        return os.path.join(self.cwd, name)

    def sections(self, version):
        """[(title, kind, items)]: kind 'kv' (key, value), 'params' (key, value), 'code'
        (lines), 'files' (path, size, date, digest), 'error' (lines)."""
        t0 = datetime.datetime.fromtimestamp(self.t0)
        dt = getattr(self, "t1", time.time()) - self.t0
        run = [("status", self.status), ("started", f"{t0:%Y-%m-%d %H:%M:%S}"),
               ("wall time", _dur(dt)), ("directory", self.cwd)]
        if self.namespace is not None:
            par = [(k, _fmt(v)) for k, v in sorted(vars(self.namespace).items())
                   if k not in ("func", "cmd", "no_report") and not callable(v)]
        else:
            par = [("arguments", " ".join(self.argv[1:]))]
        sec = [("", "banner", _banner_lines(self.banner, version)),
               ("Run", "kv", run),
               ("Command", "code", ["xphd " + " ".join(_q(x) for x in self.argv)]),
               ("Environment", "kv", _environment(version)),
               ("Inputs", "files", list(self.inputs)),
               ("Parameters, with defaults", "params", par),
               ("Outputs", "files", [_describe(p) for p in getattr(self, "outputs", [])[:200]])]
        if self.error:
            sec.append(("Error", "error", self.error.splitlines()[-40:]))
        return sec

    def write(self, version):
        """Write the report: the PDF, the text report or both (see formats()); the paths
        written, the PDF first."""
        path = self.report_path()
        want = formats()
        t0 = datetime.datetime.fromtimestamp(self.t0)
        title = command_title(self.cmd)
        sub = f"Version {version}"
        sections = self.sections(version)
        written, figures, notes, why = [], [], [], ""
        if "pdf" not in want:
            why = ("the text report has no plots; XPHD_REPORT=both writes the PDF with "
                   "them as well")
        else:
            try:
                import matplotlib
                matplotlib.use("Agg", force=False)
                from matplotlib.backends.backend_pdf import PdfPages
            except ImportError:
                why = ("matplotlib is not installed here, so no PDF and nothing drawn "
                       "(pip install matplotlib)")
            else:
                try:
                    figures = self._pdf(path, title, sub, t0, sections, notes,
                                        matplotlib, PdfPages)
                    written.append(path)
                except Exception as e:
                    why = f"the PDF could not be written ({type(e).__name__}: {e})"
                    print(f"\n[xphd] no PDF report: {type(e).__name__}: {e}")
        if "txt" in want or not written:
            txt = path[:-4] + ".txt"
            text = text_report(title, sub, self.status, sections, self.log.getvalue(),
                               figures, notes, why, path if written else None)
            with open(txt, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            written.append(txt)
        return written

    def _pdf(self, path, title, sub, t0, sections, notes, matplotlib, PdfPages):
        """The PDF report; what its result figures show, as text."""
        import matplotlib.pyplot as plt
        doc = _Doc(title, sub, self.status, cmd=self.cmd)
        doc.sections(sections)
        log = _log_lines(self.log.getvalue())
        with matplotlib.rc_context({"font.family": _family("sans")}):
            plots = plot_outputs(self.cmd, self.outputs, notes)
        shown = [_figure_text(f) for f in plots]
        if notes:
            doc.heading("Notes", color=MUTED)
            doc.kv([("plot", n) for n in notes])
        if not plots or doc.fits_code(len(log)):
            doc.code("Log", log)
            figs = doc.figs + plots
        else:
            tail = _Doc(title, sub, None, continued=True)
            tail.code("Log", log)
            figs = doc.figs + plots + tail.figs
        _footers(figs, f"xphd {self.cmd}", t0)
        try:
            for fonttype in (42, 3):
                try:
                    with matplotlib.rc_context({"pdf.fonttype": fonttype}), \
                            PdfPages(path) as pdf:
                        for fig in figs:
                            pdf.savefig(fig)
                        meta = pdf.infodict()
                        meta["Title"] = f"xphd {self.cmd} report"
                        meta["Subject"] = " ".join(self.argv)
                    break
                except Exception:
                    if fonttype == 3:
                        raise
        finally:
            for fig in figs:
                plt.close(fig)
        return shown


def _walk(root, depth):
    try:
        with os.scandir(root) as it:
            for e in it:
                if e.is_file(follow_symlinks=False):
                    yield e.path
                elif e.is_dir(follow_symlinks=False) and depth > 0:
                    yield from _walk(e.path, depth - 1)
    except OSError:
        return


def _describe(p):
    p = os.path.abspath(p)
    try:
        st = os.stat(p)
    except OSError:
        return (p, "missing", "", "")
    when = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
    if os.path.isdir(p):
        try:
            n = len(os.listdir(p))
        except OSError:
            n = 0
        return (p, f"dir, {n} entries", when, "")
    digest = ""
    if st.st_size <= HASH_LIMIT:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(2**20), b""):
                h.update(chunk)
        digest = "sha256 " + h.hexdigest()[:16]
    return (p, _size(st.st_size), when, digest)


def _row(r):
    p, size, when, digest = r
    return f"{p}\n      {size}   {when}   {digest}".rstrip()


def _size(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _dur(s):
    s = int(round(s))
    return f"{s // 3600:d} h {s % 3600 // 60:02d} min {s % 60:02d} s" if s >= 3600 else \
        f"{s // 60:d} min {s % 60:02d} s"


def _q(x):
    return f"'{x}'" if (" " in x or "'" in x) else x


def _fmt(v):
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return str(v)


def _environment(version):
    mods = []
    for name in ("numpy", "scipy", "matplotlib", "netCDF4", "yambopy", "mpi4py"):
        m = sys.modules.get(name)
        if m is not None:
            mods.append(f"{name} {getattr(m, '__version__', '?')}")
    try:
        from . import parallel
        ranks = parallel.size()
    except Exception:
        ranks = 1
    threads = ", ".join(f"{k}={os.environ[k]}" for k in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                        if k in os.environ) or "not set"
    return [("xphd", version), ("python", platform.python_version()),
            ("libraries", ", ".join(mods)),
            ("host", f"{platform.node()}, {platform.system()} {platform.release()}"),
            ("MPI ranks", str(ranks)), ("threads", threads)]


def _wrap(lines, width):
    out = []
    for ln in lines:
        for part in ln.split("\n"):
            out.extend(textwrap.wrap(part, width, subsequent_indent="      ",
                                     break_long_words=True, replace_whitespace=False) or [""])
    return out


def _banner_lines(text, version):
    """The banner as shown in the terminal, without its "$ xphd ..." line (the Command section
    has it) and without the blank lines around it."""
    if not text:
        from .banner import banner_text
        text = banner_text(version)
    lines = [ln.rstrip() for ln in text.splitlines()
             if not ln.lstrip().startswith("$ xphd")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


TEXT_WIDTH = 100


def text_report(title, sub, status, sections, log, figures=(), notes=(), why="",
                pdf=None, width=TEXT_WIDTH):
    """The report as plain text, for a terminal (less, cat) where there is no PDF viewer: the
    same sections as the PDF, each figure by its title, panels and caption, and the whole log
    as the terminal left it."""
    from .banner import GROUP
    bar = "=" * width
    out = [bar, _ends(f"  {title}", (status or "").upper(), width - 2), f"  {sub}",
           f"  {GROUP}", bar, ""]
    out += _flatten(sections, width)
    out.append(_rule("Plots", width))
    if figures:
        out.append(f"  {len(figures)} figure{'s' if len(figures) > 1 else ''} in "
                   f"{os.path.basename(pdf)}, next to this file; copy it to a machine "
                   "with a PDF viewer")
        for i, (head, panels, caption) in enumerate(figures, 1):
            out.append("")
            out += _hang(f"  Figure {i}.  ", head, width)
            for p in panels:
                out += _hang("    ", p, width)
            if caption:
                out += _hang("    ", caption, width)
    elif why:
        out += _hang("  none: ", why, width)
    else:
        out.append("  none for these outputs" +
                   (f"; the full report is {os.path.basename(pdf)}" if pdf else ""))
    out.append("")
    if notes:
        out.append(_rule("Notes", width))
        for n in notes:
            out += _hang("  ", n, width)
        out.append("")
    out.append(_rule("Log", width))
    lines = _clean_log(log)
    out += lines if any(ln.strip() for ln in lines) else ["(nothing printed)"]
    return "\n".join(out).rstrip() + "\n"


def _flatten(sections, width=TEXT_WIDTH):
    """The sections as plain text lines: a rule per section, keys aligned, values wrapped under
    themselves; paths and the command line are never broken, so they can be copied."""
    out = []
    for title, kind, items in sections:
        items = list(items)
        if kind == "banner":
            out += items + [""]
            continue
        out.append(_rule(title, width))
        if kind in ("kv", "params"):
            w = min(max((len(k) for k, _ in items), default=0), 26)
            for k, v in items:
                out += _hang(f"  {k:<{w}}  ", str(v), width)
        elif kind == "files":
            out += [f"  {ln}" for r in items for ln in _row(r).splitlines()] or ["  (none)"]
        else:
            out += [f"  {ln}" for ln in items] or ["  (none)"]
        out.append("")
    return out


def _rule(title, width):
    return f"-- {title} " + "-" * max(3, width - len(title) - 4)


def _ends(left, right, width):
    return left + " " * max(2, width - len(left) - len(right)) + right


def _hang(head, text, width):
    """head + text wrapped to width, the continuation under the text."""
    lines = textwrap.wrap(text, max(20, width - len(head)), break_long_words=False,
                          break_on_hyphens=False) or [""]
    return [head + lines[0]] + [" " * len(head) + ln for ln in lines[1:]]


def _figure_text(fig):
    """(title, panel titles, caption) of a result figure, mathtext as Unicode."""
    st = getattr(fig, "_suptitle", None)
    head = _plain(st.get_text()) if st is not None else ""
    panels = []
    for a in fig.axes:
        t = a.get_title(loc="left") or a.get_title()
        if a.get_visible() and t:
            panels.append(_plain(t))
    return head, panels, _plain(getattr(fig, "_xphd_caption", ""))


_TEX = {"Gamma": "Γ", "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω",
        "Lambda": "Λ", "Phi": "Φ", "Psi": "Ψ", "alpha": "α",
        "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
        "varepsilon": "ε", "eta": "η", "theta": "θ", "kappa": "κ",
        "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
        "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
        "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
        "hbar": "ħ", "sum": "Σ", "langle": "⟨", "rangle": "⟩",
        "times": "×", "cdot": "·", "to": "→", "rightarrow": "→",
        "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥",
        "approx": "≈", "sim": "~", "pm": "±", "infty": "∞",
        "prime": "′", "partial": "∂", "propto": "∝", "circ": "°",
        "AA": "Å", "perp": "⊥", "parallel": "∥",
        ",": " ", ";": " ", "!": "", " ": " ", "quad": "  ", "|": "‖"}
_SUP = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷"
                     "⁸⁹⁻⁺")


def _plain(s):
    """Matplotlib mathtext ($...$) as plain Unicode."""
    if "$" not in s:
        return s

    def math(m):
        t = m.group(1)
        for _ in range(3):
            t = re.sub(r"\\(?:mathbf|mathrm|mathit|mathsf|boldsymbol|text|operatorname)"
                       r"\{([^{}]*)\}", r"\1", t)
        t = re.sub(r"\\([A-Za-z]+)(?: (?=[A-Za-z]))?|\\([,;! |])",
                   lambda c: _TEX.get(c.group(1) or c.group(2), c.group(1) or c.group(2)), t)
        t = re.sub(r"\^\{?([0-9+-]+)\}?", lambda c: c.group(1).translate(_SUP), t)
        t = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", t)
        t = re.sub(r"_\{([^{}]*)\}", r"_\1", t)
        return t.replace("{", "").replace("}", "")

    return re.sub(r"\$([^$]*)\$", math, s)


def _clean_log(text):
    """The log as the terminal left it: of a line redrawn after carriage returns (a progress
    bar), only its last state."""
    shown = [ln.split("\r") for ln in text.rstrip("\n").split("\n")]
    return [next((p for p in reversed(ps) if p.strip()), "") for ps in shown]


def _log_lines(text, width=118, per_page=100):
    lines = _wrap(_clean_log(text), width)
    cap = LOG_PAGES * per_page
    if len(lines) > cap:
        k = cap // 2
        lines = lines[:k] + [f"... {len(lines) - 2 * k} lines left out ..."] + lines[-k:]
    return lines or ["(nothing printed)"]


_FONTS = {}


def _font(kind):
    """The first installed family of SANS or MONO, TrueType files only (they embed as searchable
    text; a CFF .otf may not)."""
    if kind not in _FONTS:
        try:
            from matplotlib import font_manager
            have = {f.name for f in font_manager.fontManager.ttflist
                    if f.fname.lower().endswith(".ttf")}
        except Exception:
            have = set()
        prefs = SANS if kind == "sans" else MONO
        _FONTS[kind] = next((n for n in prefs if n in have),
                            "sans-serif" if kind == "sans" else "monospace")
    return _FONTS[kind]


def _family(kind):
    """The chosen family, then DejaVu (shipped with matplotlib) for any glyph it lacks, e.g.
    Greek letters in a Latin-only font."""
    return [_font(kind), "DejaVu Sans" if kind == "sans" else "DejaVu Sans Mono"]


class _Doc:
    """Text pages laid out top to bottom, A4 portrait, new page when full."""
    W, H = 8.27, 11.69
    L, R, BOT = 0.07, 0.93, 0.05

    def __init__(self, title, subtitle, status, continued=False, header=None, cmd=None):
        self.title, self.subtitle, self.status, self.cmd = title, subtitle, status, cmd
        self.header = header or f"{title}, continued"
        self.figs = []
        self._new(first=not continued)

    def _pt(self, pts):
        return pts / (self.H * 72)

    def _cap(self, width, size, mono=False):
        """characters per line in a width (figure fraction) at a font size."""
        return max(10, int(width * self.W * 72 / ((0.60 if mono else 0.56) * size)))

    def _new(self, first=False):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig = plt.figure(figsize=(self.W, self.H))
        sans = _family("sans")
        if first:
            fig.patches.append(Rectangle((0, 0.906), 1, 0.094, transform=fig.transFigure,
                                         fc=INK, ec="none", zorder=0))
            fig.text(self.L, 0.973, self.title, color="white", fontsize=18, weight="bold",
                     family=sans, va="center")
            if self.status:
                fig.text(self.R, 0.973, self.status.upper(), ha="right", va="center",
                         color="white", fontsize=9, weight="bold", family=sans,
                         bbox=dict(boxstyle="round,pad=0.45",
                                   fc=OK if self.status == "completed" else BAD, ec="none"))
            fig.text(self.L, 0.944, self.subtitle, color=RULE, fontsize=8.5, family=sans,
                     va="center")
            from .banner import GROUP
            fig.text(self.L, 0.923, GROUP, color="#9FB3C8", fontsize=8, family=sans,
                     va="center")
            self.y = 0.888
        else:
            fig.text(self.L, 0.972, self.header, color=ACCENT, fontsize=10,
                     weight="bold", family=sans, va="center")
            self.y = 0.945
        self.fig = fig
        self.figs.append(fig)

    def _ensure(self, pts):
        if self.y - self._pt(pts) < self.BOT:
            self._new()

    def _line(self, x, text, size, color=INK, mono=False, weight="normal"):
        self.fig.text(x, self.y, text, fontsize=size, color=color, weight=weight,
                      family=_family("mono" if mono else "sans"), va="top")

    def heading(self, text, color=ACCENT):
        from matplotlib.lines import Line2D
        self._ensure(62)
        self.y -= self._pt(8)
        self._line(self.L, text, 11, color=color, weight="bold")
        self.y -= self._pt(15)
        self.fig.add_artist(Line2D([self.L, self.R], [self.y, self.y], color=color,
                                   lw=0.7, alpha=0.45, transform=self.fig.transFigure))
        self.y -= self._pt(5)

    def kv(self, rows, size=8.0, key_w=0.15):
        cap = self._cap(self.R - self.L - key_w, size)
        for k, v in rows:
            lines = textwrap.wrap(str(v), cap) or [""]
            lh = 1.4 * size
            self._ensure(lh * len(lines))
            self._line(self.L, k, size, color=MUTED)
            for i, ln in enumerate(lines):
                color = INK
                if k == "status":
                    color = OK if v == "completed" else BAD
                self._line(self.L + key_w, ln, size, color=color,
                           weight="bold" if k == "status" else "normal")
                self.y -= self._pt(lh)

    def params(self, rows, size=7.5):
        short = [(k, v) for k, v in rows if len(v) <= 34]
        long = [(k, v) for k, v in rows if len(v) > 34]
        half = (self.R - self.L) / 2
        lh = 1.4 * size
        for i in range(0, len(short), 2):
            self._ensure(lh)
            for j, (k, v) in enumerate(short[i:i + 2]):
                x = self.L + j * half
                self._line(x, k, size, color=MUTED)
                self._line(x + 0.17, v, size)
            self.y -= self._pt(lh)
        if long:
            self.kv(long, size=size, key_w=0.17)

    def files(self, rows, size=7.0):
        if not rows:
            self._ensure(12)
            self._line(self.L, "none", 8, color=MUTED)
            self.y -= self._pt(12)
            return
        cap = self._cap(self.R - self.L, size, mono=True)
        for p, sz, when, digest in rows:
            lines = textwrap.wrap(p, cap, break_long_words=True) or [p]
            self._ensure(1.35 * size * len(lines) + 12)
            for ln in lines:
                self._line(self.L, ln, size, mono=True)
                self.y -= self._pt(1.35 * size)
            meta = "  \u00b7  ".join(x for x in (sz, when, digest) if x)
            self._line(self.L + 0.02, meta, 6.8, color=MUTED)
            self.y -= self._pt(12)

    def _lh(self, size):
        return 1.32 * size

    def fits_code(self, n, size=6.8):
        """would n log lines (with their heading) fit on this page?"""
        need = self._pt(36 + n * self._lh(size) + 10)
        return self.y - need >= self.BOT

    def code(self, title, lines, size=6.8, color=ACCENT, wrap=True):
        from matplotlib.patches import Rectangle
        if title:
            self.heading(title, color=color)
        if wrap:
            lines = _wrap(lines, self._cap(self.R - self.L - 0.02, size, mono=True))
        lh = self._lh(size)
        i = 0
        while i < len(lines):
            self._ensure(2 * lh + 6)
            room = int((self.y - self.BOT - self._pt(6)) / self._pt(lh))
            chunk = lines[i:i + max(1, room)]
            top = self.y + self._pt(3)
            for ln in chunk:
                self._line(self.L + 0.01, ln, size, mono=True)
                self.y -= self._pt(lh)
            self.fig.patches.append(Rectangle((self.L - 0.004, self.y - self._pt(2)),
                                              self.R - self.L + 0.008,
                                              top - self.y + self._pt(2),
                                              transform=self.fig.transFigure, fc=CODE_BG,
                                              ec="none", zorder=-1))
            self.y -= self._pt(6)
            i += len(chunk)

    def masthead(self, lines):
        """The banner in two columns."""
        from matplotlib.patches import Rectangle
        from . import banner as B
        if not lines:
            return
        art, rest = [], []
        for ln in lines:
            if not art and not ln.strip():
                continue
            (rest if (rest or (art and not ln.strip())) else art).append(ln)
        version = next((ln.strip() for ln in rest if ln.strip().startswith("Version ")), "")
        run = next((ln.strip() for ln in rest if ln.strip().startswith("python ")), "")
        self._ensure(300)
        top = self.y
        split = self.L + 0.54 * (self.R - self.L)
        xl, wl = self.L + 0.012, split - self.L - 0.055
        self.y -= self._pt(6)

        def put(text, size, color=INK, weight="normal", mono=False, x=None, gap=1.42):
            self.fig.text(xl if x is None else x, self.y, text, fontsize=size, color=color,
                          weight=weight, family=_family("mono" if mono else "sans"), va="top")
            self.y -= self._pt(gap * size)

        try:
            renderer = self.fig.canvas.get_renderer()
        except AttributeError:
            renderer = None

        def wrap(text, width, size):
            """Lines of `text` no wider than `width` (figure fraction), measured in the font in
            use, so a wide font (Montserrat) still fits."""
            if renderer is None:
                return textwrap.wrap(text, max(10, int(width * self.W * 72 / (0.62 * size))))
            limit = width * self.W * self.fig.dpi
            probe = self.fig.text(0, 0, "", fontsize=size, family=_family("sans"))
            lines, cur = [], ""
            for word in text.split():
                trial = f"{cur} {word}".strip()
                probe.set_text(trial)
                if cur and probe.get_window_extent(renderer).width > limit:
                    lines.append(cur)
                    cur = word
                else:
                    cur = trial
            probe.remove()
            return lines + ([cur] if cur else [])

        def para(text, size, color=INK, indent=0.0):
            for ln in wrap(text, wl - indent, size):
                put(ln, size, color=color, x=xl + indent)

        for ln in art:
            put(ln.rstrip(), 6.6, color=ACCENT, weight="bold", mono=True, gap=1.15)
        self.y -= self._pt(5)
        if version:
            put(version.replace("  --  ", "  \u2014  ").replace("-Phonon", "\u2013Phonon"),
                9.0, weight="bold", gap=1.75)
        put("Developers", 7.6, color=ACCENT, weight="bold", gap=1.6)
        half = wl / 2
        for i in range(0, len(B.DEVELOPERS), 2):
            for j, (name, n) in enumerate(B.DEVELOPERS[i:i + 2]):
                self.fig.text(xl + 0.01 + j * half, self.y, f"{name}$^{{{n}}}$", fontsize=7.2,
                              color=INK, family=_family("sans"), va="top")
            self.y -= self._pt(1.55 * 7.2)
        self.y -= self._pt(2)
        for n, aff in enumerate(B.AFFILIATIONS, 1):
            first = True
            for ln in wrap(aff, wl - 0.025, 6.3):
                if first:
                    self.fig.text(xl + 0.01, self.y, f"$^{{{n}}}$", fontsize=6.3, color=MUTED,
                                  family=_family("sans"), va="top")
                    first = False
                put(ln, 6.3, color=MUTED, x=xl + 0.022)
        refs = [r for r in (B.PAPER, B.SOFTWARE) if r and r.strip()]
        if refs:
            self.y -= self._pt(6)
            put("If you use xphd, please cite", 7.6, color=ACCENT, weight="bold", gap=1.6)
            for r in refs:
                y0 = self.y
                para(r.strip(), 6.5, indent=0.022)
                self.fig.text(xl + 0.01, y0, "\u2022", fontsize=6.5, color=INK,
                              family=_family("sans"), va="top")
        if B.ACKNOWLEDGEMENT and B.ACKNOWLEDGEMENT.strip():
            self.y -= self._pt(6)
            put("Acknowledgements", 7.6, color=ACCENT, weight="bold", gap=1.6)
            para(B.ACKNOWLEDGEMENT.strip(), 6.5, indent=0.01)
        if run:
            self.y -= self._pt(6)
            put(run, 6.3, color=MUTED, x=xl + 0.01)
        bottom = self.y - self._pt(6)
        h = top - bottom - self._pt(8)
        ax = self.fig.add_axes([split, bottom + self._pt(4), self.R - split - 0.005, h])
        _flowchart(ax, self.cmd)
        self.fig.lines.append(__import__("matplotlib").lines.Line2D(
            [split - 0.012, split - 0.012], [bottom + self._pt(8), top - self._pt(8)],
            color=RULE, lw=0.6, transform=self.fig.transFigure))
        self.y = bottom
        self.fig.patches.append(Rectangle((self.L - 0.004, self.y), self.R - self.L + 0.008,
                                          top - self.y, transform=self.fig.transFigure,
                                          fc=CODE_BG, ec="none", zorder=-1))
        self.y -= self._pt(4)

    def sections(self, sections):
        for title, kind, items in sections:
            if kind == "banner":
                self.masthead(items)
                continue
            if kind == "code":
                self.code(title, items, size=8.0)
                continue
            if kind == "error":
                self.code(title, items, size=7.0, color=BAD)
                continue
            self.heading(title)
            getattr(self, {"kv": "kv", "params": "params", "files": "files"}[kind])(items)


TITLES = {
    "ibz": "Irreducible Q-points", "dmats": "Rotation matrices",
    "generate": "Exciton\u2013phonon archives", "check-archive": "Archive check",
    "verify-archives": "Archive set check", "check": "Archive diagnostics",
    "matdyn": "Fine-mesh phonons", "matdyn-split": "matdyn chunks",
    "matdyn-merge": "matdyn merge", "modes-check": "Phonon modes check",
    "mode-labels": "Phonon mode labels", "linewidth": "Linewidths",
    "sweep": "Linewidths over the zone", "unfold": "Unfold to the full zone",
    "selfenergy": "Self-energy", "lineshape": "Absorption lineshape",
    "parity": "Exciton parity", "band-parity": "Band parity",
    "irreps": "Irreducible representations", "selection-rule": "Selection rule",
    "dipoles": "Exciton dipoles", "chirality": "Helicity and valley polarization",
    "bte": "BTE", "arpes": "ARPES", "hpc-info": "HPC info",
}


def command_title(cmd):
    """The name in the report's title bar."""
    return TITLES.get(cmd, cmd.replace("-", " ").capitalize())


FAMILIES = {"qe": ("#E8EEFB", "#8CA0D8"), "yambo": ("#FBE4E1", "#D9776F"),
            "letz": ("#F0E8F6", "#A98BC4"), "xphd": ("#E3EFE7", "#4C9A6A")}
HERE = ("#FFE9C7", "#E07B00")
PIPE = [
    ("scf", "qe", 50, 5, 30, 8, ["pw.x  scf"], False),
    ("ph", "qe", 16, 20, 32, 8, ["ph.x  phonons"], False),
    ("nscf", "qe", 80, 20, 36, 8, ["pw.x  nscf \u2192 SAVE"], False),
    ("q2r", "qe", 16, 38, 32, 11, ["q2r.x \u00b7 matdyn.x", "fine-mesh phonons"], False),
    ("letz", "letz", 48, 38, 26, 11, ["LetzElPhC", "electron\u2013phonon"], False),
    ("yambo", "yambo", 80, 38, 32, 11, ["yambo", "GW + BSE at all Q"], False),
    ("matdyn", "xphd", 16, 59, 32, 11, ["matdyn \u00b7 mode-labels", "phonon energies"], False),
    ("gen", "xphd", 64, 59, 44, 11, ["ibz \u00b7 dmats \u00b7 generate",
                                     "exciton\u2013phonon archives"], False),
    ("lw", "xphd", 12.5, 82, 24, 12, ["linewidth", "sweep \u00b7 lineshape"], False),
    ("bte", "xphd", 37.5, 82, 22, 12, ["bte", "populations"], False),
    ("parity", "xphd", 62.5, 82, 24, 12, ["parity \u00b7 irreps", "selection-rule"], False),
    ("dip", "xphd", 87.5, 82, 22, 12, ["dipoles", "chirality"], False),
    ("arpes", "xphd", 37.5, 102, 22, 11, ["arpes", "photoemission"], False),
]
PIPE_BOX = {
    **dict.fromkeys(("ibz", "dmats", "generate", "check-archive", "verify-archives", "check"),
                    "gen"),
    **dict.fromkeys(("matdyn", "matdyn-split", "matdyn-merge", "modes-check", "mode-labels"),
                    "matdyn"),
    **dict.fromkeys(("linewidth", "sweep", "unfold", "selfenergy", "lineshape"), "lw"),
    **dict.fromkeys(("parity", "band-parity", "irreps", "selection-rule"), "parity"),
    **dict.fromkeys(("dipoles", "chirality"), "dip"),
    "bte": "bte", "arpes": "arpes",
}


def _fit(text, ax, width):
    """Shrink `text` until it is narrower than `width` (data units)."""
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        return
    ax.apply_aspect()
    limit = abs(ax.transData.transform((width, 0))[0] - ax.transData.transform((0, 0))[0])
    size = text.get_fontsize()
    for _ in range(8):
        w = text.get_window_extent(renderer).width
        if w <= limit or size <= 4.0:
            break
        size = max(4.0, size * limit / w * 0.97)
        text.set_fontsize(size)


def _flowchart(ax, cmd=None):
    """Quantum ESPRESSO, LetzElPhC and yambo above, xphd below; the box that holds `cmd` is
    marked."""
    from matplotlib.patches import FancyBboxPatch
    box = {b[0]: b for b in PIPE}
    here = PIPE_BOX.get(cmd)
    ax.set_xlim(-1, 101)
    ax.set_ylim(114, -1)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.axis("off")
    sans = _family("sans")

    def edge(k, side, d=0.0):
        _, _, cx, cy, w, h, _, _ = box[k]
        return {"t": (cx + d, cy - h / 2), "b": (cx + d, cy + h / 2),
                "l": (cx - w / 2, cy + d), "r": (cx + w / 2, cy + d)}[side]

    def arrow(pts, dashed=False):
        pts = [tuple(map(float, p)) for p in pts]
        ls = (0, (2.5, 2)) if dashed else "solid"
        if len(pts) > 2:
            ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=MUTED,
                    lw=0.7, ls=ls, solid_joinstyle="miter", zorder=1)
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, ls=ls,
                                    mutation_scale=6, shrinkA=0, shrinkB=0), zorder=1)

    for key, fam, cx, cy, w, h, lines, dashed in PIPE:
        fc, ec = HERE if key == here else FAMILIES[fam]
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec,
                                    lw=1.6 if key == here else 0.8,
                                    ls=(0, (3, 2)) if dashed else "solid", zorder=2))
        n = len(lines)
        for i, ln in enumerate(lines):
            t = ax.text(cx, cy + (i - (n - 1) / 2) * 4.6, ln, ha="center", va="center",
                        fontsize=6.2 if i == 0 else 5.5, weight="bold" if i == 0 else "normal",
                        color=INK if i == 0 else MUTED, family=sans, zorder=3)
            _fit(t, ax, w - 3.0)
    y1, y2 = 13.0, 71.0
    arrow([edge("scf", "b"), (50, y1), (16, y1), edge("ph", "t")])
    arrow([(50, y1), (80, y1), edge("nscf", "t")])
    arrow([edge("ph", "b"), edge("q2r", "t")])
    arrow([edge("ph", "r"), (48, 20), edge("letz", "t")])
    arrow([edge("nscf", "b", -10), edge("letz", "r", -3)])
    arrow([edge("nscf", "b"), edge("yambo", "t")])
    arrow([edge("q2r", "b"), edge("matdyn", "t")])
    arrow([edge("letz", "b"), (48, edge("gen", "t")[1])])
    arrow([edge("yambo", "b"), (80, edge("gen", "t")[1])])
    ax.plot([64, 64, 12.5], [edge("gen", "b")[1], y2, y2], color=MUTED, lw=0.7, zorder=1)
    ax.plot([64, 87.5], [y2, y2], color=MUTED, lw=0.7, zorder=1)
    for k in ("lw", "bte", "parity", "dip"):
        arrow([(box[k][2], y2), edge(k, "t")])
    arrow([edge("matdyn", "b", -5), (box["lw"][2] - 1.5, edge("lw", "t")[1])], dashed=True)
    arrow([edge("matdyn", "b", 9), (box["bte"][2] - 6, edge("bte", "t")[1])], dashed=True)
    arrow([edge("bte", "b"), edge("arpes", "t")])
    keys = [(FAMILIES["qe"], "Quantum ESPRESSO", MUTED), (FAMILIES["yambo"], "yambo", MUTED),
            (FAMILIES["letz"], "LetzElPhC", MUTED), (FAMILIES["xphd"], "xphd", MUTED)]
    if here:
        keys.append((HERE, "this run", HERE[1]))
    for i, ((fc, ec), label, color) in enumerate(keys):
        x, y = 55, 93 + i * 4.6
        ax.add_patch(FancyBboxPatch((x, y - 1.6), 3.2, 3.2,
                                    boxstyle="round,pad=0,rounding_size=0.7", fc=fc, ec=ec,
                                    lw=1.1 if label == "this run" else 0.7, zorder=2))
        ax.text(x + 5, y, label, va="center", fontsize=5.4, color=color, family=sans,
                weight="bold" if label == "this run" else "normal")


def _footers(figs, title, t0):
    n = len(figs)
    for i, fig in enumerate(figs, 1):
        fig.text(0.07, 0.018, f"{title}  \u00b7  {t0:%Y-%m-%d %H:%M}", fontsize=7,
                 color=MUTED, family=_family("sans"))
        fig.text(0.93, 0.018, f"{i} / {n}", fontsize=7, color=MUTED, ha="right",
                 family=_family("sans"))


def plot_outputs(cmd, outputs, notes=None):
    """Figures of the main results the run wrote: archive, parity, linewidth, sweep, lineshape,
    BTE and ARPES files, recognised by their keys."""
    files = [p for p in outputs if p.endswith(".npz") and os.path.isfile(p)]
    order = (sorted(files) if len(files) > BATCH else
             sorted(files, key=os.path.getsize, reverse=True))
    maps, rest = [], []
    for p in order:
        t = _delay_map(p)
        (maps if t is not None else rest).append((t, p))
    extra = []
    if len(maps) >= 2:
        try:
            extra.append(_plot_arpes_delays(sorted(maps)))
        except Exception as e:
            if notes is not None:
                notes.append(f"maps at {len(maps)} delays: not drawn ({type(e).__name__}: {e})")
        rest = [p for _, p in rest]
    else:
        rest = [p for _, p in maps + rest]
    figs = []
    for p in rest:
        if len(figs) + len(extra) >= MAX_PLOTS:
            break
        try:
            with np.load(p, allow_pickle=False) as z:
                fig = _plot_known(_Arrays(z), p)
        except Exception as e:
            if notes is not None:
                notes.append(f"{os.path.basename(p)}: not drawn ({type(e).__name__}: {e})")
            continue
        if fig is not None:
            figs.append(fig)
    return figs + extra


def _delay_map(p):
    """The delay (fs) of a time-resolved ARPES map, None for anything else."""
    try:
        with np.load(p, allow_pickle=False) as z:
            k = set(z.files)
            if not ("omega" in k and ("I_fine" in k or "I_path" in k) and "time" in k):
                return None
            if "T" in k and np.isfinite(float(z["T"])):
                return None
            return float(z["time"])
    except Exception:
        return None


class _Arrays(Mapping):
    """The arrays of an open .npz, each read on first use and then kept, so recognising a file
    by its keys costs nothing."""

    def __init__(self, z):
        self._z, self._got = z, {}

    def __getitem__(self, key):
        if key not in self._got:
            self._got[key] = self._z[key]
        return self._got[key]

    def __contains__(self, key):
        return key in self._z.files

    def __iter__(self):
        return iter(self._z.files)

    def __len__(self):
        return len(self._z.files)


def _plot_known(d, p):
    name = os.path.basename(p)
    k = set(d)
    if {"G_grid", "E_m_grid", "hw_grid", "q_red"} <= k:
        return _plot_archive(d, name)
    if {"gamma_A", "T", "eps2_full"} <= k:
        return _plot_lineshape(d, name)
    if {"T", "total", "E_n"} <= k:
        return _plot_linewidth(d, name)
    if {"t", "F", "E"} <= k:
        return _plot_bte(d, name)
    if "omega" in k and ("I_fine" in k or "I_path" in k):
        return _plot_arpes(d, name)
    if {"times", "omega", "edc_zone_t"} <= k:
        return _plot_arpes_series(d, name)
    if {"Q_red", "E_n"} <= k and ("LW" in k or "LW_eV" in k):
        return _plot_sweep(d, name)
    if {"chi", "E", "Q_red"} <= k:
        return _plot_parity(d, name)
    return None


def _fig(n, title, caption=""):
    """A4 landscape with n panels, a title and a caption under the panels."""
    import matplotlib.pyplot as plt
    rows = 1 if n <= 2 else 2
    cols = 1 if n == 1 else 2
    fig, ax = plt.subplots(rows, cols, figsize=(11.69, 8.27), squeeze=False)
    lines = textwrap.wrap(caption, 172) if caption else []
    fig.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.1 + 0.019 * len(lines),
                        wspace=0.28, hspace=0.42)
    fig.suptitle(title, fontsize=13, color=ACCENT, weight="bold")
    fig._xphd_caption = caption
    if lines:
        fig.text(0.07, 0.045, "\n".join(lines), fontsize=8.5, color=MUTED, va="bottom",
                 linespacing=1.45)
    axs = ax.ravel()
    for a in axs[n:]:
        a.set_visible(False)
    return fig, axs


def _panel(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="left", fontsize=10.5, color=INK,
                 weight="bold", pad=8)
    ax.grid(alpha=0.25, lw=0.5)


def _cbar(fig, mappable, ax, label):
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import NullFormatter
    cb = fig.colorbar(mappable, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    norm = mappable.norm
    if isinstance(norm, LogNorm) and norm.vmin and norm.vmax:
        cb.ax.yaxis.set_minor_formatter(NullFormatter())
        _ticks125(cb.ax.yaxis, norm.vmin, norm.vmax)
    return cb


def _ticks125(axis, lo, hi):
    """On a log axis spanning under 2.5 decades: ticks at 1, 2, 5 x 10^k as plain numbers (20,
    50, 100), no labelled minor ticks."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    if lo > 0 and hi > lo and np.log10(hi / lo) < 2.5:
        axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
        axis.set_minor_formatter(NullFormatter())


def _plain_log(ax, axis="y"):
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    _ticks125(ax.yaxis if axis == "y" else ax.xaxis, lo, hi)


def _groups(E, tol=1e-3):
    """Runs of states within tol (eV) of the previous one: [[0, 1], [2], ...]."""
    out = []
    for i in np.argsort(E, kind="stable"):
        if out and abs(E[i] - E[out[-1][-1]]) <= tol:
            out[-1].append(int(i))
        else:
            out.append([int(i)])
    return out


def _ev(x):
    return f"{float(x):.4f} eV"


def _vec(v):
    return "(" + ", ".join(f"{float(x):.4g}" for x in np.asarray(v, float).ravel()[:3]) + ")"


_B60 = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
_B120 = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2]])
_GAMMA, _M, _K = np.zeros(2), np.array([0.5, 0.0]), np.array([0.5, 0.5 / np.sqrt(3)])
_PATH = (("Γ", _GAMMA), ("M", _M), ("K", _K), ("Γ", _GAMMA))


def _ops():
    """The 12 operations of the hexagon: rotations by 60 deg and mirrors."""
    out = []
    for i in range(6):
        c, s = np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)
        R = np.array([[c, -s], [s, c]])
        out += [R, R @ np.diag([1.0, -1.0])]
    return out


def _key(r):
    r = np.mod(np.round(r, 6), 1.0)
    r[np.isclose(r, 1.0)] = 0.0
    return tuple(np.round(r, 5))


def _basis(q, vals):
    """The reciprocal basis (rows, Cartesian) under which `vals` (one row per q) is invariant
    under a 60 deg rotation; the pw.x one if both are."""
    q = np.asarray(q, float)[:, :2]
    v = np.asarray(vals, float).reshape(len(q), -1)
    where = {_key(r): i for i, r in enumerate(q)}
    best, err = _B60, np.inf
    for B in (_B60, _B120):
        R = _ops()[2]
        rot = (q @ B) @ R.T @ np.linalg.inv(B)
        pairs = [(i, where[_key(r)]) for i, r in enumerate(rot) if _key(r) in where]
        if len(pairs) < max(3, len(q) // 4):
            continue
        i, j = np.array(pairs).T
        e = np.mean(np.abs(v[i] - v[j])) / max(np.mean(np.abs(v)), 1e-300)
        if e < err - 1e-9:
            best, err = B, e
    return best


def _images(q, B, sym=True):
    """(nq, 12*9, 2): every q under the hexagon's operations (only the identity with sym=False)
    and the nearest lattice translations, Cartesian."""
    c = np.asarray(q, float)[:, :2] @ B
    G = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], float) @ B
    ops = _ops() if sym else _ops()[:1]
    return np.concatenate([(c @ R.T)[:, None, :] + G[None] for R in ops], axis=1)


def _path(q, B, sym=True):
    """The points of the set that lie on Gamma-M-K-Gamma: indices, distance along the path
    (units of |b|), the positions of the corners."""
    pts = _images(q, B, sym)
    idx, x, ticks, x0 = [], [], [0.0], 0.0
    for s, ((_, A), (_, C)) in enumerate(zip(_PATH[:-1], _PATH[1:])):
        L = np.linalg.norm(C - A)
        u = (C - A) / L
        rel = pts - A
        t = rel @ u
        perp = np.abs(rel @ np.array([-u[1], u[0]]))
        iq, im = np.nonzero((perp < 1e-6) & (t > -1e-6) & (t < L + 1e-6))
        tt = t[iq, im]
        seen = set()
        for j in np.argsort(tt, kind="stable"):
            key = round(float(tt[j]) / L, 5)
            if key in seen or (s > 0 and key == 0.0):
                continue
            seen.add(key)
            idx.append(int(iq[j]))
            x.append(x0 + float(tt[j]))
        x0 += L
        ticks.append(x0)
    return np.array(idx, int), np.array(x), ticks


def _path_axis(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels([lab for lab, _ in _PATH])
    for t in ticks[1:-1]:
        ax.axvline(t, color=RULE, lw=0.8, zorder=0)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.grid(axis="x", visible=False)


def _no_path(ax):
    ax.text(0.5, 0.5, "no points of the set on Γ–M–K–Γ",
            ha="center", va="center", color=MUTED, transform=ax.transAxes)


def _zone(fig, ax, q, vals, B, label, log=True, cmap="magma", vmin=None, vmax=None):
    """vals (one per q) over the first Brillouin zone, with its edge and Gamma, M, K marked."""
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.tri import Triangulation
    pts = _images(q, B, sym=False)
    r = np.linalg.norm(pts, axis=2)
    keep = r <= r.min(axis=1, keepdims=True) + 1e-6
    iq, im = np.nonzero(keep)
    xy = pts[iq, im]
    v = np.asarray(vals, float)[iq]
    pos = v[v > 0]
    if log and pos.size and pos.max() < 10 * pos.min() and vmin is None:
        log = False
    if log:
        hi = pos.max() if pos.size else 1.0
        lo = max(pos.min() if pos.size else hi * 1e-6, hi * 1e-6)
        norm = LogNorm(vmin=vmin or lo, vmax=vmax or hi)
        v = np.clip(v, norm.vmin, None)
    else:
        norm = Normalize(vmin=np.nanmin(v) if vmin is None else vmin,
                         vmax=np.nanmax(v) if vmax is None else vmax)
    m = ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1]), v, shading="gouraud",
                     cmap=cmap, norm=norm, rasterized=True)
    ang = np.pi / 6 + np.arange(7) * np.pi / 3
    ax.plot(np.cos(ang) / np.sqrt(3), np.sin(ang) / np.sqrt(3), color=INK, lw=0.8)
    for lab, P, off in (("Γ", _GAMMA, (-0.07, -0.06)), ("M", _M, (0.012, -0.012)),
                        ("K", _K, (0.012, 0.0))):
        ax.plot(*P, "o", ms=3, color="white", mec=INK, mew=0.6)
        ax.text(P[0] + off[0], P[1] + off[1], lab, fontsize=10, color=INK, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.62, 0.62)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    _cbar(fig, m, ax, label)
    return m


def _deg_set(E, n=0, tol=1e-4):
    """State n and every state within tol (eV) of it."""
    E = np.asarray(E, float)
    return np.nonzero(np.abs(E - E[n]) <= tol)[0]


def _states(sel):
    sel = [int(i) + 1 for i in sel]
    return f"n = {sel[0]}" if len(sel) == 1 else "n = " + ", ".join(map(str, sel))


def _plot_archive(d, name):
    import matplotlib.pyplot as plt
    q = np.asarray(d["q_red"], float)
    hw = 1e3 * np.asarray(d["hw_grid"], float)
    Em = np.asarray(d["E_m_grid"], float)
    i0 = int(np.argmin(np.linalg.norm(np.mod(q[:, :2] + 0.5, 1) - 0.5, axis=1)))
    En = np.asarray(d["E_n_grid"], float) if "E_n_grid" in d else Em[i0]
    g2 = np.asarray(d["g2_grid"], float) if "g2_grid" in d else np.abs(d["G_grid"]) ** 2
    nq, nmod = hw.shape
    nexc = Em.shape[1]
    sel = _deg_set(En)
    out = g2[:, :, sel, :].sum(axis=(2, 3)) / len(sel)
    B = _basis(q, hw)
    idx, x, ticks = _path(q, B, sym=False)
    Q = d.get("Q_red", np.zeros(3))
    mesh = d.get("mesh", None)
    mesh_s = "x".join(str(int(m)) for m in np.asarray(mesh)[:2]) if mesh is not None else str(nq)
    colors = plt.get_cmap("tab10").colors

    fig, ax = _fig(4, f"{name}:  exciton-phonon coupling at Q = {_vec(Q)}",
                   f"{mesh_s} q-mesh ({nq} points), {nexc} exciton states, {nmod} phonon "
                   f"branches; initial state {_states(sel)}, Eₙ(Q) = {_ev(En[sel[0]])}. "
                   "(a), (c), (d) are against the phonon momentum q; (b) against the exciton "
                   "momentum k = Q + q. (a) Phonon energies from hw_grid at the mesh points on "
                   "the path (markers), branches numbered by energy at each q. (b) Exciton bands "
                   "E(k), read from E_m_grid as E(Q+q): the same bands in every archive of a set; "
                   "dot: the initial state, where Q or a point equivalent to it lies on the path; "
                   "dashed: Eₙ(Q); shaded: Eₙ(Q) ± ħωₘₐₓ, the energies one phonon can reach. "
                   "(c) Total coupling out of the initial state, Σ |Gₙₘ(q,ν)|² over ν and m, "
                   "over the first Brillouin zone. (d) The same sum per branch along the path, "
                   "colours as in (a).")

    _panel(ax[0], "a", "Phonon dispersion on the archive mesh")
    if len(idx):
        for nu in range(nmod):
            ax[0].plot(x, hw[idx, nu], "-o", ms=2.6, lw=1.1, color=colors[nu % 10],
                       label=f"ν = {nu + 1}")
        _path_axis(ax[0], ticks)
        ax[0].legend(fontsize=7.5, ncol=3, loc="best", framealpha=0.9)
    else:
        _no_path(ax[0])
    ax[0].set_ylabel(r"$\hbar\omega_\nu(\mathbf{q})$  (meV)")
    ax[0].set_ylim(bottom=min(0.0, float(np.nanmin(hw))) - 2)

    _panel(ax[1], "b", "Exciton bands, k = Q + q")
    wmax = float(np.nanmax(hw)) * 1e-3
    E0 = float(En[sel[0]])
    kq = np.mod(np.asarray(Q, float)[:2] + q[:, :2], 1.0)
    kidx, kx, kticks = _path(kq, B, sym=True)
    if len(kidx):
        for m in range(nexc):
            ax[1].plot(kx, Em[kidx, m], "-", lw=1.4 if m == 0 else 0.7,
                       color=ACCENT if m == 0 else MUTED, alpha=1 if m == 0 else 0.8)
        _, qx, _ = _path(np.asarray(Q, float)[None, :], B, sym=True)
        if len(qx):
            ax[1].plot(qx, np.full(len(qx), E0), "o", ms=7, color=BAD, mec="white", mew=1,
                       zorder=5, label="initial state at Q")
            ax[1].legend(fontsize=8, loc="upper right")
        else:
            ax[1].text(0.99, 0.97, "Q is off the path", transform=ax[1].transAxes,
                       ha="right", va="top", fontsize=8, color=MUTED)
        _path_axis(ax[1], kticks)
    else:
        _no_path(ax[1])
    ax[1].axhspan(E0 - wmax, E0 + wmax, color=ACCENT, alpha=0.10, lw=0)
    ax[1].axhline(E0, color=ACCENT, ls="--", lw=1.0)
    ax[1].text(0.99, E0 + wmax, rf"$E_n(\mathbf{{Q}})$ = {E0:.3f} eV", ha="right", va="bottom",
               transform=ax[1].get_yaxis_transform(), fontsize=8.5, color=ACCENT)
    ax[1].set_ylabel(r"$E_m(\mathbf{k})$  (eV)")

    _panel(ax[2], "c", f"Coupling out of {_states(sel)} over the zone")
    _zone(fig, ax[2], q, out.sum(axis=1), B,
          r"$\sum_{\nu,m}|G_{nm}(\mathbf{q},\nu)|^2$  (eV$^2$)")

    _panel(ax[3], "d", "Coupling per phonon branch")
    if len(idx):
        hi = float(out.max()) if out.max() > 0 else 1.0
        for nu in range(nmod):
            ax[3].semilogy(x, np.maximum(out[idx, nu], hi * 1e-8), "-o", ms=2.4, lw=1.0,
                           color=colors[nu % 10], label=f"ν = {nu + 1}")
        ax[3].set_ylim(hi * 1e-6, hi * 3)
        _path_axis(ax[3], ticks)
    else:
        _no_path(ax[3])
    ax[3].set_ylabel(r"$\sum_m|G_{nm}(\mathbf{q},\nu)|^2$  (eV$^2$)")
    return fig


def _plot_parity(d, name):
    E, chi = np.asarray(d["E"], float), np.asarray(d["chi"], float)
    q = np.asarray(d["Q_red"], float)
    B = _basis(q, E)
    idx, x, ticks = _path(q, B)
    fig, ax = _fig(1, f"{name}:  σh parity of the excitons",
                   f"{E.shape[0]} Q points, {E.shape[1]} states. Exciton energies Eₙ(Q) "
                   "along Γ–M–K–Γ coloured by the σh parity χ = <σh>: +1 even, −1 odd; "
                   "values in between mark states the mirror mixes.")
    _panel(ax[0], "a", "Exciton bands by parity")
    if len(idx):
        for n in range(E.shape[1]):
            ax[0].plot(x, E[idx, n], "-", color=RULE, lw=0.6, zorder=1)
        X = np.repeat(x[:, None], E.shape[1], axis=1)
        s = ax[0].scatter(X.ravel(), E[idx].ravel(), c=chi[idx].ravel(), s=9,
                          cmap="coolwarm", vmin=-1, vmax=1, zorder=2)
        _cbar(fig, s, ax[0], r"$\chi = \langle\sigma_h\rangle$")
        _path_axis(ax[0], ticks)
    else:
        _no_path(ax[0])
    ax[0].set_ylabel(r"$E_n(\mathbf{Q})$  (eV)")
    return fig


def _plot_linewidth(d, name):
    import matplotlib.pyplot as plt
    T = np.atleast_1d(np.asarray(d["T"], float))
    tot = 1e3 * np.atleast_2d(np.asarray(d["total"], float))
    E = np.asarray(d["E_n"], float)
    has_modes = "mode_em" in d and "mode_ab" in d
    cmap = plt.get_cmap("viridis")
    fig, ax = _fig(2, f"{name}:  exciton-phonon linewidths",
                   f"{len(E)} exciton states at {len(T)} temperature(s). (a) FWHM of every "
                   "state against its energy, one curve per temperature; numbers mark the state "
                   "index. (b) " + ("The FWHM of state 1 split by phonon branch (emission + "
                                    "absorption), against temperature." if has_modes else
                                    "Emission and absorption parts of the FWHM of state 1 "
                                    "against temperature."))
    _panel(ax[0], "a", "Width of every exciton")
    for i, t in enumerate(T):
        c = cmap(i / max(len(T) - 1, 1))
        ax[0].semilogy(E, np.maximum(tot[i], 1e-6), "o", ms=5, color=c, alpha=0.85,
                       label=f"{t:g} K")
    for grp in _groups(E, max(1e-3, 0.012 * float(np.ptp(E)))):
        top = max(float(tot[:, grp].max()), 1e-6)
        g = sorted(i + 1 for i in grp)
        lab = (str(g[0]) if len(g) == 1 else f"{g[0]}\u2013{g[-1]}"
               if g[-1] - g[0] == len(g) - 1 else ",".join(map(str, g)))
        ax[0].annotate(lab, (E[grp].mean(), top), xytext=(0, 6), textcoords="offset points",
                       fontsize=7.5, color=MUTED, ha="center")
    ax[0].set(xlabel=r"$E_n$  (eV)", ylabel="FWHM  (meV)")
    ax[0].margins(y=0.12)
    _plain_log(ax[0])
    ax[0].legend(fontsize=8)
    _panel(ax[1], "b", "State 1 by channel")
    if has_modes:
        me = 1e3 * np.asarray(d["mode_em"], float)[:, 0, :]
        ma = 1e3 * np.asarray(d["mode_ab"], float)[:, 0, :]
        tab = plt.get_cmap("tab10").colors
        for nu in range(me.shape[1]):
            ax[1].plot(T, me[:, nu] + ma[:, nu], "o-", ms=4, color=tab[nu % 10],
                       label=f"ν = {nu + 1}")
        ax[1].plot(T, tot[:, 0], "k--", lw=1, label="total")
    else:
        ax[1].plot(T, tot[:, 0], "k-o", ms=4, label="total")
        if "emission" in d:
            ax[1].plot(T, 1e3 * np.atleast_2d(d["emission"])[:, 0], "s--", label="emission")
        if "absorption" in d:
            ax[1].plot(T, 1e3 * np.atleast_2d(d["absorption"])[:, 0], "^:", label="absorption")
    ax[1].set(xlabel="T  (K)", ylabel="FWHM  (meV)")
    ax[1].legend(fontsize=8, ncol=2)
    return fig


def _plot_sweep(d, name):
    import matplotlib.pyplot as plt
    lw = 1e3 * np.asarray(d["LW_eV"] if "LW_eV" in d else d["LW"], float)
    T = np.atleast_1d(np.asarray(d["T_all"] if "T_all" in d else d["T"], float))
    E = np.asarray(d["E_n"], float)
    q = np.asarray(d["Q_red"], float)
    if lw.ndim != 3 or lw.shape[1:] != E.shape:
        return None
    B = _basis(q, E)
    idx, x, ticks = _path(q, B)
    cmap = plt.get_cmap("viridis")
    fig, ax = _fig(2, f"{name}:  linewidths over the zone",
                   f"{E.shape[0]} Q points, {E.shape[1]} states, T = "
                   f"{', '.join(f'{t:g}' for t in T)} K. (a) FWHM of the lowest exciton at each Q, "
                   f"{T[-1]:g} K, over the first Brillouin zone. (b) The same along "
                   "Γ–M–K–Γ for every temperature.")
    _panel(ax[0], "a", f"Lowest exciton, {T[-1]:g} K")
    _zone(fig, ax[0], q, lw[-1, :, 0], B, "FWHM  (meV)", cmap="magma")
    _panel(ax[1], "b", "Lowest exciton along the path")
    if len(idx):
        for i, t in enumerate(T):
            ax[1].semilogy(x, np.maximum(lw[i, idx, 0], 1e-6), "-o", ms=2.5,
                           color=cmap(i / max(len(T) - 1, 1)), label=f"{t:g} K")
        _path_axis(ax[1], ticks)
        _plain_log(ax[1])
        ax[1].legend(fontsize=8)
    else:
        _no_path(ax[1])
    ax[1].set_ylabel("FWHM  (meV)")
    return fig


def _plot_lineshape(d, name):
    import matplotlib.pyplot as plt
    T = np.atleast_1d(np.asarray(d["T"], float))
    k = set(d)
    fig, ax = _fig(2, f"{name}:  absorption line of the A exciton",
                   (f"E_A = {_ev(d['E_A'])}, {len(T)} temperature(s). " if "E_A" in k else "")
                   + "(a) Width of the A line: from the full self-energy matrix, its diagonal "
                   "only, and the FWHM read off \u03b5\u2082. (b) \u03b5\u2082 with the full "
                   "self-energy at every temperature, each normalised to its peak.")
    _panel(ax[0], "a", "Width against temperature")
    ax[0].plot(T, 1e3 * np.asarray(d["gamma_A_full"]), "o-", label="Γ_A, full Σ matrix")
    ax[0].plot(T, 1e3 * np.asarray(d["gamma_A"]), "s--", label="Γ_A, diagonal Σ")
    if "fwhm_full" in k:
        ax[0].plot(T, 1e3 * np.asarray(d["fwhm_full"]), "^:", label="FWHM of ε₂ (full)")
    ax[0].set(xlabel="T  (K)", ylabel="width  (meV)")
    ax[0].legend(fontsize=8)
    _panel(ax[1], "b", r"$\varepsilon_2(\omega)$ near $E_A$")
    x = 1e3 * (np.asarray(d["x_plot"], float) - float(d["E_A"]))
    cmap = plt.get_cmap("viridis")
    eps = np.atleast_2d(np.asarray(d["eps2_full"], float))
    for i, t in enumerate(T):
        e = eps[i]
        ax[1].plot(x, e / max(e.max(), 1e-300), color=cmap(i / max(len(T) - 1, 1)),
                   label=f"{t:g} K")
    widest = np.nanmax(1e3 * np.asarray(d["fwhm_full" if "fwhm_full" in k else "gamma_A_full"]))
    half = max(10 * widest, 5.0) if np.isfinite(widest) else np.inf
    ax[1].set(xlabel=r"$\hbar\omega - E_A$  (meV)", ylabel=r"$\varepsilon_2$  (normalised)",
              xlim=(max(x.min(), -half), min(x.max(), half)))
    ax[1].legend(fontsize=8)
    return fig


def _plot_bte(d, name):
    import matplotlib.pyplot as plt
    t, F, E = np.asarray(d["t"], float), np.asarray(d["F"], float), np.asarray(d["E"], float)
    N = F.sum(axis=1)
    N0 = N[0] if N[0] > 0 else 1.0
    Em = (F * E[None]).sum(axis=1) / np.where(N > 0, N, 1)
    inj = np.nonzero(F[0] > 1e-12 * F[0].max())[0] if F[0].max() > 0 else np.array([], int)
    Tk = float(d["T"]) if "T" in d else np.nan
    fig, ax = _fig(3, f"{name}:  real-time Boltzmann transport"
                   + (f", {Tk:g} K" if np.isfinite(Tk) else ""),
                   f"{F.shape[1]} states, {len(t)} snapshots up to {t[-1]:g} fs; "
                   f"{len(inj)} state(s) populated at t = 0. (a) Mean exciton energy above the "
                   "lowest state. (b) Population against energy at four times. (c) Where the "
                   "population is at the last time: summed over the states of each Q, over the "
                   "first Brillouin zone.")
    _panel(ax[0], "a", "Mean energy")
    ax[0].plot(t, 1e3 * (Em - E.min()), color=ACCENT)
    ax[0].set(xlabel="t  (fs)", ylabel=r"$\langle E\rangle - E_{\min}$  (meV)", xscale="symlog")
    _panel(ax[1], "b", "Energy distribution")
    cmap = plt.get_cmap("viridis")
    times = np.unique(np.linspace(0, len(t) - 1, 4).astype(int))
    e = 1e3 * (E - E.min())
    o = np.argsort(e)
    top = 0.0
    for i in times:
        c = np.cumsum(np.clip(F[i, o], 0, None))
        if c[-1] > 0:
            top = max(top, float(e[o][np.searchsorted(c, 0.999 * c[-1])]))
    bins = np.linspace(0.0, 1.1 * top if top > 0 else e.max(), 70)
    for j, i in enumerate(times):
        h, edges = np.histogram(e, bins=bins, weights=F[i] / N0)
        ax[1].plot(0.5 * (edges[1:] + edges[:-1]), h, color=cmap(j / max(len(times) - 1, 1)),
                   label=f"t = {t[i]:.4g} fs")
    ax[1].set(xlabel=r"$E - E_{\min}$  (meV)", ylabel="population / N(0)")
    ax[1].legend(fontsize=8)
    _panel(ax[2], "c", f"Population over the zone, t = {t[-1]:g} fs")
    if "Q_red" in d:
        Q = np.asarray(d["Q_red"], float)
        keys = [_key(r) for r in Q[:, :2]]
        uniq = {}
        for i, kk in enumerate(keys):
            uniq.setdefault(kk, []).append(i)
        qs = np.array([Q[v[0]] for v in uniq.values()])
        pop = np.array([F[-1, v].sum() for v in uniq.values()]) / N0
        _zone(fig, ax[2], qs, pop, _basis(qs, pop), "population / N(0)", cmap="viridis")
    else:
        _no_path(ax[2])
    return fig


def _plot_arpes(d, name):
    """The map along the path and, when the file has them, the occupied excitons that make it,
    against their own momentum Q."""
    from matplotlib.colors import LogNorm
    k = set(d)
    vbm = float(d["vbm"]) if "vbm" in k else None
    om = np.asarray(d["omega"], float) - (vbm or 0.0)
    elab = r"$E - E_{\mathrm{VBM}}$  (eV)" if vbm is not None else "E  (eV)"
    fine = "I_fine" in k
    I = np.asarray(d["I_fine"] if fine else d["I_path"], float)
    x = np.asarray(d["x_fine"], float) if fine and "x_fine" in k else np.arange(I.shape[0])
    t = float(d["time"]) if "time" in k else np.nan
    Tk = float(d["T"]) if "T" in k else np.nan
    when = (f"thermal, {Tk:g} K" if np.isfinite(Tk) else
            f"t = {t:g} fs" if np.isfinite(t) else "")
    occ = fine and {"E_path", "N_path"} <= k and np.shape(d["N_path"])[0] == len(x)
    sig = float(d["sigma"]) if "sigma" in k else 0.0
    ylim = _signal_window(om, I.max(axis=0), sig)

    fig, ax = _fig(2 if occ else 1, f"{name}:  exciton photoemission"
                   + (f", {when}" if when else ""),
                   "Photoemission takes out the electron: an exciton of momentum Q and energy E "
                   "gives signal at the electron's momentum k and at E + \u03b5\u1d65(k \u2212 Q), a "
                   "replica of the valence band around the hole. "
                   "(a) Intensity along the path, energies "
                   + ("from the valence maximum" if vbm is not None else "as in the file")
                   + "; Gaussian resolution \u03c3"
                   + (f" = {sig * 1e3:g} meV" if "sigma" in k else "") + ". "
                   + ("(b) The occupied excitons along the path against their own momentum Q: "
                      "bands E(Q), markers coloured by occupation." if occ else ""))

    _panel(ax[0], "a", "Intensity along the path (electron momentum k)")
    m = ax[0].pcolormesh(x, om, I.T, shading="auto", cmap="viridis", rasterized=True)
    _cbar(fig, m, ax[0], "intensity  (arb. u.)")
    _path_ticks(ax[0], d, k, fine)
    ax[0].grid(False)
    if vbm is not None and ylim[0] < 0 < ylim[1]:
        ax[0].axhline(0.0, color="white", lw=0.6, ls="--", alpha=0.8)
    ax[0].set(ylabel=elab, ylim=ylim)

    if occ:
        _panel(ax[1], "b", "Occupied excitons (exciton momentum Q)")
        E, N = np.asarray(d["E_path"], float), np.asarray(d["N_path"], float)
        for n in range(E.shape[1]):
            ax[1].plot(x, E[:, n], "-", color=RULE, lw=0.6, zorder=1)
        w = N / max(N.max(), 1e-300)
        keep = w > 1e-4
        s = ax[1].scatter(np.repeat(x[:, None], E.shape[1], 1)[keep], E[keep], c=w[keep], s=6,
                          cmap="magma_r", norm=LogNorm(vmin=1e-4, vmax=1), zorder=2)
        _cbar(fig, s, ax[1], "occupation / max")
        _path_ticks(ax[1], d, k, fine, color=RULE)
        if len(x):
            ax[1].set_xlim(x.min(), x.max())
        lo = float(E[keep].min()) if keep.any() else float(E.min())
        ax[1].set(ylabel=r"$E(\mathbf{Q})$  (eV)", ylim=(lo - 0.1, lo + 1.0))
    return fig


def _signal_window(om, prof, sig=0.0):
    """(lo, hi) of the energies that hold the signal, padded."""
    on = np.nonzero(prof > 1e-3 * prof.max())[0] if prof.max() > 0 else np.arange(om.size)
    pad = max(0.15, 5 * sig, 0.2 * float(om[on[-1]] - om[on[0]]))
    return max(om.min(), om[on[0]] - pad), min(om.max(), om[on[-1]] + pad)


def _path_ticks(ax, d, k, fine, color="white"):
    ticks = (np.asarray(d["xt_fine"], float) if fine and "xt_fine" in k else
             np.asarray(d["ticks"], float) if "ticks" in k else None)
    labels = [str(s) for s in np.atleast_1d(d["path"])] if "path" in k else None
    labels = [("\u0393" if s.upper() in ("G", "GAMMA") else s) for s in labels] if labels else None
    if ticks is not None and len(ticks):
        ax.set_xticks(ticks)
        if labels and len(labels) == len(ticks):
            ax.set_xticklabels(labels)
        for xt in ticks[1:-1]:
            ax.axvline(xt, color=color, lw=0.6, alpha=0.7)


def _plot_arpes_delays(maps):
    """maps: [(delay, path)], sorted."""
    pick = [maps[i] for i in np.unique(np.linspace(0, len(maps) - 1, 4).round().astype(int))]
    data = []
    for t, p in pick:
        with np.load(p, allow_pickle=False) as z:
            k = set(z.files)
            fine = "I_fine" in k
            d = {key: z[key] for key in ("omega", "vbm", "x_fine", "xt_fine", "ticks", "path",
                                         "sigma", "T") if key in k}
            d["I"] = np.asarray(z["I_fine"] if fine else z["I_path"], float)
            data.append((t, p, d, k, fine))
    vmax = max(float(d["I"].max()) for _, _, d, _, _ in data) or 1.0
    los, his = [], []
    for _, _, d, _, _ in data:
        om = np.asarray(d["omega"], float) - float(d.get("vbm", 0.0))
        lo, hi = _signal_window(om, d["I"].max(axis=0), float(d.get("sigma", 0.0)))
        los.append(lo)
        his.append(hi)
    ylim = (min(los), max(his))
    vbm = "vbm" in data[0][3]
    elab = r"$E - E_{\mathrm{VBM}}$  (eV)" if vbm else "E  (eV)"
    folder = os.path.basename(os.path.dirname(os.path.abspath(pick[0][1]))) or "."
    fig, ax = _fig(len(data), f"{folder}:  photoemission at {len(data)} of {len(maps)} delays",
                   f"{len(maps)} maps written, delays {maps[0][0]:g} to {maps[-1][0]:g} fs; shown: "
                   + ", ".join(f"{t:g} fs" for t, *_ in data) + ". One colour scale for all "
                   "panels: the zone-integrated intensity counts the excitons and is conserved, "
                   "so every change is a redistribution. Signal at the electron's momentum k, "
                   "at E + \u03b5\u1d65(k \u2212 Q).")
    for a, (t, p, d, k, fine), letter in zip(ax, data, "abcd"):
        om = np.asarray(d["omega"], float) - float(d.get("vbm", 0.0))
        I = d["I"]
        x = np.asarray(d["x_fine"], float) if fine and "x_fine" in k else np.arange(I.shape[0])
        _panel(a, letter, f"t = {t:g} fs")
        m = a.pcolormesh(x, om, I.T, shading="auto", cmap="viridis", vmin=0, vmax=vmax,
                         rasterized=True)
        _path_ticks(a, d, k, fine)
        a.grid(False)
        a.set(ylabel=elab, ylim=ylim)
    _cbar(fig, m, list(ax[:len(data)]), "intensity  (arb. u., common scale)")
    return fig


def _plot_arpes_series(d, name):
    """Photoemission over the whole zone against the delay, and the mean exciton energy
    relaxing."""
    k = set(d)
    t = np.asarray(d["times"], float)
    vbm = float(d["vbm"]) if "vbm" in k else 0.0
    om = np.asarray(d["omega"], float) - vbm
    elab = r"$E - E_{\mathrm{VBM}}$  (eV)" if "vbm" in k else "E  (eV)"
    sig = float(d["sigma"]) if "sigma" in k else 0.0
    ez = np.atleast_2d(np.asarray(d["edc_zone_t"], float))
    Em = np.asarray(d["E_mean_t"], float) if "E_mean_t" in k else None
    E0 = float(d["E_exc_min"]) if "E_exc_min" in k else (np.nanmin(Em) if Em is not None else 0)

    def cell_edges(v):
        v = np.asarray(v, float)
        if len(v) < 2:
            return np.array([v[0] - 0.5, v[0] + 0.5])
        return np.concatenate([[v[0] - 0.5 * (v[1] - v[0])], 0.5 * (v[1:] + v[:-1]),
                               [v[-1] + 0.5 * (v[-1] - v[-2])]])

    edges = cell_edges(t)
    pos = t[t > 0]
    lin = float(pos.min()) if pos.size else 1.0

    def delay_axis(a):
        a.set_xscale("symlog", linthresh=lin)
        a.set_xlim(edges[0], edges[-1])
        a.set_xlabel("delay  (fs)")

    fig, ax = _fig(2, f"{name}:  photoemission against the delay",
                   f"{len(t)} delays, {t[0]:g} to {t[-1]:g} fs. (a) Energy distribution of the "
                   "signal over the whole zone against the delay. (b) Mean exciton energy above "
                   "the lowest exciton. Symmetric-log delay axis.")
    _panel(ax[0], "a", "Whole zone")
    lo, hi = _signal_window(om, ez.max(axis=0), sig)
    m = ax[0].pcolormesh(edges, cell_edges(om), ez.T, shading="flat", cmap="viridis",
                         rasterized=True)
    _cbar(fig, m, ax[0], "intensity  (arb. u.)")
    ax[0].grid(False)
    ax[0].set(ylabel=elab, ylim=(lo, hi))
    delay_axis(ax[0])
    _panel(ax[1], "b", "Mean exciton energy")
    if Em is not None:
        ax[1].plot(t, 1e3 * (Em - E0), "o-", ms=4, color=ACCENT)
        ax[1].set_ylabel(r"$\langle E_X\rangle - E_{\min}$  (meV)")
        delay_axis(ax[1])
    else:
        ax[1].set_visible(False)
    return fig
