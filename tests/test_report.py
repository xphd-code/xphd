"""The PDF report written after every command."""
import os
import re
import sys

import numpy as np
import pytest

from xphd import banner as xbanner
from xphd import cli, report

pytest.importorskip("matplotlib")


def _pages(path):
    raw = open(path, "rb").read()
    return raw.count(b"/Type /Page\n") + raw.count(b"/Type /Page ") + raw.count(b"/Type /Page>")


def test_report_after_a_run(archive, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XPHD_REPORT", "pdf")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "lw.npz"
    cli.main(["linewidth", str(archive), "--T", "77", "300", "-o", str(out)])
    rep = tmp_path / "lw_report.pdf"
    assert rep.exists() and open(rep, "rb").read(5) == b"%PDF-"
    raw = open(rep, "rb").read()
    assert b"xphd linewidth report" in raw and _pages(rep) >= 2
    assert b"Type3" not in raw
    assert f"report: {rep}" in capsys.readouterr().out


def test_no_report_flag_and_env(archive, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XPHD_REPORT", "1")
    cli.main(["--no-report", "linewidth", str(archive), "-o", str(tmp_path / "a.npz")])
    monkeypatch.setenv("XPHD_REPORT", "0")
    cli.main(["linewidth", str(archive), "-o", str(tmp_path / "b.npz")])
    assert not list(tmp_path.glob("*_report.*"))


def test_usage_error_writes_nothing_but_a_failure_does(archive, tmp_path, monkeypatch):
    monkeypatch.setenv("XPHD_REPORT", "pdf")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["linewidth", str(archive), "--no-such-option"])
    assert not list(tmp_path.glob("*_report.pdf"))
    with pytest.raises(FileNotFoundError):
        cli.main(["linewidth", str(tmp_path / "missing.npz")])
    reps = list(tmp_path.glob("xphd_linewidth_*_report.pdf"))
    assert len(reps) == 1


def test_run_record(archive, tmp_path, monkeypatch):
    """Inputs are described, untouched inputs are not outputs, the parsed options (with
    defaults) are kept, a failure is recorded."""
    monkeypatch.chdir(tmp_path)
    old = tmp_path / "old.npz"
    np.savez(old, x=np.arange(3.0))
    run = report.Run(["linewidth", str(archive), "--T", "77", "-o", "new.npz"])
    with run.capture():
        cli.main(["--no-report", "linewidth", str(archive), "--T", "77", "-o", "new.npz"])
    run.finish()
    outs = [os.path.basename(p) for p in run.outputs]
    assert outs == ["new.npz"]
    assert any(str(archive) in r[0] and r[3].startswith("sha256") for r in run.inputs)
    assert run.namespace.refine == 1 and run.namespace.T == [77.0]
    text = "\n".join(report._flatten(run.sections("test")))
    assert re.search(r"status +completed", text) and re.search(r"refine +1\n", text)
    bad = report.Run(["linewidth", "x.npz"])
    bad.finish(ValueError("boom"))
    text = "\n".join(report._flatten(bad.sections("test")))
    assert re.search(r"status +FAILED", text) and "ValueError: boom" in text


def test_only_known_results_are_plotted(tmp_path):
    """Rotation matrices and other arrays are not drawn; a result file is; a plot that fails is
    noted, not fatal."""
    import matplotlib
    matplotlib.use("Agg")
    files = {}
    for name, d in (("a.npz", dict(omega=np.linspace(0, 1, 50), im=np.ones(50))),
                    ("b.npz", dict(M=np.ones((30, 40)), c=np.ones((3, 4, 5)) + 1j)),
                    ("c.npz", dict(s=np.array("text"))),
                    ("parity.npz", dict(chi=np.ones((4, 3)), E=np.ones((4, 3)),
                                        Q_red=np.zeros((4, 3)))),
                    ("broken.npz", dict(chi=np.ones(2), E=np.ones(5), Q_red=np.ones(7))),
                    ("Dmats.npy", None)):
        p = tmp_path / name
        if d is None:
            np.save(p, np.ones((12, 4, 3, 3)))
        else:
            np.savez(p, **d)
        files[name] = str(p)
    for name in ("a.npz", "b.npz", "c.npz", "Dmats.npy"):
        notes = []
        assert report.plot_outputs("any", [files[name]], notes) == [] and notes == []
    notes = []
    assert len(report.plot_outputs("parity", [files["parity.npz"]], notes)) == 1 and not notes
    notes = []
    assert report.plot_outputs("parity", [files["broken.npz"]], notes) == []
    assert len(notes) == 1 and notes[0].startswith("broken.npz: not drawn")


def test_zone_path_and_reciprocal_basis():
    """The path picks the mesh points on Gamma-M-K-Gamma, and the basis is read off the data."""
    n = 24
    q = np.array([(i / n, j / n, 0.0) for i in range(n) for j in range(n)])
    for B in (report._B60, report._B120):
        imgs = report._images(q, B)[:, :9]
        f = np.linalg.norm(imgs, axis=2).min(axis=1)
        assert report._basis(q, f) is B
        idx, x, ticks = report._path(q, B)
        assert len(idx) == n // 2 + 1 + n // 6 + n // 3 and len(ticks) == 4
        assert np.all(np.diff(x) > 0)
        k = idx[n // 2 + n // 6]
        assert np.isclose(f[k], 1 / np.sqrt(3))
        assert np.isclose(f[idx[n // 2]], 0.5)


def test_archive_plot_is_labelled(archive):
    """The archive figure: four panels, each titled, the paths labelled."""
    import matplotlib
    matplotlib.use("Agg")
    figs = report.plot_outputs("generate", [str(archive)])
    assert len(figs) == 1
    axes = [a for a in figs[0].axes if a.get_visible() and a.get_title(loc="left")]
    assert [a.get_title(loc="left")[:3] for a in axes] == ["(a)", "(b)", "(c)", "(d)"]
    labels = [t.get_text() for t in axes[0].get_xticklabels()]
    assert labels == ["\u0393", "M", "K", "\u0393"]


def test_report_carries_the_printed_banner(archive, tmp_path, monkeypatch, capsys):
    """The report shows the banner the terminal showed (same lettering), less its command line;
    with the banner switched off the report still has one."""
    monkeypatch.setenv("XPHD_REPORT", "pdf")
    monkeypatch.delenv("XPHD_NO_BANNER", raising=False)
    monkeypatch.chdir(tmp_path)
    seen = []
    write = report.Run.write

    def spy(self, version):
        seen.append(report._banner_lines(self.banner, version))
        return write(self, version)

    monkeypatch.setattr(report.Run, "write", spy)
    cli.main(["check", str(archive)])
    out = capsys.readouterr().out
    lines = seen[-1]
    assert lines and all(ln in out for ln in lines)
    text = "\n".join(lines)
    for part in ("Exciton-Phonon Dynamics", "Developers:", "Sitangshu Bhattacharya [2]",
                 "please cite", "doi.org/" if xbanner.DOI else "github.com/", "Acknowledgements", "python "):
        assert part in text
    assert "$ xphd" not in text
    monkeypatch.setenv("XPHD_NO_BANNER", "1")
    cli.main(["check", str(archive)])
    assert "Exciton-Phonon Dynamics" not in capsys.readouterr().out
    assert "Exciton-Phonon Dynamics" in "\n".join(seen[-1])


def test_progress_bars_keep_only_their_last_state(tmp_path, monkeypatch):
    """A bar redrawn with carriage returns is one line in the log, as on the terminal; colour
    codes are dropped."""
    import sys
    monkeypatch.chdir(tmp_path)
    run = report.Run(["check", "x"])
    with run.capture():
        print("start")
        for i in range(0, 101, 5):
            sys.stderr.write(f"\rLoading: {i:3d}%|{'#' * (i // 5):<20}| {i}/100")
        sys.stderr.write("\n")
        sys.stderr.write("\r" + " " * 50 + "\r")
        print("\x1b[1mdone\x1b[0m")
    assert report._log_lines(run.log.getvalue()) == [
        "start", "Loading: 100%|####################| 100/100", "done"]


def test_batch_report_is_named_after_the_command(tmp_path, monkeypatch):
    """Many data files (generate all): the report goes in their folder, named after the command,
    and the plots take the first files by name."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "archives"
    out.mkdir()
    run = report.Run(["generate", "all", str(out)])
    for i in range(6):
        np.savez(out / f"GI_ExcPh_Q{i + 1:04d}.npz", x=np.ones(10 * (6 - i)))
    run.finish()
    rep = run.report_path()
    assert os.path.dirname(rep) == str(out)
    assert os.path.basename(rep).startswith("xphd_generate_")
    one = report.Run(["linewidth", "a.npz", "-o", "lw.npz"])
    np.savez(tmp_path / "lw.npz", x=np.ones(3))
    one.finish()
    assert one.report_path() == str(tmp_path / "lw_report.pdf")


def test_time_resolved_arpes_gives_a_series_and_a_delay_figure(tmp_path):
    """arpes --times --series: one figure of the series, one of the maps at four delays on a
    common scale; thermal maps stay single figures."""
    import matplotlib
    matplotlib.use("Agg")
    om = np.linspace(-1.0, 1.0, 60)
    x = np.linspace(0, 1, 20)
    files = []
    for t in (0, 10, 100, 1000, 3000):
        I = np.exp(-((om[None] - 0.3 * np.cos(3 * x[:, None]) - 0.001 * t) / 0.05) ** 2)
        p = tmp_path / f"arpes_{t}fs.npz"
        np.savez(p, I_fine=I, x_fine=x, xt_fine=np.array([0, 0.3, 0.6, 1.0]), omega=om,
                 path=np.array(["M", "G", "K", "M"]), time=np.array(float(t)),
                 T=np.array(np.nan), vbm=np.array(-0.5), sigma=np.array(0.01))
        files.append(str(p))
    ser = tmp_path / "series.npz"
    nt = 5
    np.savez(ser, times=np.array([0, 10, 100, 1000, 3000.]), omega=om,
             edc_zone_t=np.ones((nt, om.size)), edc_box_t=np.ones((nt, 2, om.size)),
             omega_x=np.linspace(3, 4, 30), dos_occ_t=np.ones((nt, 30)),
             E_mean_t=np.linspace(3.8, 3.4, nt), box_label=np.array(["G", "K"]),
             vbm=np.array(-0.5), E_exc_min=np.array(3.4), sigma=np.array(0.01))
    notes = []
    figs = report.plot_outputs("arpes", files + [str(ser)], notes)
    assert not notes and len(figs) == 2
    titles = [f._suptitle.get_text() for f in figs]
    assert "against the delay" in titles[0] and "4 of 5 delays" in titles[1]
    thermal = tmp_path / "thermal.npz"
    np.savez(thermal, I_fine=np.ones((20, 60)), x_fine=x, omega=om, time=np.array(0.0),
             T=np.array(77.0), vbm=np.array(-0.5))
    figs = report.plot_outputs("arpes", [str(thermal)])
    assert len(figs) == 1 and "thermal, 77 K" in figs[0]._suptitle.get_text()


def test_title_and_pipeline_box():
    """The title bar names the step (BTE, ARPES, Linewidths); the pipeline beside the banner
    marks the box that holds the command."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert report.command_title("bte") == "BTE" and report.command_title("arpes") == "ARPES"
    assert report.command_title("linewidth") == "Linewidths"
    assert report.command_title("some-new-cmd") == "Some new cmd"
    keys = {b[0] for b in report.PIPE}
    assert set(report.PIPE_BOX.values()) <= keys
    for cmd, box in (("generate", "gen"), ("sweep", "lw"), ("irreps", "parity"),
                     ("bte", "bte"), ("arpes", "arpes"), ("matdyn", "matdyn")):
        assert report.PIPE_BOX[cmd] == box
    fig, ax = plt.subplots()
    report._flowchart(ax, "bte")
    here = [p for p in ax.patches if p.get_edgecolor()[:3] == matplotlib.colors.to_rgb(report.HERE[1])]
    assert len(here) == 2
    plt.close(fig)


def test_short_run_is_short(archive, tmp_path, monkeypatch):
    """A command that writes nothing and prints little: the masthead and the run record on page
    1, the log at most one more page."""
    monkeypatch.setenv("XPHD_REPORT", "pdf")
    monkeypatch.chdir(tmp_path)
    cli.main(["check", str(archive)])
    reps = list(tmp_path.glob("xphd_check_*_report.pdf"))
    assert len(reps) == 1 and _pages(reps[0]) <= 2


def test_which_reports():
    """XPHD_REPORT chooses; otherwise the system: text on Linux, PDF elsewhere."""
    f = report.formats
    assert f({}, "win32") == ("pdf",) and f({"XPHD_REPORT": "1"}, "darwin") == ("pdf",)
    assert f({}, "linux") == ("txt",) and f({"XPHD_REPORT": "1", "DISPLAY": ":0"}, "linux") == ("txt",)
    assert f({"XPHD_REPORT": "pdf"}, "linux") == ("pdf",)
    assert f({"XPHD_REPORT": "txt"}, "win32") == ("txt",)
    assert f({"XPHD_REPORT": "both"}, "linux") == ("pdf", "txt")


def test_mathtext_as_plain_text():
    p = report._plain
    assert p(r"$\hbar\omega_\nu(\mathbf{q})$  (meV)") == "ħω_ν(q)  (meV)"
    assert p(r"$\sum_m|G_{nm}(\mathbf{q},\nu)|^2$  (eV$^2$)") == \
        "Σ_m|G_nm(q,ν)|²  (eV²)"
    assert p(r"$E - E_{\mathrm{VBM}}$  (eV)") == "E - E_VBM  (eV)"
    assert p(r"$\langle E\rangle - E_{\min}$") == "⟨E⟩ - E_min"
    assert p(r"cm$^{-1}$ along $\Gamma$-M") == "cm⁻¹ along Γ-M"
    assert p("no maths here") == "no maths here"


def test_text_report_on_linux(archive, tmp_path, monkeypatch, capsys):
    """On Linux the text report alone, saying how to get the plots."""
    monkeypatch.setenv("XPHD_REPORT", "1")
    monkeypatch.setattr(report.sys, "platform", "linux")
    monkeypatch.chdir(tmp_path)
    cli.main(["linewidth", str(archive), "-o", str(tmp_path / "lw.npz")])
    assert not (tmp_path / "lw_report.pdf").exists()
    txt = tmp_path / "lw_report.txt"
    text = txt.read_text(encoding="utf-8")
    assert "XPHD_REPORT=both" in text and "-- Log " in text
    assert f"report: {txt}" in capsys.readouterr().out
    monkeypatch.setattr(report.sys, "platform", "win32")
    cli.main(["linewidth", str(archive), "-o", str(tmp_path / "w.npz")])
    assert (tmp_path / "w_report.pdf").exists() and not (tmp_path / "w_report.txt").exists()


def test_both_reports(archive, tmp_path, monkeypatch, capsys):
    """XPHD_REPORT=both: the PDF and the text report, which has every section, each figure by
    title, panels and caption with no mathtext left, the log."""
    monkeypatch.setenv("XPHD_REPORT", "both")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "lw.npz"
    cli.main(["linewidth", str(archive), "--T", "77", "300", "-o", str(out)])
    pdf, txt = tmp_path / "lw_report.pdf", tmp_path / "lw_report.txt"
    assert pdf.exists() and txt.exists()
    printed = capsys.readouterr().out
    assert f"report: {pdf}\nreport: {txt}" in printed
    text = txt.read_text(encoding="utf-8")
    for part in ("Linewidths", "COMPLETED", "Version ", "IIIT Allahabad", "Developers:",
                 "-- Run ", "-- Command ", "-- Inputs ", "-- Parameters, with defaults ",
                 "-- Outputs ", "-- Plots ", "lw_report.pdf", "Figure 1.", "(a)",
                 "-- Log ", "sha256"):
        assert part in text, part
    plots = text.split("-- Plots ")[1].split("-- Log ")[0]
    assert "$" not in plots and "\\" not in plots
    assert max(len(ln) for ln in plots.splitlines()) <= report.TEXT_WIDTH
    assert f"xphd linewidth {archive}" in text


def test_text_report_alone_without_matplotlib(archive, tmp_path, monkeypatch, capsys):
    """No matplotlib (a bare cluster install): the text report alone, saying why."""
    monkeypatch.setenv("XPHD_REPORT", "pdf")
    monkeypatch.chdir(tmp_path)
    for name in [m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib.")]:
        monkeypatch.setitem(sys.modules, name, None)
    cli.main(["linewidth", str(archive), "-o", str(tmp_path / "lw.npz")])
    assert not (tmp_path / "lw_report.pdf").exists()
    text = (tmp_path / "lw_report.txt").read_text(encoding="utf-8")
    assert "matplotlib is not installed" in text and "-- Log " in text
    assert "report: " + str(tmp_path / "lw_report.txt") in capsys.readouterr().out


def test_a_failed_pdf_leaves_the_text_report(archive, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XPHD_REPORT", "pdf")
    monkeypatch.chdir(tmp_path)

    def broken(self, *a, **k):
        raise RuntimeError("no fonts")

    monkeypatch.setattr(report.Run, "_pdf", broken)
    cli.main(["linewidth", str(archive), "-o", str(tmp_path / "lw.npz")])
    text = (tmp_path / "lw_report.txt").read_text(encoding="utf-8")
    assert "the PDF could not be written (RuntimeError: no fonts)" in text
    assert "no PDF report: RuntimeError: no fonts" in capsys.readouterr().out
