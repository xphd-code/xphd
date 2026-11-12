"""The start-up banner: shown for commands, never for --help / --version."""
import pytest

from xphd.banner import ART, banner_text


def test_every_face_renders_with_version_and_command():
    for face in ART:
        t = banner_text("1.0.0", ["linewidth", "a.npz"], face=face)
        assert "Version 1.0.0" in t and "$ xphd linewidth a.npz" in t
        assert "Developers:" in t
        for name, n in (("Rishabh Saraswat", 1), ("Mantasha Ansari", 2), ("Rekha Verma", 1),
                        ("Sitangshu Bhattacharya", 2)):
            assert f"{name} [{n}]" in t
        assert "[1] Nanoscale Electro-thermal Laboratory" in t
        assert "[2] Electronic Structure Theory Group" in t
        assert "The developers thank" in t and "LetzElPhC" not in t
        assert len(t.splitlines()) >= 8


def test_banner_printed_for_a_command(tmp_path, capsys, monkeypatch):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_archive_symmetry import _archive
    from xphd.cli import main
    monkeypatch.delenv("XPHD_NO_BANNER", raising=False)
    _archive(tmp_path / "a.npz")
    main(["check-archive", str(tmp_path / "a.npz")])
    out = capsys.readouterr().out
    assert "Exciton-Phonon Dynamics" in out and "READY" in out
    assert out.index("Exciton-Phonon Dynamics") < out.index("READY")


def test_no_banner_for_help_or_version_or_when_switched_off(tmp_path, capsys, monkeypatch):
    from xphd.cli import main
    for args in (["--help"], ["--version"], ["linewidth", "--help"]):
        with pytest.raises(SystemExit):
            main(args)
        assert "Exciton-Phonon Dynamics" not in capsys.readouterr().out
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_archive_symmetry import _archive
    _archive(tmp_path / "a.npz")
    monkeypatch.setenv("XPHD_NO_BANNER", "1")
    main(["check-archive", str(tmp_path / "a.npz")])
    assert "Exciton-Phonon Dynamics" not in capsys.readouterr().out
