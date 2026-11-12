import numpy as np
import pytest

A_LAT = 3.2


def hex_vecs(a=A_LAT):
    return np.array([a, 0.0]), np.array([-a / 2, a * np.sqrt(3) / 2])


@pytest.fixture
def archive(tmp_path):
    """Synthetic archive with the real key layout and a SHUFFLED q order."""
    n, nm, ne = 12, 6, 4
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    c = (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky)
         + np.cos(2 * np.pi * (kx + ky))) / 3
    E_n = np.array([6.00, 6.00, 6.08, 6.15])
    E_m = E_n[None, None, :] - 0.05 * (c - 1)[..., None]
    hw = (np.array([0., 0., 0., .0348, .0873, .0873])[None, None, :]
          + 0.012 * (1 - c)[..., None])
    rng = np.random.default_rng(1)
    G = (rng.standard_normal((n, n, nm, ne, ne))
         + 1j * rng.standard_normal((n, n, nm, ne, ne))) * 1e-3
    G[0, 0, :3] = 0.0
    Ge = G * 0.4
    Gh = G - Ge

    idx = rng.permutation(n * n)
    qr = np.stack([(idx // n) / n, (idx % n) / n, np.zeros(n * n)], 1)

    def flat(a):
        return a.reshape(n * n, *a.shape[2:])[idx]

    p = tmp_path / "arc.npz"
    np.savez(p, G_grid=flat(G), g2_grid=flat(np.abs(G) ** 2),
             Ge_grid=flat(Ge), Gh_grid=flat(Gh),
             E_m_grid=flat(E_m), hw_grid=flat(hw), E_n_grid=E_n,
             q_red=qr, Q_red=np.zeros(3), mesh=np.array([n, n, 1]))
    return p


@pytest.fixture(autouse=True)
def _no_report(monkeypatch):
    """CLI tests write no PDF report unless a test asks for one."""
    monkeypatch.setenv("XPHD_REPORT", "0")
