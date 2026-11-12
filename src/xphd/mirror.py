"""Mirror (sigma_h) parity for layers that are not planar, and for spinors."""
import numpy as np

__all__ = ["mirror_plane_z", "sigma_h_permutation", "phonon_parity",
           "band_labels"]


def mirror_plane_z(pos_car):
    """z of the mirror plane: halfway between the highest and lowest atom."""
    z = np.asarray(pos_car, float)[:, 2]
    return 0.5 * (z.max() + z.min())


def sigma_h_permutation(pos_car, species, cell, tol=1e-3):
    """pi[kappa] = the atom that sigma_h carries atom kappa onto."""
    pos = np.asarray(pos_car, float)
    cell = np.asarray(cell, float)
    z0 = mirror_plane_z(pos)
    img = pos.copy()
    img[:, 2] = 2 * z0 - pos[:, 2]
    inv = np.linalg.inv(cell)
    perm = np.full(len(pos), -1)
    for k in range(len(pos)):
        for m in range(len(pos)):
            if species[m] != species[k]:
                continue
            d = (img[k] - pos[m]) @ inv
            d[:2] -= np.rint(d[:2])
            if np.linalg.norm(d @ cell) < tol:
                perm[k] = m
                break
        if perm[k] < 0:
            raise ValueError(f"atom {k} has no mirror image: the layer has no "
                             "horizontal mirror plane")
    if len(np.unique(perm)) != len(perm):
        raise ValueError("sigma_h does not map the atoms one to one")
    return perm


def phonon_parity(evec, perm):
    """sigma_h parity of phonon eigenvectors."""
    e = np.asarray(evec)
    Me = e * np.array([1.0, 1.0, -1.0])
    num = np.sum(np.conj(e[..., perm, :]) * Me, axis=(-2, -1))
    den = np.sum(np.abs(e) ** 2, axis=(-2, -1))
    return np.real(num / den)


def band_labels(diag, tol=0.1):
    """+-1 labels from the diagonal of D(sigma_h), and which kind of bands."""
    d = np.asarray(diag)
    near_real = np.abs(np.abs(d.real) - 1) < tol
    near_imag = np.abs(np.abs(d.imag) - 1) < tol
    if near_imag.sum() > near_real.sum():
        return np.sign(d.imag), "spinor", near_imag
    return np.sign(d.real), "spinless", near_real
