"""A planar binary honeycomb with NN and NNN springs."""
import sys

import numpy as np

conv = sys.argv[1] if len(sys.argv) > 1 else "lattice"
layout = sys.argv[2] if len(sys.argv) > 2 else "GaN"
if layout == "GaN":
    scale = 1.0
    frac = np.array([[0.0, 0.0], [1 / 3, 2 / 3]])
    mass = np.array([69.7, 14.0])
else:
    scale = 0.987484665
    frac = np.array([[0.0, 0.0], [2 / 3, 1 / 3]])
    mass = np.array([14.0, 69.7])
A = scale * np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2]])
B = np.linalg.inv(A).T
tau = frac @ A
kNN = dict(L=5.0, T=2.0, Z=1.2)
kNNN = dict(L=1.1, T=0.4, Z=0.3)


def block(d, k):
    dh = d / np.linalg.norm(d)
    P = np.outer(dh, dh)
    F = np.zeros((3, 3))
    F[:2, :2] = -(k["L"] * P + k["T"] * (np.eye(2) - P))
    F[2, 2] = -k["Z"]
    return F


Kc = np.array([1 / 3, 1 / 3]) @ B
D = np.zeros((6, 6), complex)
self_ = np.zeros((2, 3, 3))
dNN, dNNN = scale / np.sqrt(3), scale
for a in range(2):
    for b in range(2):
        for i in range(-3, 4):
            for j in range(-3, 4):
                RL = np.array([i, j]) @ A
                d = RL + tau[b] - tau[a]
                r = np.linalg.norm(d)
                if r < 1e-9:
                    continue
                if abs(r - dNN) < 1e-6:
                    F = block(d, kNN)
                elif abs(r - dNNN) < 1e-6:
                    F = block(d, kNNN)
                else:
                    continue
                ph = np.exp(2j * np.pi * (Kc @ RL)) if conv == "lattice" else np.exp(2j * np.pi * (Kc @ d))
                D[3 * a:3 * a + 3, 3 * b:3 * b + 3] += F * ph / np.sqrt(mass[a] * mass[b])
                self_[a] -= F
for a in range(2):
    D[3 * a:3 * a + 3, 3 * a:3 * a + 3] += self_[a] / mass[a]
assert np.allclose(D, D.conj().T)
w2, V = np.linalg.eigh(D)
freq = np.sqrt(np.abs(w2)) * 300.0
disp = V / np.sqrt(np.repeat(mass, 3))[:, None]
disp /= np.linalg.norm(disp, axis=0)
with open("matdyn.modes", "w") as f:
    f.write("\n     diagonalizing the dynamical matrix ...\n\n")
    f.write(f" q = {Kc[0]:12.4f}{Kc[1]:12.4f}{0.0:12.4f}\n")
    f.write(" " + "*" * 74 + "\n")
    for m in range(6):
        f.write(f"     freq ({m + 1:5d}) = {freq[m] / 33.356:15.6f} [THz] = {freq[m]:15.6f} [cm-1]\n")
        for a in range(2):
            v = disp[3 * a:3 * a + 3, m]
            f.write(" ( " + "  ".join(f"{x.real:10.6f} {x.imag:10.6f}" for x in v) + " )\n")
    f.write(" " + "*" * 74 + "\n")
