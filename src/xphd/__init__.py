"""xphd -- exciton-phonon dynamics from first principles."""
from .core.interp import fourier_refine, ringing_report
from .core.mesh import (blockwise, hex_norm, mesh_indices, shells,
                        shift_grid, to_grid, ws_vectors)
from .core.stats import (HBAR_EV_PS, KB_EV, bose, degeneracy_groups,
                         envelope, shell_maxima)
from .io.excph import ExcPhArchive, load_archives
from .io.matdyn import compare_to_archive, read_freq
from .linewidth import (Interference, LinewidthResult, compute,
                        fit_acoustic, interference,
                        linewidth_one)
from . import (arpes, bte, chirality, excsym, lineshape, modes, parallel,
               selfenergy, symmetry)
from .lineshape import SigmaMatrix, epsilon2, sigma_matrix
from .selfenergy import SelfEnergy, kramers_kronig, self_energy
from .sweep import LinewidthField, sweep
from .tetra import delta_weights_2d

__version__ = "2.0.1"
__author__ = "R. Saraswat, S. Bhattacharya, R. Verma, M. Ansari"
__affiliation__ = "Indian Institute of Information Technology, Allahabad"

__all__ = [
    "ExcPhArchive", "load_archives", "read_freq", "compare_to_archive", "bte",
    "compute", "linewidth_one", "LinewidthResult",
    "interference", "Interference", "fit_acoustic",
    "selfenergy", "self_energy", "SelfEnergy", "kramers_kronig",
    "lineshape", "sigma_matrix", "SigmaMatrix", "epsilon2",
    "modes",
    "arpes",
    "excsym",
    "sweep", "LinewidthField", "symmetry",
    "delta_weights_2d", "fourier_refine", "ringing_report",
    "mesh_indices", "to_grid", "blockwise", "ws_vectors", "hex_norm",
    "shells", "shift_grid",
    "bose", "degeneracy_groups", "envelope", "shell_maxima",
    "KB_EV", "HBAR_EV_PS", "__version__",
]
