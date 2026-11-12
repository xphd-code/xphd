"""Write the files the fake yambopy expects: ns.db1, ndb.elph, 9 BSE dbs, Dmats."""
import os
import sys
import numpy as np

root = sys.argv[1]
os.makedirs(os.path.join(root, "SAVE"), exist_ok=True)
os.makedirs(os.path.join(root, "BSE"), exist_ok=True)
open(os.path.join(root, "SAVE", "ns.db1"), "w").close()
open(os.path.join(root, "ndb.elph"), "w").close()
for j in range(9):
    open(os.path.join(root, "BSE", f"ndb.BS_diago_Q{j+1}"), "w").close()
np.save(os.path.join(root, "Dmats.npy"), np.tile(np.eye(4), (1, 9, 1, 1)))
