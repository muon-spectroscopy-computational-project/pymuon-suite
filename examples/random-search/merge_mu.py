import sys
import glob
from ase import io, Atoms
from ase.build import make_supercell

from pymuonsuite.utils import make_3x3
from pymuonsuite.schemas import load_input_file, MuAirssSchema

atoms = io.read(sys.argv[1])
if len(sys.argv) > 2:
    params = load_input_file(sys.argv[2], MuAirssSchema)
    scell = make_3x3(params["supercell"])
    atoms = make_supercell(atoms, scell)

for f in glob.glob("muon-airss-out/dftb+/*/geo_end.xyz"):
    a = io.read(f)
    atoms += Atoms("H", positions=a.get_positions()[-1, None, :])

io.write("merged.cif", atoms)
