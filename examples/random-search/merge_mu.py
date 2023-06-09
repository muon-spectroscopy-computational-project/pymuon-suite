import sys
import glob
from ase import io, Atoms
from ase.build import make_supercell

from pymuonsuite.utils import make_3x3, get_element_from_custom_symbol
from pymuonsuite.schemas import load_input_file, MuAirssSchema

atoms = io.read(sys.argv[1])
params = {}
if len(sys.argv) > 2:
    params = load_input_file(sys.argv[2], MuAirssSchema)
    scell = make_3x3(params["supercell"])
    atoms = make_supercell(atoms, scell)

mu_symbol_element = get_element_from_custom_symbol(params.get("mu_symbol", "H:mu"))

for f in glob.glob("muon-airss-out/dftb+/*/geo_end.xyz"):
    a = io.read(f)
    atoms += Atoms(mu_symbol_element, positions=a.get_positions()[-1, None, :])

io.write("merged.cif", atoms)
