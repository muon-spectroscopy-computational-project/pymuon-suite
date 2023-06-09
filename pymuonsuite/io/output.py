"""
Author: Simone Sturniolo and Adam Laverack
"""


import os
from datetime import datetime

from ase import Atoms, io

import numpy as np

from pymuonsuite.io.castep import ReadWriteCastep
from pymuonsuite.io.dftb import ReadWriteDFTB
from pymuonsuite.utils import make_muonated_supercell, safe_create_folder

from scipy.constants import physical_constants as pcnst

from soprano.collection import AtomsCollection
from soprano.utils import silence_stdio


def write_cluster_report(args, params, clusters):
    if params["clustering_method"] == "hier":
        clustinfo = """
Clustering method: Hierarchical
    t = {t}
""".format(
            t=params["clustering_hier_t"]
        )
    elif params["clustering_method"] == "kmeans":
        clustinfo = """
Clustering method: k-Means
    k = {k}
""".format(
            k=params["clustering_kmeans_k"]
        )

    with open(params["name"] + "_clusters.txt", "w") as f:
        f.write(
            """
****************************
|                          |
|       PYMUON-SUITE       |
|  MuAirss Clusters report |
|                          |
****************************

Name: {name}
Date: {date}
Structure file(s): {structs}
Parameter file: {param}
{clustinfo}

*******************

""".format(
                name=params["name"],
                date=datetime.now(),
                structs=args.structures,
                param=args.parameter_file,
                clustinfo=clustinfo,
            )
        )

        for name, cdata in clusters.items():
            f.write("Clusters for {0}:\n".format(name))

            if params["clustering_save_min"] or params["clustering_save_type"]:
                if params["clustering_save_folder"] is not None:
                    clustering_save_path = safe_create_folder(
                        params["clustering_save_folder"]
                    )
                else:
                    clustering_save_path = safe_create_folder(
                        "{0}_clusters".format(params["name"])
                    )
                if not clustering_save_path:
                    raise RuntimeError("Could not create folder {0}")

            for calc, clusts in cdata.items():
                # Computer readable
                fdat = open(
                    params["name"] + "_{0}_{1}_clusters.dat".format(name, calc),
                    "w",
                )

                f.write("CALCULATOR: {0}\n".format(calc))
                (cinds, cgroups), ccolls, gvecs = clusts

                f.write("\t{0} clusters found\n".format(max(cinds)))

                min_energy_structs = []

                for i, g in enumerate(cgroups):
                    f.write(
                        "\n\n\t-----------\n\tCluster "
                        "{0}\n\t-----------\n".format(i + 1)
                    )
                    f.write("\tStructures: {0}\n".format(len(g)))
                    coll = ccolls[i + 1]
                    E = gvecs[g, 0]
                    Emin = np.amin(E)
                    Eavg = np.average(E)
                    Estd = np.std(E)

                    f.write("\n\tEnergy (eV):\n")
                    f.write("\tMinimum\t\tAverage\t\tStDev\n")
                    f.write(
                        "\t{0:.2f}\t\t{1:.2f}\t\t{2:.2f}\n".format(Emin, Eavg, Estd)
                    )
                    structure = coll[np.argmin(E)].structures[0]

                    fdat.write(
                        "\t".join(
                            map(
                                str,
                                [
                                    i + 1,
                                    len(g),
                                    Emin,
                                    Eavg,
                                    Estd,
                                    structure.positions[-1][0],
                                    structure.positions[-1][1],
                                    structure.positions[-1][2],
                                ],
                            )
                        )
                        + "\n"
                    )

                    f.write(
                        "\n\tMinimum energy structure: {0}\n".format(
                            structure.info["name"]
                        )
                    )

                    # update symbol and mass of defect in minimum energy structure
                    structure = coll[np.argmin(E)].structures[0]
                    csp = structure.get_chemical_symbols()[:-1] + [
                        params.get("mu_symbol", "H:mu")
                    ]
                    structure.set_array("castep_custom_species", np.array(csp))
                    masses = structure.get_masses()[:-1]
                    masses = np.append(
                        masses,
                        params.get("particle_mass_amu", pcnst["muon mass in u"][0]),
                    )
                    structure.set_masses(np.array(masses))

                    # Save minimum energy structure
                    if (
                        params["clustering_save_type"] == "structures"
                        or params["clustering_save_min"]
                    ):
                        success = write_structure(
                            params,
                            clustering_save_path,
                            calc,
                            f"{params['name']}_{calc}_min_cluster_{i+1}",
                            structure,
                        )
                        if not success:
                            return

                    min_energy_structs.append(structure)

                    f.write("\n\n\tStructure list:")

                    for j, s in enumerate(coll):
                        if j % 4 == 0:
                            f.write("\n\t")
                        f.write("{0}\t".format(s.info["name"]))

                fdat.close()

                if params["clustering_save_type"] == "input":
                    calc_path = os.path.join(clustering_save_path, calc)

                    sname = "{0}_min_cluster".format(params["name"])

                    io_formats = {
                        "castep": ReadWriteCastep,
                        "dftb+": ReadWriteDFTB,
                    }
                    try:
                        write_method = io_formats[params["clustering_save_format"]](
                            params
                        ).write
                    except KeyError as e:
                        print(
                            "ERROR: Calculator type {0} is not "
                            "recognised. Modify 'clustering_save_format'"
                            " to be one of: {1}".format(e, list(io_formats.keys()))
                        )
                        return

                    if params["clustering_save_format"] == "dftb+":
                        from pymuonsuite.data.dftb_pars import get_license

                        with open(
                            os.path.join(clustering_save_path, "dftb.LICENSE"),
                            "w",
                        ) as license_file:
                            license_file.write(get_license())

                    min_energy_structs = AtomsCollection(min_energy_structs)
                    # here we remove the structure's name so the original
                    # numbering of the structs is removed:
                    for i, a in enumerate(min_energy_structs):
                        min_energy_structs.structures[i].info.pop("name", None)

                    min_energy_structs.save_tree(
                        calc_path,
                        write_method,
                        name_root=sname,
                        opt_args={"calc_type": "GEOM_OPT"},
                        safety_check=2,
                    )

                # Print distance matrix

                f.write("\n\n\t----------\n\n\tSimilarity (ranked):\n")

                centers = np.array([np.average(gvecs[g], axis=0) for g in cgroups])
                dmat = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)

                inds = np.triu_indices(len(cgroups), k=1)
                for i in np.argsort(dmat[inds]):
                    c1 = inds[0][i]
                    c2 = inds[1][i]
                    d = dmat[c1, c2]
                    f.write(
                        "\t{0} <--> {1} (distance = {2:.3f})\n".format(
                            c1 + 1, c2 + 1, d
                        )
                    )

            f.write("\n--------------------------\n\n")

        f.write("\n==========================\n\n")


def write_structure(
    params: dict, clustering_save_path: str, calc: str, seedname: str, structure: Atoms
) -> bool:
    """Writes the minimal energy structure in a cluster to file.

    | Args:
    |   params (dict): PyMuonSuite MUAIRSS parameters.
    |   clustering_save_path (str): Directory to save the clusters in.
    |   calc (str): Calculator used for the optimisation.
    |   seedname (str): Seedname for the saved structure files.
    |   structure (Atoms): ASE Atoms to save.
    |
    | Returns:
    |   (bool): Whether the structure was written to file successfully.
    """
    # For backwards-compatibility with old pymuonsuite versions
    if params["clustering_save_min"]:
        if params["clustering_save_format"] is None:
            params["clustering_save_format"] = "cif"

    try:
        calc_path = os.path.join(clustering_save_path, calc)
        if not os.path.exists(calc_path):
            os.mkdir(calc_path)

        with silence_stdio():
            if calc == "uep":
                structure = make_muonated_supercell(
                    structure, params["supercell"], params["mu_symbol"]
                )

            if params["clustering_save_format"].lower() == "cell":
                # Account for CASTEP calc settings
                read_write_castep = ReadWriteCastep(params)
                read_write_castep.write_cell(structure, calc_path, seedname)
            else:
                filename = f"{seedname}.{params['clustering_save_format']}"
                io.write(os.path.join(calc_path, filename), structure)

        return True

    except io.formats.UnknownFileTypeError as e:
        print(
            f"ERROR: File format '{e}' is not recognised. "
            "Modify 'clustering_save_format' and try again."
        )
        return False
    except ValueError as e:
        print(f"ERROR: {e}. Modify 'clustering_save_format' and try again.")
        return False


def write_phonon_report(args, params, phdata):
    with open(params["name"] + "_phonons.txt", "w") as f:
        f.write(
            """
    ****************************
    |                          |
    |       PYMUON-SUITE       |
    |    ASE Phonons report    |
    |                          |
    ****************************

    Name: {name}
    Date: {date}
    Structure file: {structs}
    Parameter file: {param}

    *******************

    """.format(
                name=params["name"],
                date=datetime.now(),
                structs=args.structure_file,
                param=args.parameter_file,
            )
        )

        # Write k-point path
        f.write("K-point Path: \n")
        for kp in phdata.path:
            f.write("\t{0}\n".format(kp))

        f.write("\n\n------------------\n\n")

        for i, kp in enumerate(phdata.path):
            f.write("K-point {0}: {1}\n\n".format(i + 1, kp))
            # Write frequencies
            f.write("\tFrequencies (cm^-1): \n")
            for j, om in enumerate(phdata.frequencies[i]):
                f.write("\t{0}\t{1}\n".format(j + 1, om))

            f.write("\n\tDisplacements: \n")
            for j, m in enumerate(phdata.modes[i]):
                f.write("\t\tMode {0}:\n".format(j + 1))
                f.write("\t\tAtom x\t\t\t y\t\t\t z\n")
                for k, d in enumerate(m):
                    d = np.real(d)
                    f.write("\t\t{0}\t{1: .6f}\t{2: .6f}\t{3: .6f}\n".format(k + 1, *d))


def write_symmetry_report(args, symdata, wpoints, fpos):
    print("Wyckoff points symmetry report for {0}".format(args.structure))
    print("Space Group International Symbol: " "{0}".format(symdata["international"]))
    print("Space Group Hall Number: " "{0}".format(symdata["hall_number"]))
    print("Absolute\t\tFractional\t\tHessian constraints\tOccupied")

    # List any Wyckoff point that does not already have an atom in it
    vformat = "[{0:.3f} {1:.3f} {2:.3f}]"
    for wp in wpoints:
        occ = np.any(
            np.isclose(np.linalg.norm(fpos - wp.fpos, axis=1), 0, atol=args.symprec)
        )
        ps = vformat.format(*wp.pos)
        fps = vformat.format(*wp.fpos)
        print("{0}\t{1}\t{2}\t\t\t{3}".format(ps, fps, wp.hessian, "X" if occ else ""))
