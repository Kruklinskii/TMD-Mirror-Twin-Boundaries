# MTB TMD Generator Documentation
This PYTHON application generates a POSCAR file for a unit cell of 4|4P mirror twin boundary (MTB) triangular structures for a given transition metal dichalcogenide (TMD). The program uses a config.txt file to read all the necessary input data. The output file has a name format of e.g. POSCAR_Te98Mo55_MTB which indicated the atomic composition of the resulting primitive unit cell.

## Required third‑party packages
- Python
- numpy 
- pymatgen

pip install -r requirements.txt

## config.txt Format
Each line contains the following data:
1. Chemical symbol for the transition metal.
2. Chemical symbol for the chalcogen.
3. Lattice parameter in angstrom (distance between neighbouring atoms of the same species).
4. Vertical distance between the two chalcogens in a unit cell in angstroms.
5. Vertical vacuum separation between TMD layers in angstroms.
6. Size of MTB triangle edge in units of lattice parameter (integer).
7. Superlattice vector 1 (three integers separated by spaces).
8. Superlattice vector 2 (three integers separated by spaces).

config.txt format remarks:
- The config file must be in the same working directory as the script itself.
- The name of the config file must be strictly "config.txt".
- The config file must contain all eight data entries on separate lines.
- Avoid anything that does not adhere to the config format.
- Avoid unnecessary spaces.
- The superlattice vectors in (7) and (8) define the lattice vectors of the new system and are to be written as three numbers separated by spaces, e.g. "1 2 0".
- Superlattice vectors are defined as fractional coordinates of the original TMD lattice vectors $\overrightarrow{a} = (l, 0, 0)$, $\overrightarrow{b} = (l/2, l*\sqrt{3}/2, 0)$ and $\overrightarrow{c} = (0, 0, vacuum)$ where $l$ is a lattice parameter in (3).

## config.txt Example
Mo \
Te \
3.52 \
3.40 \
20 \
6 \
5 2 0 \
-2 7 0 

---
Made by Daniil Kruklinskii \
dan261299@gmail.com
