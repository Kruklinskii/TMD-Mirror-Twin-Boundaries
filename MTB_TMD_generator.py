"""
MTB TMD Generator Documentation

This PYTHON application generates a POSCAR file for a unit cell of 4|4P mirror twin boundary (MTB) triangular structures for a given transition metal dichalcogenide (TMD). The program uses a config.txt file to read all the necessary input data. The output file has a name format of e.g. POSCAR_Te98Mo55_MTB which indicated the atomic composition of the resulting primitive unit cell.

------------------------------------------------------------------------
Required third‑party packages

- numpy 
- pymatgen

------------------------------------------------------------------------
config.txt Format

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
- Superlattice vectors are defined as fractional coordinates of the original TMD lattice vectors a = (l, 0, 0), b = (l/2, l*sqrt{3}/2, 0) and c = (0, 0, vacuum) where l is a lattice parameter in (3).

------------------------------------------------------------------------
config.txt Example

Mo 
Te 
3.52 
3.40 
20 
6 
5 2 0 
-2 7 0 

------------------------------------------------------------------------
Made by Daniil Kruklinskii
dan261299@gmail.com
"""

from sys import exit
from typing import Union
from pathlib import Path
import logging

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.core.structure import StructureError
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp import Poscar

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='\n[%(levelname)s] %(asctime)s - %(message)s')
# Custom data type for config file
ConfigData = tuple[str, str, float, float, float, int, list[int, int, int], list[int, int, int]]
# CONSTANTS
OFFSET = 1e-4
CONFIG_FILE_PATH: str = "config.txt"


def print_startup_info() -> None:
    """Prints the program's startup information."""
    logging.info("\n-------------------MTB TMD Generator-------------------"
                 "\nAuthor: Daniil Kruklinskii dan261299@gmail.com\n")


def read_config(config_file_path: str) -> ConfigData:
    """
    Reads the config file and extract all the information.

    This function opens the configuration file specified by `config_file_path`
    and reads its contents line by line. It parses the lines to extract
    various configuration details like transition metal element, chalcogen element,
    unit vector length, etc. It returns a tuple containing all the extracted
    configuration values.

    Args:
        config_file_path (str): The path to the config file.

    Returns:
        ConfigData: A tuple containing all the configuration values extracted
            from the file.
    """
    try:
        config_file_path = read_path(config_file_path)

        with open(config_file_path, "r", encoding="utf-8") as f:
            lines: list[str] = f.readlines()
            transition_metal: str = lines[0].strip()
            chalcogen: str = lines[1].strip()
            l: float = float(lines[2].strip()) # Unit vector a length
            d: float = float(lines[3].strip()) # Distance between two chalcogens
            vacuum: float = float(lines[4].strip())
            triangle_size: int = int(lines[5].strip()) + 1 # The actual size in the script is smaller
            v1: list[int, int, int] = lines[6].strip().split(" ") # Superlattice vector 1
            v2: list[int, int, int] = lines[7].strip().split(" ") # Superlattice vector 2
            v1 = [int(x) for x in v1]
            v2 = [int(x) for x in v2]

            return (transition_metal, chalcogen, l, d, vacuum, triangle_size, v1, v2)

    except FileNotFoundError as path:
        logging.error(f"{path} NOT FOUND.\n"
                      "Please check the config file and ensure the"
                      " file/directory exists.")
        exit(1)
    # Catch potential error due to the incomplete config file.
    except IndexError:
        logging.error("INVALID DATA FORMAT IN CONFIG FILE.\n"
                      "Check that all the data is provided in the config"
                      " file. Chack the data in config.")
        exit(1)
    except Exception as e:
        logging.critical(f"UNEXPECTED ERROR: {e}.\n"
                         "Something went wrong during config file reading.")
        exit(2)


def read_path(path: str) -> Path:
    """
    Converts string into Path datatype from pathlib.

    Args:
        path (str): String that will be converted into Path.

    Raises:
        FileNotFoundError: Error is raised in case the file/directory does not
            exist.

    Returns:
        Path: Path of the file/directory in the Path format.

    """
    path = Path(path)

    if path.exists():
        return path
    else:
        raise FileNotFoundError(path)


def create_unit_cells(transition_metal: str, chalcogen: str, l: float, d: float, vacuum: float = 20.0) -> tuple[Structure, Structure]:
    """
    Generates two 2D unit cells (original and mirrored) for a transition‑metal dichalcogenide.

    Args:
        transition_metal (str): Chemical symbol of the transition metal (e.g., "Mo").
        chalcogen (str): Chemical symbol of the chalcogen (e.g., "S").
        l (float): In‑plane lattice constant (in angstroms).
        d (float): Vertical separation between chalcogen layers in a unit cell (in angstroms).
        vacuum (float, optional): Height of the vacuum layer along the c‑axis (in angstroms). Defaults to 20.0.

    Returns:
        tuple[Structure, Structure]: 
            - structure_A: The original monolayer unit cell (A‑type stacking).  
            - structure_B: The mirrored monolayer unit cell (B‑type stacking).
    """
    # Define important vectors
    a = np.array([l, 0, 0])  # Lattice constant 'a' in angstroms
    b = np.array([l/2, l*np.sqrt(3)/2, 0])  # Lattice constant 'b' in angstroms
    c = np.array([0, 0, vacuum])  # Lattice constant 'c' in angstroms
    
    # Define the lattice
    lattice_matrix = np.array([a, b, c])
    lattice = Lattice(lattice_matrix)
    
    # Atomic basis positions for transition metal and two chalcogens
    # Original structure
    species_A = [chalcogen, chalcogen, transition_metal]
    frac_coords_A = np.array([[1/3,   1/3,   -d/2/vacuum + 1/2], # Chalcogen
                              [1/3,   1/3,   d/2/vacuum + 1/2],  # Chalcogen
                              [0.000, 0.000, 1/2]])              # Transition metal

    # Mirrired structure
    species_B = [transition_metal, chalcogen, chalcogen]
    frac_coords_B = np.array([[1/3, 1/3, 1/2],        # Mo atom
                              [0.000, 0.000, -d/2/vacuum + 1/2],  # S atom
                              [0.000, 0.000, d/2/vacuum + 1/2]])        # S atom
    
    structure_A = Structure(lattice, species_A, frac_coords_A, coords_are_cartesian=False)
    structure_B = Structure(lattice, species_B, frac_coords_B, coords_are_cartesian=False)

    return structure_A, structure_B


def generate_MTB_unit_cell(structure_A: Structure, structure_B: Structure, v1: np.ndarray, v2: np.ndarray, triangle_size: int, l: float, offset: float = 1e-4) -> Structure:
    """
    Construct the 4|4P mirror twin boundary (MTB) triangles unit cell from two TMD structures (original and mirrored).

    This function traverses the 2D superlattice defined by supervectors v1 and v2, 
    tests each translated atomic position against lattice‐type “A” and “B” validity 
    regions, and collects only those atoms that belong to the MTB.  The resulting 
    Cartesian coordinates and species are used to build and return a pymatgen Structure.

    Args:
        structure_A (Structure): The original monolayer Structure.
        structure_B (Structure): The mirrored monolayer Structure.
        v1 (np.ndarray): Fractional supervector 1 defining the 2D supercell lattice.
        v2 (np.ndarray): Fractional supervector 2 defining the 2D supercell lattice.
        triangle_size (int): A size of the equilateral triangle edge in units of l.
        l (float): In‐plane lattice constant (angstroms).
        offset (float, optional): Tolerance for boundary inclusion. Defaults to 1e-4.

    Returns:
        Structure: A pymatgen Structure object representing the MTB trianles supercell
            with correct atomic positions in Cartesian coordinates.
    """
    cart_coords_TM = [] # Cartesian coords for transition metals
    cart_coords_C = []  # Cartesian coords for chalcogens
    atoms_TM = [] # List of all transition metals
    atoms_C = [] # List of all chalcogens
    cart_coords = [] # Final list of all atomic coordinates
    species = [] # Final list of all species
    structure = None # Final structure

    lattice_matrix = structure_A.lattice.matrix
    v1_cart = frac_to_cart(lattice_matrix, v1)
    v2_cart = frac_to_cart(lattice_matrix, v2)
    transition_metal = structure_B.species[0]
    chalcogen = structure_B.species[1]

    frac_B_shift = np.array([1/3, 1/3, 0]) # Shift between lattices A and B
    superlattice = Lattice([v1_cart, v2_cart, lattice_matrix[2]])
    m1, m2, n1, n2 = traverse_range(v1_cart, v2_cart)
    
    for i in range(m1, m2):
        for j in range(n1, n2):
            frac_shift = np.array([i, j, 0])
            positions_A = [frac_to_cart(lattice_matrix, frac_shift + structure_A.frac_coords[0]),
                           frac_to_cart(lattice_matrix, frac_shift + structure_A.frac_coords[1]),
                           frac_to_cart(lattice_matrix, frac_shift + structure_A.frac_coords[2])]
            positions_B = [frac_to_cart(lattice_matrix, frac_shift + structure_B.frac_coords[0] + frac_B_shift),
                           frac_to_cart(lattice_matrix, frac_shift + structure_B.frac_coords[1] + frac_B_shift),
                           frac_to_cart(lattice_matrix, frac_shift + structure_B.frac_coords[2] + frac_B_shift)]
            species_A = structure_A.species
            species_B = structure_B.species
            
            for pos, specie in zip(positions_A, species_A):
                if is_valid_point(pos, v1_cart, v2_cart, l, triangle_size, "A", offset):
                    if specie == transition_metal:
                        cart_coords_TM.append(pos)
                        atoms_TM.append(specie)
                    else:
                        cart_coords_C.append(pos)
                        atoms_C.append(specie)
    
            for pos, specie in zip(positions_B, species_B):
                if is_valid_point(pos, v1_cart, v2_cart, l, triangle_size, "B", offset):
                    if specie == transition_metal:
                        cart_coords_TM.append(pos)
                        atoms_TM.append(specie)
                    else:
                        cart_coords_C.append(pos)
                        atoms_C.append(specie)
    
    cart_coords = cart_coords_TM + cart_coords_C
    species = atoms_TM + atoms_C

    try:
        structure = Structure(
            superlattice,
            species,
            cart_coords,
            coords_are_cartesian=True,
            validate_proximity=True
            )

        return structure
    except StructureError as exc:
        logging.error("INVALID ATOMIC STRUCTURE.\n"
                      "Atoms are too close. Check the data provided in the config file (especially the distances).")
        exit(1)
    except Exception as e:
        logging.critical(f"UNEXPECTED ERROR: {e}.\n"
                         "Something went wrong during structure generation.")
        exit(2)

def frac_to_cart(lattice_matrix: Union[np.ndarray, list], frac_coords: Union[np.ndarray, list]) -> np.ndarray:
    """
    Converts fractional coordinates into Cartesian coordinates using a given lattice matrix.

    Args:
        lattice_matrix (Union[np.ndarray, list]): A 3x3 matrix where each row is a lattice vector.
        frac_coords (Union[np.ndarray, list]): A list or 1D NumPy array of 3 fractional coordinates.

    Returns:
        np.ndarray: A 1D NumPy array of 3 Cartesian coordinates.
    """
    lattice_matrix = np.asarray(lattice_matrix)
    frac_coords = np.asarray(frac_coords)

    # Compute the Cartesian coordinates
    cart_coords = np.dot(lattice_matrix.T, frac_coords)

    return cart_coords


def get_translated_points(point: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> list[np.ndarray]:
    """
    Get eight points (+/-v1, +/-v2, +/-v1 +/- v2) defined by v1 and v2 around a given point.

    Args:
        point (np.ndarray): An original point.
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.

    Returns:
        list[np.ndarray]: List of 3D vectors.
    """
    return [point, point + v1, point - v1, point + v2, point - v2, point + v1 + v2, point - v1 - v2, point + v1 - v2, point - v1 + v2]


def is_inside_unit_cell(point: np.ndarray, v1: np.ndarray, v2: np.ndarray, offset: float = 1e-4) -> bool:
    """
    Determines whether a point lies inside a 2D unit cell defined by two lattice vectors using barycentric coordinates.

    Args:
        point (np.ndarray): A 3-element array representing the point to test.
        v1 (np.ndarray): A 3-element array representing the first lattice vector.
        v2 (np.ndarray): A 3-element array representing the second lattice vector.
        offset (float, optional): A small numerical tolerance for edge inclusion. Default is 1e-4.

    Returns:
        bool: True if the point is inside the unit cell, False otherwise.
    """
    # Compute the 2D determinant of the lattice parallelogram
    D_pq = np.linalg.det([v1[0:2], v2[0:2]])
    D_rq = np.linalg.det([point[0:2], v2[0:2]])
    D_pr = np.linalg.det([v1[0:2], point[0:2]])

    # Barycentric coordinates
    M_rq = D_rq / D_pq
    M_pr = D_pr / D_pq

    return (0 - offset <= M_rq < 1 - offset) and (0 - offset <= M_pr < 1 - offset)


def is_inside_triangle(point: np.ndarray, triangle_size: int, l: float, offset: float = 1e-4) -> bool:
    """
    Checks whether a point lies inside a triangle formed by inequalities
    y > 0, y < sqrt(3)*x and y < -sqrt(3)*x + sqrt(3)*triangle_size*l.
    
    Args:
        point (np.ndarray): A 3-element array representing the point to test.
        triangle_size (int): A size of the equilateral triangle edge in units of l.
        l (float): Size units of the triangle.
        offset (float, optional): A small numerical tolerance for edge inclusion. Default is 1e-4.

    Returns:
        bool: True if the point is inside the triangle, False otherwise.
    """
    return (point[1] > 0 - offset and point[1] <  np.sqrt(3)*point[0] + offset and point[1] < -np.sqrt(3)*point[0] + np.sqrt(3)*triangle_size*l + offset)


def is_outside_triangle(point: np.ndarray, triangle_size: int, l: float, offset: float = 1e-4) -> bool:
    """
    Checks whether a point lies outside a triangle formed by inequalities
    y > 0, y < sqrt(3)*x and y < -sqrt(3)*x + sqrt(3)*triangle_size*l.
    
    Args:
        point (np.ndarray): A 3-element array representing the point to test.
        triangle_size (int): A size of the equilateral triangle edge in units of l.
        l (float): Size units of the triangle.
        offset (float, optional): A small numerical tolerance for edge inclusion. Default is 1e-4.

    Returns:
        bool: True if the point is outside the triangle, False otherwise.
    """
    return (point[1] <= 0 + offset or point[1] >= np.sqrt(3)*point[0] - offset or point[1] >= -np.sqrt(3)*point[0] + np.sqrt(3)*triangle_size*l - offset)


def is_valid_point(point: np.ndarray, v1: np.ndarray, v2: np.ndarray, l: float, triangle_size: float, lattice_type: str, offset: float = 1e-4) -> bool:
    """
    Determine whether a point is valid within a periodic triangular lattice and is inside a unit
    cell formed by two supervectors of type A or B.

    Args:
        point (np.ndarray): The 3D coordinate of the point to test.
        v1 (np.ndarray): The first lattice vector defining the periodic unit cell.
        v2 (np.ndarray): The second lattice vector defining the periodic unit cell.
        l (float): Size units of the triangle.
        triangle_size (float): A size of the equilateral triangle edge in units of l.
        lattice_type (str): The lattice type, either "A" or "B".  
            - "A": Valid points must lie in the unit cell but **outside** the triangle.  
            - "B": Valid points must lie in the unit cell and **inside** the triangle.
        offset (float, optional): A small numerical tolerance for edge inclusion. Default is 1e-4.

    Returns:
        bool: True if the point satisfies the periodicity and triangle inclusion/exclusion rules
              for the specified lattice type; False otherwise.
    """
    #offset = 1e-4  # Small tolerance to avoid precision issues

    # Generate 8 points (±v1, ±v2, ±v1±v2)
    points = get_translated_points(point, v1, v2)

    # Makes sure that the point is inside the lattice formed by the supervectors (ensures periodicity)
    in_unit_cell = is_inside_unit_cell(point, v1, v2, offset)

    # The triangle is defined by 3 inequalities forming an upward-pointing triangle.
    in_triangle = any((is_inside_triangle(pt, triangle_size, l, offset)) for pt in points)
    outside_triangle = all((is_outside_triangle(pt, triangle_size, l, offset)) for pt in points) # Not just !in_triangle

    # Point in lattice A must be outside the triangle
    if lattice_type == "A" and in_unit_cell and outside_triangle:
        return True

    # Point in lattice B must be inside the triangle
    elif lattice_type == "B" and in_unit_cell and in_triangle:
        return True

    return False


def traverse_range(v1: np.ndarray, v2: np.ndarray) -> tuple[int, int, int, int]:
    """
    Compute integer ranges m ∈ [m1, m2] and n ∈ [n1, n2] such that
    the set {m*v1 + n*v2 | m1 ≤ m ≤ m2, n1 ≤ n ≤ n2} contains all integer-grid 
    points inside the parallelogram spanned by v1 and v2.

    Args:
        v1 (np.ndarray): First 2D vector [v1_x, v1_y].
        v2 (np.ndarray): Second 2D vector [v2_x, v2_y].

    Returns:
        tuple[int, int, int, int]: (m1, m2, n1, n2) integer bounds for m and n.
    """
    m1 = - (abs(int(v1[0])) + abs(int(v2[0])))
    m2 =   abs(int(v1[0])) + abs(int(v2[0]))
    n1 = - (abs(int(v1[1])) + abs(int(v2[1])))
    n2 =   abs(int(v1[1])) + abs(int(v2[1]))
    return m1, m2, n1, n2


def save_to_poscar(structure: Structure) -> None:
    """
    Saves a pymatgen Structure object to a POSCAR file named according to its formula.

    Args:
        structure (Structure): The pymatgen Structure to be written to POSCAR format.

    Returns:
        None: Writes a file named "POSCAR_<formula>_MTB" in the current working directory.
    """
    comment = "_".join([structure.formula.replace(" ", ""), "MTB"])
    poscar = Poscar(structure, comment=comment)
    
    # Write the POSCAR file
    poscar_name = f"POSCAR_{comment}"
    poscar.write_file(poscar_name)
    logging.info(f"POSCAR file saved as \"{poscar_name}\"")


def main(offset: float) -> None:

    # Starting text
    print_startup_info()

    # Reading the data from config.txt
    transition_metal, chalcogen, l, d, vacuum, triangle_size, v1, v2 = read_config(CONFIG_FILE_PATH)

    # Get the two TMD hexagonal unit cell structures (original and mirrored)
    structure_A, structure_B = create_unit_cells(transition_metal, chalcogen, l, d, vacuum)

    # Generate 4|4P mirror twin boundary unit cell
    structure = generate_MTB_unit_cell(structure_A, structure_B, v1, v2, triangle_size, l, offset)

    # Save the POSCAR
    save_to_poscar(structure)


if __name__ == "__main__":
    main(OFFSET)
