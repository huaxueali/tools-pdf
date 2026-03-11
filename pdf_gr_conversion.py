"""
pdf_gr_conversion.py

Utilities to convert experimental reduced PDF G(r) to pair correlation function g(r).

Definition:
    g(r) = G(r) / (4 * pi * r * rho0) + 1

where:
    - G(r): reduced pair distribution function
    - g(r): pair correlation function
    - rho0: number density in atoms / Å^3

"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple, Union, Optional

import numpy as np


ArrayLike = Union[np.ndarray, Iterable[float]]


def number_density_from_mass_density(
    mass_density_g_cm3: float,
    molar_mass_g_mol: float,
    atoms_per_formula_unit: float = 1.0,
) -> float:
    """
    Convert mass density to number density in atoms / Å^3.

    Parameters
    ----------
    mass_density_g_cm3 : float
        Mass density in g/cm^3.
    molar_mass_g_mol : float
        Molar mass of the chemical formula unit in g/mol.
    atoms_per_formula_unit : float, optional
        Number of atoms per formula unit.
        Examples:
            Si -> 1
            SiO2 -> 3
            Al2O3 -> 5

    Returns
    -------
    float
        Number density in atoms / Å^3.
    """
    avogadro = 6.02214076e23  # mol^-1
    # formula units / cm^3
    n_formula_cm3 = mass_density_g_cm3 * avogadro / molar_mass_g_mol
    # atoms / cm^3
    n_atoms_cm3 = n_formula_cm3 * atoms_per_formula_unit
    # 1 cm^3 = 1e24 Å^3
    return n_atoms_cm3 / 1e24


def formula_mass(composition: dict[str, float]) -> float:
    """
    Calculate molar mass from a composition dictionary.

    Parameters
    ----------
    composition : dict[str, float]
        Example:
            {"Si": 1}
            {"Si": 1, "O": 2}
            {"Al": 2, "O": 3}

    Returns
    -------
    float
        Molar mass in g/mol.
    """
    atomic_weights = {
        "H": 1.00794,
        "He": 4.002602,
        "Li": 6.941,
        "Be": 9.012182,
        "B": 10.811,
        "C": 12.0107,
        "N": 14.0067,
        "O": 15.9994,
        "F": 18.9984032,
        "Ne": 20.1797,
        "Na": 22.98976928,
        "Mg": 24.3050,
        "Al": 26.9815386,
        "Si": 28.0855,
        "P": 30.973762,
        "S": 32.065,
        "Cl": 35.453,
        "Ar": 39.948,
        "K": 39.0983,
        "Ca": 40.078,
        "Sc": 44.955912,
        "Ti": 47.867,
        "V": 50.9415,
        "Cr": 51.9961,
        "Mn": 54.938045,
        "Fe": 55.845,
        "Co": 58.933195,
        "Ni": 58.6934,
        "Cu": 63.546,
        "Zn": 65.38,
        "Ga": 69.723,
        "Ge": 72.64,
        "As": 74.92160,
        "Se": 78.96,
        "Br": 79.904,
        "Kr": 83.798,
        "Ag": 107.8682,
        "Cd": 112.411,
        "In": 114.818,
        "Sn": 118.710,
        "Sb": 121.760,
        "Te": 127.60,
        "I": 126.90447,
        "Ba": 137.327,
        "W": 183.84,
        "Pt": 195.084,
        "Au": 196.966569,
        "Pb": 207.2,
    }

    total = 0.0
    for element, count in composition.items():
        if element not in atomic_weights:
            raise KeyError(f"Atomic weight for element '{element}' is not defined.")
        total += atomic_weights[element] * count
    return total


def atoms_per_formula_unit(composition: dict[str, float]) -> float:
    """
    Return total atoms per formula unit.
    """
    return float(sum(composition.values()))


def gr_to_small_gr(
    r: ArrayLike,
    big_gr: ArrayLike,
    rho0: float,
    r_min_valid: float = 1e-12,
) -> np.ndarray:
    """
    Convert reduced PDF G(r) to pair correlation function g(r).

    Parameters
    ----------
    r : array-like
        Distance array in Å.
    big_gr : array-like
        G(r) values.
    rho0 : float
        Number density in atoms / Å^3.
    r_min_valid : float
        Minimum r to avoid division by zero.

    Returns
    -------
    np.ndarray
        g(r) array.
    """
    r = np.asarray(r, dtype=float)
    big_gr = np.asarray(big_gr, dtype=float)

    if r.shape != big_gr.shape:
        raise ValueError("r and big_gr must have the same shape.")
    if rho0 <= 0:
        raise ValueError("rho0 must be positive.")

    small_gr = np.full_like(big_gr, np.nan, dtype=float)
    mask = r > r_min_valid
    small_gr[mask] = big_gr[mask] / (4.0 * math.pi * r[mask] * rho0) + 1.0
    return small_gr


def load_two_column_data(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a text file containing two numeric columns, ignoring non-numeric lines.

    Parameters
    ----------
    filepath : str or Path
        Input text file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        r, y arrays
    """
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric two-column data found in file: {filepath}")

    data = np.array(rows, dtype=float)
    return data[:, 0], data[:, 1]


def save_two_column_data(
    filepath: Union[str, Path],
    x: ArrayLike,
    y: ArrayLike,
    header: Optional[str] = None,
) -> None:
    """
    Save two-column data.

    Parameters
    ----------
    filepath : str or Path
        Output text file.
    x, y : array-like
        Arrays to save.
    header : str, optional
        Header line.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    data = np.column_stack([x, y])
    np.savetxt(filepath, data, header="" if header is None else header)


def convert_file_gr_to_small_gr(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    rho0: float,
    header: Optional[str] = "r(Angstrom) g(r)",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a two-column G(r) file, convert to g(r), and save.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        r, g(r)
    """
    r, big_gr = load_two_column_data(input_file)
    small_gr = gr_to_small_gr(r, big_gr, rho0=rho0)
    save_two_column_data(output_file, r, small_gr, header=header)
    return r, small_gr
