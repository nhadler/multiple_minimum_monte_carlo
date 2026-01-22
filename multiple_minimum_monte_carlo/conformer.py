"""Module for managing conformer generation and molecular structure handling.

This module provides the Conformer class for creating and managing molecular conformers
from SMILES strings or XYZ coordinate files using RDKit and ASE.
"""

from typing import Optional, List
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from multiple_minimum_monte_carlo import cheminformatics


class Conformer:
    """Represents a molecular conformer with associated metadata and 3D coordinates.

    This class handles the creation and management of molecular conformers from either
    SMILES strings or XYZ coordinate files. It supports atom mapping, charge and spin
    multiplicity specification, and atom constraints for geometry optimization.

    Attributes:
        smiles (str): The SMILES string representing the molecule.
        mapped (bool): Whether the SMILES string is atom-mapped.
        input_xyz (str): Path to an XYZ file containing input coordinates.
        charge (int): The formal charge of the molecule.
        spin_multiplicity (int): The spin multiplicity of the molecule (2S+1).
        constrained_atoms (List[int]): List of atom indices to be constrained.
        mol (Chem.Mol): RDKit molecule object.
        atoms (ase.Atoms): ASE Atoms object with 3D coordinates.
        bonded_atoms (List[Tuple[int, int]]): List of bonded atom pairs.
    """

    def __init__(
        self,
        smiles: Optional[str] = None,
        mapped: Optional[bool] = False,
        input_xyz: Optional[str] = None,
        charge: Optional[int] = None,
        spin_multiplicity: Optional[int] = 1,
        constrained_atoms: Optional[List[int]] = None,
    ) -> None:
        """Initialize a conformer object from a SMILES string or XYZ file.

        Creates a molecular conformer with optional atom mapping, input coordinates,
        and constrained atoms. At least one of smiles or input_xyz must be provided.

        Args:
            smiles: The SMILES string representing the molecule. If atom-mapped,
                set mapped=True.
            mapped: Whether the SMILES string is atom-mapped. If True, uses a
                custom molecule creation function that preserves atom ordering;
                otherwise, hydrogens are added to the molecule.
            input_xyz: Path to an XYZ file containing input coordinates. If None
                and smiles is provided, a conformer is generated using ETKDG.
            charge: The formal charge of the molecule. Required if only input_xyz
                is provided without smiles. If None and smiles is provided, the
                charge is inferred from the molecule.
            spin_multiplicity: The spin multiplicity of the molecule (2S+1), where
                S is the total spin quantum number. Default is 1 (singlet).
                Examples: 1=singlet, 2=doublet, 3=triplet.
            constrained_atoms: List of atom indices (0-based) to be constrained
                during conformer generation or optimization.

        Raises:
            ValueError: If neither smiles nor input_xyz is provided.
            ValueError: If only input_xyz is provided without specifying charge.
        """
        self.smiles = smiles
        self.mapped = mapped
        self.input_xyz = input_xyz
        self.charge = charge
        self.spin_multiplicity = spin_multiplicity
        self.constrained_atoms = constrained_atoms

        # Check whether atom rearrangement is necessary
        if self.smiles is not None:
            if self.mapped:
                self.mol = cheminformatics.make_mol(self.smiles)
            else:
                self.mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles))

        # Check whether conformer generation/mapping is necessary
        if self.input_xyz is None and self.smiles is None:
            raise ValueError("Conformer needs either smiles or input_xyz!")
        elif self.input_xyz is None:
            self.generate_conformer()
        elif self.smiles is None:
            if self.charge is None:
                raise ValueError(
                    "Charge must be specified if no SMILES string provided!"
                )
            self.mol_from_xyz()
            self.atoms = read(input_xyz)
        else:
            self.atoms = read(input_xyz)
            self.add_xyz_to_mol()

        if self.charge is None:
            self.charge = Chem.GetFormalCharge(self.mol)
        self.bonded_atoms = cheminformatics.get_bonded_atoms(self.mol)
        self.atoms.info["charge"] = self.charge
        self.atoms.info["spin_multiplicity"] = self.spin_multiplicity

    def generate_conformer(self) -> None:
        """Generate a 3D conformer using RDKit's ETKDG algorithm.

        Uses the Experimental-Torsion Knowledge Distance Geometry (ETKDG)
        method to generate initial coordinates, followed by UFF optimization
        to refine the geometry.

        Updates:
            self.atoms: ASE Atoms object with generated 3D coordinates.
        """
        AllChem.EmbedMolecule(self.mol)
        AllChem.UFFOptimizeMolecule(self.mol)
        self.atoms = cheminformatics.mol_to_ase_atoms(self.mol)

    def mol_from_xyz(self) -> None:
        """Generate an RDKit molecule from an XYZ coordinate file.

        Reads coordinates from the XYZ file, determines bond connectivity
        based on distances and the specified charge, and creates the SMILES
        representation.

        Updates:
            self.mol: RDKit molecule object with determined bonds.
            self.smiles: SMILES string generated from the molecule.
        """
        raw_mol = Chem.MolFromXYZFile(self.input_xyz)
        mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineBonds(mol, charge=self.charge)
        self.mol = mol
        self.smiles = Chem.MolToSmiles(mol)

    def add_xyz_to_mol(self) -> None:
        """Add 3D coordinates from an XYZ file to the existing RDKit molecule.

        Reads coordinates from the XYZ file and adds them as a conformer to
        the molecule object that was created from the SMILES string.

        Updates:
            self.mol: RDKit molecule object with added conformer.
        """
        raw_mol = Chem.MolFromXYZFile(self.input_xyz)
        conf = raw_mol.GetConformer()
        self.mol.AddConformer(conf, assignId=True)
