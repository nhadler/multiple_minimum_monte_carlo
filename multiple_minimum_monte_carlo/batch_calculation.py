"""Module for running batched geometry optimizations.

This module provides classes for performing batched geometry optimizations
on multiple molecular structures simultaneously, leveraging GPU acceleration
through TorchSim for improved performance.
"""

from typing import Optional, List, Tuple, Any
import numpy as np
import ase
from ase.constraints import FixAtoms
import torch

EV_TO_KCAL = 23.0605
"""float: Conversion factor from electron volts to kilocalories per mole."""


class BatchCalculation:
    """Abstract base class for batched molecular calculations.

    This class defines the interface for performing batched energy calculations
    and geometry optimizations on multiple molecular structures simultaneously.
    """

    def __init__(self):
        """Initialize the batch calculation object."""
        pass

    def run(
        self,
        atoms_list: List[ase.Atoms],
        constrained_atoms_list: Optional[List[List[int]]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Run batched geometry optimizations on the given list of atoms.

        Args:
            atoms_list: List of ASE Atoms objects representing molecules to optimize.
            constrained_atoms_list: List of lists of atom indices to constrain for
                each molecule during optimization.

        Returns:
            Tuple containing the optimized positions (np.ndarray with shape
            (n_molecules, n_atoms, 3)) and energies (list of floats in kcal/mol).
        """
        pass

    def energy(self, atoms: ase.Atoms) -> float:
        """Calculate energies for a list of atoms objects.

        Args:
            atoms: List of ASE Atoms objects representing molecules.

        Returns:
            List of energies in kcal/mol.
        """
        pass


class TorchSimCalculation(BatchCalculation):
    """Batched geometry optimization using TorchSim.

    This class leverages TorchSim for GPU-accelerated batched geometry optimizations
    using machine learning interatomic potentials. It enables efficient processing
    of multiple conformers in parallel on GPU hardware.

    Attributes:
        model: TorchSim ModelInterface for computing energies and forces.
        optimizer: TorchSim Optimizer for performing optimizations.
        device: Device to run calculations on ('cpu' or 'cuda').
        dtype: PyTorch data type for calculations.
        max_cycles: Maximum number of optimization steps.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        device: Optional[str] = "cpu",
        dtype: Optional[Any] = torch.float32,
        max_cycles: Optional[int] = 1000,
    ) -> None:
        """Initialize the TorchSim batch calculation.

        Args:
            model: TorchSim ModelInterface instance (e.g., MACE, NequIP) for
                computing energies and forces.
            optimizer: TorchSim Optimizer instance for performing geometry
                optimizations.
            device: Device to run calculations on. Options are 'cpu' or 'cuda'
                for GPU acceleration. Default is 'cpu'.
            dtype: PyTorch data type for calculations. Default is torch.float32.
            max_cycles: Maximum number of optimization steps. Default is 1000.

        Raises:
            ImportError: If TorchSim is not installed.
            TypeError: If model is not a ModelInterface instance.
            TypeError: If optimizer is not an Optimizer instance.
        """
        try:
            from torch_sim.models.interface import ModelInterface
            from torch_sim.optimizers import Optimizer
        except ImportError:
            raise ImportError("TorchSim is not installed")
        self.model = model
        if not isinstance(self.model, ModelInterface):
            raise TypeError("model must be an instance of ModelInterface")
        self.optimizer = optimizer
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError("optimizer must be an instance of Optimizer")
        self.device = device
        self.dtype = dtype
        self.max_cycles = max_cycles

    def run(
        self,
        atoms_list: List[ase.Atoms],
        constrained_atoms_list: Optional[List[List[int]]] = None,
    ):
        """Run batched geometry optimizations using TorchSim.

        Optimizes all structures in atoms_list simultaneously using the configured
        model and optimizer. Supports optional atom constraints for each structure.

        Args:
            atoms_list: List of ASE Atoms objects representing molecules to optimize.
            constrained_atoms_list: Optional list of lists of atom indices to constrain
                for each molecule. Must have the same length as atoms_list if provided.

        Returns:
            Tuple of (positions, energies) where positions is a numpy array of shape
            (n_molecules, n_atoms, 3) containing optimized coordinates, and energies
            is a list of floats representing final energies in kcal/mol.

        Raises:
            ImportError: If TorchSim is not installed.
            ValueError: If atoms_list and constrained_atoms_list have different lengths.
        """
        try:
            import torch_sim as ts
        except ImportError:
            raise ImportError("TorchSim is not installed")
        if constrained_atoms_list is not None:
            if len(atoms_list) != len(constrained_atoms_list):
                raise ValueError(
                    "Length of atoms_list and constrained_atoms_list must be the same"
                )
            for atoms, constrained_atoms in zip(atoms_list, constrained_atoms_list):
                if len(constrained_atoms) > 0:
                    atoms.set_constraint(FixAtoms(constrained_atoms))
        final_state = ts.optimize(
            system=atoms_list, model=self.model, optimizer=self.optimizer
        )
        positions = final_state.positions.detach().numpy().astype(np.float64)
        positions = positions.reshape(len(atoms_list), -1, 3)
        energies = list(
            final_state.energy.detach().numpy().astype(np.float64) * EV_TO_KCAL
        )
        return positions, energies

    def energy(self, atoms_list: List[ase.Atoms]) -> List[float]:
        """Calculate energies for a list of structures using TorchSim.

        Computes energies for all structures in atoms_list simultaneously using
        the configured model.

        Args:
            atoms_list: List of ASE Atoms objects representing molecules.

        Returns:
            List of energies in kcal/mol, one for each structure in atoms_list.

        Raises:
            ImportError: If TorchSim is not installed.
        """
        try:
            import torch_sim as ts
        except ImportError:
            raise ImportError("TorchSim is not installed")
        state = ts.io.atoms_to_state(atoms_list, device=self.device, dtype=self.dtype)
        result = self.model(state)
        energies = list(
            result["energy"].detach().numpy().astype(np.float64) * EV_TO_KCAL
        )
        return energies
