from abc import ABC, abstractmethod
import numpy as np

class ReactionDiffusionSystem(ABC):
    """
    Abstract base class for 1D reaction-diffusion systems with arbitrary species.
    """

    def __init__(self, species_names, initial_conditions, diffusion_rates, parameters):
        """
        Parameters:
            species_names (list of str): e.g., ['A', 'B']
            initial_conditions (dict): initial concentration arrays
            diffusion_rates (dict): mapping species -> diffusion coefficient
            parameters (dict): model-specific parameters
        """
        self.species_names = species_names
        self.state = initial_conditions
        self.D = diffusion_rates
        self.params = parameters

    @abstractmethod
    def reaction_diffusion_rhs(self, dx):
        """
        Computes the time derivative d[species]/dt for each species.
        Must return a dict mapping species names to 1D arrays.
        """
        pass

    def step(self, dx, dt):
        """
        Performs a single time step using forward Euler.
        """
        dstate_dt = self.reaction_diffusion_rhs(dx)
        for s in self.species_names:
            self.state[s] += dt * dstate_dt[s]

    def get_state(self):
        return self.state

class ActivatorInhibitor(ReactionDiffusionSystem):
    def reaction_diffusion_rhs(self, dx):
        A = self.state['A']
        B = self.state['B']
        D_A = self.D['A']
        D_B = self.D['B']
        p = self.params

        # Compute Laplacians (1D, periodic BC)
        lap_A = (np.roll(A, -1) - 2 * A + np.roll(A, 1)) / dx**2
        lap_B = (np.roll(B, -1) - 2 * B + np.roll(B, 1)) / dx**2

        # Avoid division by zero in A^2 / B
        safe_B = np.where(B <= 1e-8, 1e-8, B)

        # Model equations
        dA_dt = (
            p['s'] * (A**2 / safe_B + p['b_a']) - p['r_a'] * A + D_A * lap_A
        )
        dB_dt = (
            p['s'] * A**2 - p['r_b'] * B + p['b_b'] + D_B * lap_B
        )

        return {'A': dA_dt, 'B': dB_dt}
