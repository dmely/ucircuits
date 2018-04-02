"""
Original implementation of Mely, Linsley, & Serre (2018) [1].

References
----------
[1] Mely, Linsley, & Serre. Psychological Review. (2018).
"""
from collections import deque

import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.autograd as autograd

from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class CircuitParameters:
    """
    Wiring and weights of the circuit, and their symbols in the paper.
    """
    near_size : int = 9                               # `|N|`
    far_size : int = 29                               # `|F|`
    feedforward_strength : float = 4.50               # `xi`
    feedback_strength : float = 1.00                  # `zeta`
    untuned_inhibition_linear : float = 1.00          # `alpha`
    untuned_inhibition_offset : float = 1.00          # `mu`
    tuned_inhibition_linear : float = 3.00            # `beta`
    tuned_inhibition_offset : float = 0.30            # `nu`
    tuned_excitation_strength : float = 1.00          # `gamma`
    broadly_tuned_excitation_strength : float = 1.00  # `delta`
    broadly_tuned_excitation_scale : float = 0.15     # `stigma`
    time_constant_inhibition : float = 6.00           # `eta`
    time_constant_excitation : float = 6.00           # `tau`
    decay_constant_inhibition : float = 0.50          # `epsilon`
    decay_constant_excitation : float = 0.50          # `sigma`


class ContextualCircuit(nn.Module):
    """
    Original implementation of the 'contextual circuit' of Mely, Linsley, &
    Serre (2018).

    The forward pass ingests a tensor of dimensions (N, K, H, W), respectively
    the batch, feature (channel), height and width dimensions, and returns a
    tensor of dimensions:

    * If keeping history: (N, T, K, H, W) where T is the time dimensions
    * Otherwise: (N, K, H, W) just like the input

    Parameters
    ----------
    TBD
    """
    def __init__(self,
                 num_units,
                 parameters=None,
                 num_iters=100,
                 step_size=3,
                 keeptime=True):

        super().__init__()

        self.parameters = parameters or CircuitParameters()
        self.num_iters = num_iters
        self.step_size = step_size
        self.keeptime = keeptime

        # Circuit recurrent inputs within and outside the receptive field
        scale = self.parameters.broadly_tuned_excitation_scale
        inner = self.parameters.near_size
        outer = self.parameters.far_size

        self.untuned_CRF = UntunedCRF(num_units)
        self.broadly_tuned_CRF = BroadlyTunedCRF(num_units, scale)
        self.tuned_near_ECRF = TunedNearECRF(num_units, inner)
        self.tuned_far_ECRF = TunedFarECRF(num_units, inner, outer)

    def forward(self, inbound):
        """
        Forward Euler numerical integration.
        """
        X = inbound.clone()
        Y = inbound.clone()

        # Keep history of activations?
        if self.keeptime:
            N, K, H, W = inbound.shape
            Z = torch.zeros(
                size=(N, self.num_iters, K, H, W),
                dtype=inbound.dtype)

        # Effective step sizes and decay constants
        dt_x = self.step_size / self.parameters.time_constant_inhibition
        dt_y = self.step_size / self.parameters.time_constant_excitation
        k_x = self.parameters.decay_constant_inhibition ** 2 * dt_x
        k_y = self.parameters.decay_constant_excitation ** 2 * dt_y

        for t in tqdm(range(self.num_iters)):

            # Compute various pools
            U = self.untuned_CRF(Y)
            T = self.tuned_far_ECRF(Y)
            P = self.tuned_near_ECRF(X)
            Q = self.broadly_tuned_CRF(X)

            # "Shunting" multipliers
            untuned_inhibition_factor = (
                self.parameters.untuned_inhibition_linear * X +
                self.parameters.untuned_inhibition_offset)

            tuned_inhibition_factor = (
                self.parameters.tuned_inhibition_linear * X +
                self.parameters.tuned_inhibition_offset)

            # Left-hand side of the ODE for X[t]
            total_contributions_X = nn.ReLU()(
                + self.parameters.feedforward_strength * inbound
                - untuned_inhibition_factor * U
                - tuned_inhibition_factor * T)

            # Left-hand side of the ODE for Y[t]
            total_contributions_Y = nn.ReLU()(
                + self.parameters.feedback_strength * X
                + self.parameters.tuned_excitation_strength * P
                + self.parameters.broadly_tuned_excitation_strength * Q)

            # Forward Euler numerical integration
            X = (1 - k_x) * X + total_contributions_X * dt_x
            Y = (1 - k_y) * Y + total_contributions_Y * dt_y
            
            if self.keeptime:
                Z[:, t] = Y
        
        if not self.keeptime:
            Z = Y

        return Z


class UntunedCRF(nn.Conv2d):
    """
    Untuned classical receptive field
    """
    def __init__(self, K):
        """
        Parameters
        ----------
        K : int
            Number of feature channels
        """
        super().__init__(
            in_channels=K,
            out_channels=K,
            kernel_size=(K, K, 1, 1))

        kernel = torch.ones((K, K, 1, 1)) / K
        self.weight = nn.Parameter(kernel)


class BroadlyTunedCRF(nn.Conv2d):
    """
    Broadly-tuned classical receptive field
    """
    def __init__(self, K, scale):
        """
        Parameters
        ----------
        K : int
            Number of feature channels
        scale : float
            Standard deviation of the gaussian tuning.
        """
        super().__init__(
            in_channels=K,
            out_channels=K,
            kernel_size=(K, K, 1, 1))

        kernel = self._get_kernel(K, scale)
        self.weight = nn.Parameter(kernel)

    @staticmethod
    def _get_kernel(K, scale):
        """
        Compute convolutional kernel.
        """
        kernel = np.zeros((K, K, 1, 1), dtype=np.float32)
        weights = stats.norm.pdf(
            x=np.linspace(0, 1, K),
            loc=np.linspace(0, 1, K)[K // 2],
            scale=scale)

        for i in range(K):
            kernel[i] = np.roll(weights, K // 2 + i).reshape(-1, 1, 1)

        return torch.from_numpy(kernel)


class TunedNearECRF(nn.Conv2d):
    """
    Tuned near extra-classical receptive field
    """
    def __init__(self, K, size):
        super().__init__(
            in_channels=K,
            out_channels=K,
            kernel_size=(K, K, size, size),
            padding=(size - 1) // 2)

        # Zero center of kernel to get annular kernel
        kernel = np.zeros((K, K, size, size), dtype=np.float32)
        kernel[range(K), range(K)] = 1
        kernel[range(K), range(K), size // 2, size // 2] = 0

        # Convert to kernel to a mean filter over its support
        normalizer = kernel.sum(axis=(-2, -1), keepdims=True)
        normalizer[normalizer == 0] = 1
        kernel /= normalizer
        kernel = torch.from_numpy(kernel)

        self.weight = nn.Parameter(kernel)


class TunedFarECRF(nn.Conv2d):
    """
    Tuned far extra-classical receptive field
    """
    def __init__(self, K, inner, outer):
        super().__init__(
            in_channels=K,
            out_channels=K,
            kernel_size=(K, K, outer, outer),
            padding=(outer - 1) // 2)

        # Zero center of kernel to get annular kernel
        u = outer // 2 - inner // 2
        v = outer // 2 + inner // 2 + 1

        kernel = np.zeros((K, K, outer, outer), dtype=np.float32)
        kernel[range(K), range(K)] = 1
        kernel[range(K), range(K), u:v, u:v] = 0

        # Convert to kernel to a mean filter over its support
        normalizer = kernel.sum(axis=(-2, -1), keepdims=True)
        normalizer[normalizer == 0] = 1
        kernel /= normalizer
        kernel = torch.from_numpy(kernel)

        self.weight = nn.Parameter(kernel)
