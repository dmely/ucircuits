"""
A simpler implementation of the contextual circuit [1] based on ring
attractor networks [2].

References
----------
[1] Mely, Linsley, & Serre. Psychological Review. (2018).
[2] Ben-Yishai, Lev Bar-Or, & Sompolinsky. PNAS. (1995).
"""
import logging
from collections import deque

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.autograd as autograd

from dataclasses import dataclass
from tqdm import tqdm

LOG = logging.getLogger(__name__)


@dataclass
class CircuitParameters:
    """
    Wiring and weights of the circuit, and their symbols in the paper.
    """
    time_constant : float = 5
    decay_constant : float = 1
    near_size : int = 9
    far_size : int = 29
    near_strength : float = 0.40
    far_strength : float = 0.40
    ring_bandwidth : float = 0.50
    ring_min : float = -0.50
    ring_max : float = 0.10


class RingNetworkLattice(nn.Module):
    """
    Improved, ring attractor network-based implementation of the 'contextual
    circuit' of Mely, Linsley, & Serre (2018).

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
                 step_size=1,
                 keeptime=True):

        super().__init__()

        LOG.debug("Setting up circuit.")
        self.parameters = parameters or CircuitParameters()

        # ODE integration parameters
        self.num_iters = num_iters
        self.step_size = step_size
        self.keeptime = keeptime

        # Circuit recurrent inputs within and outside the receptive field
        self.feature_connections = FeatureConnections(
            kernel_min=self.parameters.ring_min,
            kernel_max=self.parameters.ring_max,
            bandwidth=self.parameters.ring_bandwidth,
            K=num_units)

        self.spatial_connections = SpatialConnections(
            near_sum=self.parameters.near_strength,
            far_sum=self.parameters.far_strength,
            near_size=self.parameters.near_size,
            far_size=self.parameters.far_size,
            K=num_units)

    def forward(self, inbound):
        """
        Forward Euler numerical integration.
        """
        X = torch.zeros_like(inbound)

        # Keep history of activations?
        if self.keeptime:
            N, K, H, W = inbound.shape
            X_t = torch.zeros(
                size=(N, self.num_iters, K, H, W),
                dtype=inbound.dtype)

        # Integration constants
        h = self.step_size
        r = self.parameters.time_constant
        k = self.parameters.decay_constant

        LOG.debug("Integrating circuit ODE.")

        for t in tqdm(range(self.num_iters)):

            # Sum contributions to each unit
            S = 0.5 * inbound
            S += self.feature_connections(X)
            S += self.spatial_connections(X)

            # Update equation corresponding to Euler(step size = h)
            X = (1 - h * k ** 2 / r) * X +  h / r * nn.ReLU()(S)

            if self.keeptime:
                X_t[:, t] = X

        return X_t if self.keeptime else X


class FeatureConnections(nn.Conv2d):
    """
    Kernel with both excitatory and inhibitory weights responsible for
    bell-shaped tuning in the steady-state of the ring attractor of [2].

    Parameters
    ----------
    kernel_min : float
        Minimum value of the kernel across the tuning domain.
    kernel_max : float
        Maximum value of the kernel across the tuning domain.
    bandwidth : float
        Bandwidth of the kernel in standard deviations.

    References
    ----------
    See module docstrings.
    """
    def __init__(self, K,
                 kernel_min,
                 kernel_max,
                 bandwidth):

        if kernel_min >= 0:
            LOG.warn(
                "Minimum value is positive, thus "
                "this kernel has no inhibitory part.")

        if kernel_max <= 0:
            LOG.warn(
                "Maximum value is negative, thus "
                "this kernel has no excitatory part.")

        self.kernel_min = kernel_min
        self.kernel_max = kernel_max
        self.bandwidth = bandwidth

        super().__init__(
            in_channels=K,
            out_channels=K,
            kernel_size=(K, K, 1, 1))

        # Shift kernel to center it for each unit's tuning center
        weights = [self._kernel(K, i)[:, None, None] for i in range(K)]
        weights = np.array(weights, dtype=np.float32)
        weights = torch.from_numpy(weights)

        self.weight = nn.Parameter(weights)

    def _kernel(self, K, i):
        """
        Generate a shifted version of a circular bell-shaped kernel.
        """
        X = np.linspace(0, 1, K)  # domain of `stats.norm`

        # Normalized weight function
        weights = stats.norm.pdf(x=X, loc=X[K // 2], scale=self.bandwidth)
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        weights *= (self.kernel_max - self.kernel_min)
        weights += self.kernel_min

        # Shift kernel peak to i-th center
        weights = np.roll(weights, K // 2 + i)

        return weights


class SpatialConnections(nn.Conv2d):
    """
    Opponent surrounds from [1]: excitation in the near surround and
    inhibition in the far surround.

    Parameters
    ----------
    near_sum : float
        Total excitatory contributions from the near surround.
    near_size : int
        Size of the near excitatory surround.
    far_sum : float
        Total inhibitory contributions from the far surround.
    far_size : int
        Size of the far inhibitory surround.

    References
    ----------
    See module docstrings.
    """
    def __init__(self, K,
                 near_sum,
                 near_size,
                 far_sum,
                 far_size):

        self.near_sum = near_sum
        self.near_size = near_size
        self.far_sum = far_sum
        self.far_size = far_size

        super().__init__(
            in_channels=K,
            out_channels=K,
            kernel_size=(K, K, far_size, far_size),
            padding=(far_size - 1) // 2)

        weights = np.zeros((K, K, far_size, far_size), dtype=np.float32)
        weights[range(K), range(K)] = self._kernel(K)
        weights = torch.from_numpy(weights)

        self.weight = nn.Parameter(weights)

    def _kernel(self, K):
        kernel = np.zeros((self.far_size, self.far_size), dtype=np.float32)
        inhibition = - self.far_sum / (self.far_size ** 2 - self.near_size ** 2)
        excitation = + self.near_sum / self.near_size ** 2

        # Extent of the near surround
        u = self.far_size // 2 - self.near_size // 2
        v = self.far_size // 2 + self.near_size // 2 + 1

        kernel[:] = inhibition
        kernel[u:v, u:v] = excitation

        return kernel
