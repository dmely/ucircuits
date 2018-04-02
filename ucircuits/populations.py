"""
Utilities to compute neural population responses with idealized tuning.
"""
import numpy as np
import torch
import torch.nn as nn


class BellShapedPopulationEncoder(nn.Module):
    """
    Represents population coding of some input with bell-shaped tuning
    curves.
    
    The inbound tensor is two- (H, W) or three-dimensional (N, H, W).
    The resulting tensor is always four-dimensional (N, K, H, W).

    Parameters
    ----------
    TBD
    """
    def __init__(self, num_units, bandwidth):
        super().__init__()
        weights = make_default_weights(num_units)
        weights = 2 * weights - np.pi
        weights = weights.reshape(1, num_units, 1, 1)
        self._weights = weights
        self._bandwidth = bandwidth
    
    def forward(self, inbound):
        if isinstance(inbound, np.ndarray):
            inbound = torch.from_numpy(inbound)

        if inbound.ndim <= 1:
            raise ValueError("Invalid number of input dimensions")
        elif inbound.ndim == 2:
            batch, height, width = 1, *inbound.shape
        elif inbound.ndim == 3:
            batch, height, width = inbound.shape
        elif inbound.ndim >= 4:
            raise ValueError("Invalid number of input dimensions")

        inbound = inbound.reshape(batch, 1, height, width)
        
        # Map the stimulus (0, Pi) -> (-Pi, Pi) to match Von Mises support
        inbound = 2 * inbound - np.pi

        # Density of the Von Mises circular distribution
        kappa = torch.Tensor([1 / self._bandwidth ** 2])
        outbound = torch.exp(kappa * torch.cos(inbound - self._weights))
        outbound /= (2 * np.pi * torch.i0(kappa))

        # Mask out any missing data
        outbound[torch.isnan(outbound)] = 0

        return outbound


class BellShapedPopulationDecoder(nn.Module):
    """
    Neural observer model that decodes stimulus values given a bell-shaped
    population code.
    
    Parameters
    ----------
    TBD
    """
    def __init__(self, num_units, wrap_around=True):
        super().__init__()
        self._weights = make_default_weights(num_units)
        self._wrap_around = wrap_around

    def forward(self, inbound):
        if self._wrap_around:
            normalized_weights = normalize_weights(self._weights)
            weights_sin = torch.sin(normalized_weights)
            weights_cos = torch.cos(normalized_weights)
            outbound = torch.atan2(
                input=torch.tensordot(weights_sin, inbound, dims=([0], [-3])),
                other=torch.tensordot(weights_cos, inbound, dims=([0], [-3])))
            outbound = (outbound + np.pi) / 2
        else:
            norm = inbound.sum(dim=-3, keepdim=False)
            weights = self._weights / norm
            outbound = torch.tensordot(weights, inbound, dims=([0], [-3]))
        
        return outbound


def normalize_weights(weights):
    """
    Normalizes weights to the -Pi to Pi range.
    """
    min_weight = weights.min()
    max_weight = weights.max()
    if min_weight == max_weight:
        raise ValueError("This should never happen!")
        
    weights = (weights - min_weight) / (max_weight - min_weight)
    weights = weights * 2 * np.pi - np.pi

    return weights


def make_default_weights(num_weights):
    """
    Generates equally-spaced weights between 0 (included) and Pi (excluded).
    """
    return torch.linspace(
        start=0,
        end=(num_weights - 1) * np.pi / num_weights,
        steps=num_weights)