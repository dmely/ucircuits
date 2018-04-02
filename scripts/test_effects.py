import click
import numpy as np
import torch

from ucircuits.motifs.ring_network_lattice import RingNetworkLattice
from ucircuits.populations import BellShapedPopulationEncoder
from ucircuits.populations import BellShapedPopulationDecoder
import ucircuits.plotting as plotting


def tilt_effect(size=29, near=9, far=29, num_units=30, num_angles=10):
    middle = size // 2
    center = np.s_[
        middle - near // 2:middle + near // 2,
        middle - near // 2:middle + near // 2]
    surround = np.s_[
        middle - far // 2:middle + far // 2,
        middle - far // 2:middle + far // 2]

    t_surround = 90
    t_center = t_surround + np.linspace(0, 90, num_angles)

    S = np.zeros((size, size), dtype=np.float32) + np.nan
    S[surround] = t_surround * np.pi / 180

    circuit = RingNetworkLattice(size, size, num_units, keeptime=False)
    X = populations.OrientationTunedPopulation(num_units, bandwidth=1.00)
    dt = np.zeros((num_angles,), dtype=np.float32)

    for i, t in enumerate(t_center):
        S[center] = t * np.pi / 180
        X.encode(S)
        var = torch.from_numpy(X._data[np.newaxis])
        var = autograd.Variable(var, volatile=True)
        T = circuit.forward(var)
        Y = X.copy_from(T[-1].data.numpy().squeeze(axis=-4))
        dt[i] = Y.decode(middle, middle) - t * np.pi / 180

    return dt


if __name__ == '__main__':
    tilt_effect()
