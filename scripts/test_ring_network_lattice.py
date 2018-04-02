#!/usr/bin/env python

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from ucircuits.motifs.ring_network_lattice import RingNetworkLattice
from ucircuits.populations import BellShapedPopulationEncoder
from ucircuits.populations import BellShapedPopulationDecoder
import ucircuits.plotting as plotting


@click.command()
@click.option("--size", default=51, help="Stimulus size")
@click.option("--near", default=3, help="'Near' kernel size")
@click.option("--far", default=29, help="'Far' kernel size")
@click.option("--num-units", "num_units", default=30,
    help="Number of units in each column")
@click.option("--noise/--no-noise", default=False, help="Add noise?")
@click.option("--vnear", default=105, help="'Near' stimulus value")
@click.option("--vfar", default=90, help="'Far' stimulus value")
@click.option("--bandwidth", default=1.00, help="Tuning curve bandwidth")
@click.option("--num-iters", "num_iters", default=100,
    help="Number of units in each column")
def main(size, near, far, num_units, noise, vnear, vfar, bandwidth, num_iters):
    middle = size // 2
    electrode_x = (middle, middle - far // 4)
    electrode_y = (middle, middle - far // 4)

    center = np.s_[
        middle - near // 2:middle + near // 2,
        middle - near // 2:middle + near // 2]
    surround = np.s_[
        middle - far // 2:middle + far // 2,
        middle - far // 2:middle + far // 2]

    # Stimulus
    stimulus = np.zeros((size, size), dtype=np.float32) + np.nan
    stimulus[surround] = vfar * np.pi / 180
    stimulus[center] = vnear * np.pi / 180

    # Build encoder-circuit-decoder network
    L1 = BellShapedPopulationEncoder(num_units=num_units, bandwidth=bandwidth)
    L2 = RingNetworkLattice(
        num_units=num_units,
        keeptime=True,
        step_size=2,
        num_iters=num_iters)
    L3 = BellShapedPopulationDecoder(num_units=num_units)

    with torch.no_grad():
        X0 = torch.Tensor(stimulus)
        X1 = L1(X0)
        if noise:
            X1 += torch.randn(X1.shape) * 0.25
        X2 = L2(X1)

    # Animated population bar plot at specific location
    plotter = plotting.PopulationPlot(X2)
    plotter.bar_plot(electrode_x, electrode_y)

    # Animated decoded population map over visual field
    plotting.plot_stimulus(stimulus)
    plotter.map_plot()

    return


if __name__ == '__main__':
    click.secho("Ring network lattice test program", fg="blue")
    main()
