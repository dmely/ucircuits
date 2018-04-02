#!/usr/bin/env python
 
import click
import numpy as np
import torch
import torch.autograd as autograd

from ucircuits.motifs.contextual import ContextualCircuit
from ucircuits.motifs.contextual import CircuitParameters
from ucircuits.populations import BellShapedPopulationEncoder
from ucircuits.populations import BellShapedPopulationDecoder
import ucircuits.plotting as plotting


@click.command()
@click.option("--size", default=51, help="Stimulus size")
@click.option("--near", default=8, help="'Near' kernel size")
@click.option("--far", default=32, help="'Far' kernel size")
@click.option("--bandwidth", default=0.40, help="Tuning curve bandwidth")
@click.option("--num-units", "num_units", default=25,
    help="Number of units in each column")
def main(size, near, far, num_units, bandwidth):
    # Some parameters
    c = size // 2
    n = near
    f = far

    # Stimulus
    stimulus = np.zeros((size, size), dtype=np.float32)
    stimulus[:] = np.nan
    stimulus[c-f//2:c+f//2, c-f//2:c+f//2] = 90
    stimulus[c-n//2:c+n//2, c-n//2:c+n//2] = 90 + 15
    stimulus *= np.pi / 180

    # Build encoder-circuit-decoder network
    L1 = BellShapedPopulationEncoder(num_units=num_units, bandwidth=bandwidth)
    L2 = ContextualCircuit(num_units=num_units, keeptime=True)
    L3 = BellShapedPopulationDecoder(num_units=num_units)

    with torch.no_grad():
        X0 = torch.Tensor(stimulus)
        X1 = L1(X0)
        X2 = L2(X1)
        decoded = L3(X2)

    plotter = plotting.PopulationBarPlot(X2)
    plotter.bar_plot([c, c-f//4], [c, c-f//4])

    # Now look at 2D orientation map
    # TODO(david)


if __name__ == "__main__":
    click.secho("Contextual circuit test program", fg="blue")
    main()