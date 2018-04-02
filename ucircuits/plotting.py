"""
Plotting library for micro-circuit responses.
"""
import functools
import logging
import re

import numpy as np
from scipy import stats
from skimage import color
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import BlendedGenericTransform
from dataclasses import dataclass

LOG = logging.getLogger(__name__)
mpl.rcParams['toolbar'] = 'None'
sns.set()

TICKLABELS_0_PI = [
    r'$0$',
    r'$\frac{\pi}{8}$',
    r'$\frac{\pi}{4}$',
    r'$\frac{3\pi}{8}$',
    r'$\frac{\pi}{2}$',
    r'$\frac{5\pi}{8}$',
    r'$\frac{3\pi}{4}$',
    r'$\frac{7\pi}{8}$',
    r'$\pi$']


@dataclass
class ColorDefaults:
    bar : str = "#e23f12"
    bar_alpha : float = 0.50
    bar_spacing : float = 0.60
    marker : str = "#252525"


def enforce_visual_style(function):
    """
    Ensures that all plots follow a certain visual consistency.
    """
    @functools.wraps(function)
    def function_(*args, **kwargs):
        with sns.axes_style(style="ticks", rc=dict(color_codes=True)):
            plot_results = function(*args, **kwargs)
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            return plot_results
    return function_


class PopulationPlot:
    """
    Convenience class for plotting population recordings.

    Parameters
    ----------
    population : torch.Tensor
        Shaped (T, K, H, W) or optionally (K, H, W), respectively time,
        feature (channel), height and width dimensions. Note the
        absence of a batch (N) dimension.
    centers : [float] or None, optional
        Values that each feature index is tuned to. If not provided,
        assume equally-spaced weights.
    """
    def __init__(self, population, centers=None):
        if population.ndim == 5 and population.shape[0] == 1:
            LOG.warn(("Squeezing batch dimension of input (N=1, T, K, H, W) "
                      "-> (T, K, H, W)"))
            self._data = np.array(population).squeeze(axis=0)
        elif population.ndim not in (3, 4):
            raise ValueError("Invalid dimensions for input!")
        else:
            self._data = np.array(population)

        if centers is None:
            self._tuning_centers = np.linspace(
                0, np.pi, population.shape[-3], endpoint=False)
        else:
            centers = np.array(centers)
            if centers.ndim != 1 or len(centers) != population.shape[-3]:
                raise ValueError("Invalid dimensions for input!")
            self._tuning_centers = centers

    def bar_plot(self, x, y, t=None, ax=None):
        if self._data.ndim == 3:
            if t is not None:
                LOG.warn(("Ignoring argument `t`, the tensor for this plot "
                          "does not have a time dimension."))

            return plot_population_at(
                population=self._data,
                centers=self._tuning_centers,
                x=x, y=y, ax=ax)

        if self._data.ndim == 4:
            if t is None:
                return animate_population_at(
                    populations=self._data,
                    centers=self._tuning_centers,
                    x=x, y=y, save=None, ax=ax)

            return plot_population_at(
                population=self._data[t],
                centers=self._tuning_centers,
                x=x, y=y, ax=ax)

    def map_plot(self, t=None, ax=None):
        if self._data.ndim == 3:
            if t is not None:
                LOG.warn(("Ignoring argument `t`, the tensor for this plot "
                          "does not have a time dimension."))
            
            return decode_and_plot_population_map(
                population=self._data,
                centers=self._tuning_centers,
                ax=ax)

        if self._data.ndim == 4:
            if t is None:
                return decode_and_animate_population_map(
                    populations=self._data,
                    centers=self._tuning_centers,
                    ax=ax)

            return decode_and_plot_population_map(
                population=self._data[t],
                centers=self._tuning_centers,
                ax=ax)


@enforce_visual_style
def plot_population_at(population, centers, x, y, ax=None, return_patches=False):
    if ax is None:
        _, ax = plt.subplots()

    data = np.array(population)
    spacing = (centers[1] - centers[0]) * ColorDefaults.bar_spacing
    label = as_plot_title(population)

    try:
        points = zip(x, y)
    except TypeError:
        points = [[x, y]]
        colors = sns.husl_palette(1)
    else:
        colors = sns.husl_palette(len(x))

    # Show response histogram
    ax.set_title("Population response: {}".format(label))
    ax.set_xlabel("Tuning center")
    ax.set_ylabel("Activity")
    ax.set_xlim((np.min(centers), np.max(centers)))
    lines = []
    bars = []

    for (x, y), c in zip(points, colors):
        patches = ax.bar(centers, data[:, x, y],
            width=spacing,
            color=c,
            alpha=ColorDefaults.bar_alpha,
            label="{x}, {y}".format(x=x, y=y))

        ax.legend()

        # Show decoded value on x-axis
        z = circular_decoder(population, centers, x, y)
        line, = ax.plot([z], [0],
            color=c,
            marker=11,
            markersize=11,
            clip_on=False,
            transform=BlendedGenericTransform(ax.transData, ax.transAxes))

        lines.append(line)
        bars.append(patches)

    ax.set_xticks(np.linspace(0, np.pi, 8))
    ax.set_xticklabels(TICKLABELS_0_PI)

    if return_patches:
        return ax, lines, bars
    else:
        return ax


@enforce_visual_style
def animate_population_at(populations,
                          centers,
                          x,
                          y,
                          save=None,
                          ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    num_timesteps = len(populations)

    try:
        points = list(zip(x, y))
    except TypeError:
        points = [[x, y]]

    # Extent of y-axis depends on maximum displayed value at steady-state
    max_height = max(
        populations[-1][:, xy[0], xy[1]].max(axis=0)
        for xy in points)
    ax.set_ylim((0, max_height))

    # Draw initial state at time = 0
    _, lines, bars_nested = plot_population_at(
        population=populations[0],
        centers=centers,
        return_patches=True,
        x=x, y=y, ax=ax)

    # Flatten list of bar containers into list of bars
    electrode_ids = []
    feature_ids = []
    bars = []

    for e, bar_container in enumerate(bars_nested):
        for f, bar in enumerate(bar_container):
            electrode_ids.append(e)
            feature_ids.append(f)
            bars.append(bar)

    # Using a tuple instead of a list is critical here
    artists = tuple(bars + lines)

    #
    # Animate: this function *must* return one tuple for all changed objects
    # for animation.FuncAnimation to know what it must update in the figure.
    #
    def update_plot(i):
        fig.canvas.set_window_title(
            "Time step {} out of {}".format(i + 1, num_timesteps))

        bars = artists[:len(electrode_ids)]
        lines = artists[len(electrode_ids):]

        for (e, f, bar) in zip(electrode_ids, feature_ids, bars):
            r, c = points[e]
            data = populations[i, f, r, c]
            bar.set_height(data)

        for e, line in enumerate(lines):
            r, c = points[e]
            z = circular_decoder(populations[i], centers, r, c)
            line.set_xdata([z])
            line.set_ydata([0.04])

        return artists

    anim = animation.FuncAnimation(
        fig=fig,
        func=update_plot,
        frames=range(num_timesteps),
        init_func=lambda: update_plot(0),
        interval=150,
        blit=True)

    plt.show()
    plt.draw()

    if save:
        anim.save(save, writer='avconv')

    return anim


@enforce_visual_style
def plot_stimulus(stimulus, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    missing = np.isnan(stimulus)
    hue = np.array(stimulus) / np.pi
    sat = np.ones_like(hue)
    val = np.full_like(hue, 0.5)
    rgb = color.hsv2rgb(np.dstack((hue, sat, val)))
    rgb[missing] = [0.5, 0.5, 0.5]

    # Plot
    ax.set_title("Stimulus")
    ax.imshow(rgb)
    ax.axis("off")

    return ax


@enforce_visual_style
def decode_and_plot_population_map(population,
                                   centers,
                                   ax=None,
                                   _return_artist=False,
                                   _im=None):
    if population.ndim != 3:
        raise ValueError("Incorrect input dimensionality!")

    if ax is None:
        _, ax = plt.subplots()

    # Convert 3D population tensor into 2D color image
    decoded = circular_decoder(population, centers)
    entropy = stats.entropy(population, axis=0) / np.log(len(centers))
    hue = decoded / np.pi
    sat = 1.0 - entropy
    val = np.full_like(hue, 0.5)
    rgb = color.hsv2rgb(np.dstack((hue, sat, val)))
    rgb[np.isnan(entropy)] = [0.5, 0.5, 0.5]

    # Plot
    label = as_plot_title(population)
    ax.set_title("Population response: {}".format(label))
    if _im is None:
        _im = ax.imshow(rgb)
    else:
        _im.set_array(rgb)
    ax.axis("off")

    if _return_artist:
        return _im

    return ax


@enforce_visual_style
def decode_and_animate_population_map(populations,
                                      centers,
                                      save=None,
                                      ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    num_timesteps = len(populations)
    im = decode_and_plot_population_map(
        populations[0], centers, ax=ax, _return_artist=True)

    def update_plot(i):
        fig.canvas.set_window_title(
            "Time step {} out of {}".format(i + 1, num_timesteps))

        decode_and_plot_population_map(populations[i], centers, ax=ax, _im=im)

        return [im]

    anim = animation.FuncAnimation(
        fig=fig,
        func=update_plot,
        frames=range(num_timesteps),
        init_func=lambda: update_plot(0),
        interval=150,
        blit=True)

    plt.show()
    plt.draw()

    if save:
        anim.save(save, writer='avconv')

    return anim


def circular_decoder(tensor_khw, centers, x=None, y=None):
    if x is None and y is None:
        s = np.s_[:, :, :]
    else:
        s = np.s_[:, x, y]

    Z = tensor_khw[s].sum(axis=0)
    t = 2 * centers - np.pi
    sin = np.tensordot(np.sin(t), tensor_khw[s], axes=1) / Z
    cos = np.tensordot(np.cos(t), tensor_khw[s], axes=1) / Z
    D = np.arctan2(sin, cos)
    D = (D + np.pi) / 2

    return D


def as_plot_title(obj):
    """
    See https://stackoverflow.com/a/9283563.
    """
    if not isinstance(obj, str):
        obj = obj.__class__.__name__

    return re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', obj)
