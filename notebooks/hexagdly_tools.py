""" 
Tools to be used in combination with hexagdly.

"""

import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib import gridspec


def plot_hextensor(
    tensor,
    image_range=(0, None),
    channel_range=(0, None),
    cmap="Greys",
    figname="figure",
    mask=[],
):
    r"""Plot the hexagonal representation of a 4D tensor according to the 
        addressing sheme used by HexagDLy.

        Args:
        tensor:         torch tensor or numpy array containing the hexagonal data points
        image_range:    tuple of ints, range defining the images to be plotted
        channel_range:  tuple of ints, range defining the channels to be plotted
        cmap:           colourmap
        figname:        str, name of figure
        mask:           list of ints that depict the pixels to skip in plots 
                        counting top to bottom left to right from the top left pixel  

    """
    try:
        tensor = tensor.data.numpy()
    except:
        pass
    inshape = np.shape(
        tensor[image_range[0] : image_range[1], channel_range[0] : channel_range[1]]
    )
    inexamples = inshape[0]
    inchannels = inshape[1]
    if inexamples != 1 and inchannels != 1:
        print("Choose one image and n channels or one channel an n images to display!")
        sys.exit()
    nimages = max(inexamples, inchannels)
    hexagons = [[] for i in range(nimages)]
    intensities = [[] for i in range(nimages)]
    fig = plt.figure(figname, (5, 5))
    fig.clear()
    nrows = int(np.ceil(np.sqrt(nimages)))
    gs = gridspec.GridSpec(nrows, nrows)
    gs.update(wspace=0, hspace=0)
    for i in range(nimages):
        if inexamples >= inchannels:
            a = i
            b = 0
        else:
            a = 0
            b = i
        npixel = 0
        for x in range(np.shape(tensor[image_range[0] + a, channel_range[0] + b])[1]):
            for y in range(
                np.shape(tensor[image_range[0] + a, channel_range[0] + b])[0]
            ):
                if npixel not in mask:
                    intensity = tensor[image_range[0] + a, channel_range[0] + b, y, x]
                    hexagon = RegularPolygon(
                        (x * np.sqrt(3) / 2, -(y + np.mod(x, 2) * 0.5)),
                        6,
                        0.577349,
                        orientation=np.pi / 6,
                    )
                    intensities[i].append(intensity)
                    hexagons[i].append(hexagon)
                npixel += 1
        ax = fig.add_subplot(gs[i])
        ax.set_xlim([-1, np.shape(tensor[image_range[0] + a, channel_range[0] + b])[0]])
        ax.set_ylim(
            [
                -1.15 * np.shape(tensor[image_range[0] + a, channel_range[0] + b])[1]
                - 1,
                1,
            ]
        )
        ax.set_axis_off()
        p = PatchCollection(
            np.array(hexagons[i]), cmap=cmap, alpha=0.9, edgecolors="k", linewidth=1
        )
        p.set_array(np.array(np.array(intensities[i])))
        ax.add_collection(p)
        ax.set_aspect("equal")
        plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.tight_layout()


def plot_squaretensor(
    tensor,
    image_range=(0, None),
    channel_range=(0, None),
    cmap="Greys",
    figname="figure",
):
    r""" Same as plot_hex_tensor, just that the tensor is plotted in squares 
    in a cartesian grid.

    """
    try:
        tensor = tensor.data.numpy()
    except Exception as e:
        print("Input not given as pytorch tensor! Continuing...")
        pass
    inshape = np.shape(
        tensor[image_range[0] : image_range[1], channel_range[0] : channel_range[1]]
    )
    inexamples = inshape[0]
    inchannels = inshape[1]
    if inexamples != 1 and inchannels != 1:
        print("Choose one image and n channels or one channel an n images to display!")
        sys.exit()
    nimages = max(inexamples, inchannels)
    fig = plt.figure(figname, (6, 6))
    fig.clear()
    nrows = int(np.ceil(np.sqrt(nimages)))
    gs = gridspec.GridSpec(nrows, nrows)
    gs.update(wspace=0.2, hspace=0)
    for i in range(nimages):
        if inexamples >= inchannels:
            a = i
            b = 0
        else:
            a = 0
            b = i
        npixel = 0
        ax = fig.add_subplot(gs[i])
        ax.set_axis_off()
        ax.pcolor(tensor[a][b], cmap=cmap, edgecolors="k", linewidths=0.4)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_frame_on(True)
        plt.subplots_adjust(top=0.95, bottom=0.05)
