# HexagDLy - Processing Hexagonal Data with PyTorch

HexagDLy provides convolution and pooling methods for hexagonally sampled data within the deep learning framework [PyTorch](https://github.com/pytorch/pytorch).

- [Getting Started](#getting-started)
- [Preparing the Data](#preparing-the-data)
- [How to use HexagDLy](#how-to-use-hexagdly)
- [General Concept](#general-concept)
- [Disclaimer](#disclaimer)
- [Citing HexagDLy](#citation)


## Getting Started

There are different ways to get HexagDLy up and running on your system as shown below. Basic examples for the application of HexagDLy are given as [notebooks](notebooks). Additionally unit tests are provided in [tests](tests).

### Pip Installation

The suggested way to install HexagDLy is to set up a clean virtual python environment (e.g. with conda, see [https://www.anaconda.com/](https://www.anaconda.com/)) and use the provided pip installer. To install basic functionalities only use:

```
pip install hexagdly
```

To get all necessary dependencies to run the provided [unit tests](tests) and [notebooks](notebooks), add the ```dev``` option:

```
pip install hexagdly[dev]
```


### Manual Installation

HexagDLy requires a working installation of [PyTorch](https://github.com/pytorch/pytorch). Please visit the PyTorch website http://pytorch.org/ or github page https://github.com/pytorch/pytorch and follow the installation instructions.
If you have downloaded HexagDLy, just add the directory `hexagdly/src` to your system's `$PYTHONPATH`, e.g. by adding the following line to your `.bashrc` or `.bash_profile` (or wherever your paths can be set)

```
export PYTHONPATH='/path/to/hexagdly/src':$PYTHONPATH
```

## How to use HexagDLy

As HexagDLy is based on PyTorch, it is of advantage to be familiar with PyTorch's functionalities and concepts.
Furthermore, before applying HexagDLy, it should be ensured that the input data has the correct format. HexagDLy uses an addressing scheme to map data from a hexagonal grid to a torch tensor. An [example](notebooks/how_to_apply_adressing_scheme.ipynb) is provided that illustrates the steps to get the data into the correct layout.

If the data has this required layout and HexagDLy is installed, performing hexagonal convolutions is as easy as the following example:

```
import torch
import hexagdly
 
kernel_size, stride = 1, 4
in_channels, out_channels = 1, 3

hexconv = hexagdly.Conv2d(in_channels, out_channels, kernel_size, stride)
input = torch.rand(1,1,21,21)
output = hexconv(input)
```

In this example, a random input tensor of shape (1, 1, 21, 21) is convolved with a so called next neighbour hexagonal kernel (size = 1) with one input channel and three output channels, using a stride of four. The output is a tensor of size (1, 3, 5, 6).

HexagDLy is desinged to conserve the hexagonal symmetry of the input. Therefore, a hexagonal kernel is always 6-fold symmetric and may only move along the symmetry axes of the grid in symmetric steps.
An automatic padding of the input is applied, depending on the kernel size, stride and dimensions of the input.
The image below shows examples of how kernels of different size and stride visit certain regions of an input. The orange cells mark the hexagonal kernel centered on the top left cell, the starting point of each operation. The square gridlines depict cells on which the kernel is centered by moving it with the given stride.

![kerne size+stride](figures/kernel_size+stride.png "Examples of different kernels of different size and strides.")

**Please note**: Operations are only performed where the center point of a kernel is located within the input tensor. This could result in output columns of different length. In such cases the output will be sliced according to the shortest column. An example is the convolution with stride 3 in the center of the figure above. The red gridlines depict convolutions that are omitted in the output.

Examples for basic use-cases of HexagDLy are shown in the [notebooks](notebooks) folder.


## General Concept 

As common deep learning frameworks are designed to process data arranged in square grids, it is not (yet) possible out-of-the-box to process data points that are arranged on a hexagonal grid.
To process hexagonally sampled data with frameworks like PyTorch, it is therefore necessary to translate the information from the hexagonal grid to a square grid tensor.
Such a conversion is however not trivial, as square and hexagonal grids inhibit different symmetries, i.e. 4-fold symmetry vs. 6-fold symmetry.

This problem is solved in HexagDLy by using an addressing scheme that maps the data from its original hexagonal grid to a square grid and by adapting the convolution operations to regard the symmetry of the hexagonal grid. The applied addressing scheme basically aligns the columns of the hexagonal grid. 
By applying a standard square-grid convolution to this data, the kernel disregards the original pixel-to-pixel neighbour relation and breaks the 6-fold symmetry. In order to perform a valid hexagonal convolution it is necessary to split the kernel into sub-kernels that, in combination, cover the true neighbours of a data point in the hexagonal grid. The concept is depicted in the image below:

![violating_symmetry](figures/violating_symmetry.png "Squeezing hexagonal data in a square grid and applying square convolution kernels disregards the symmetry of the hexagonal lattice. A valid hexagonal convolution can be performed by combining custom sub-kernels.")

Due to the alternating shift between the columns of the array of data points, the sub-kernels of a hexagonal convolution kernel have to shift accordingly, depending on whether the kernel is centered on an odd or an even column of the array. 
A full hexagonal convolution with the smallest hexagonal kernel (size 1) that conserves the dimensions of the input can be broken down into a total of three sub-convolutions that are performed by applying two different sub-kernels to three differently padded versions of the input. The resulting arrays are then merged and added to obtain the desired result.
The individual steps of this operation are depicted in the image below, where a toy input tensor is convolved with hexagonal size 1 kernel (all weights set to 1, i.e. the convolution adds up all data points covered by the kernel):

![explicit_next_neighbour_conv](figures/explicit_next_neighbour_conv.png "Schematic description of the individual sub-onvolutions  and combination of the individual outputs to perform a hexagonal convolution as provided by HexagDLy.")

Following the same concept, it is feasible to construct larger hexagonal convolution kernels as well as pooling operations by increasing the strides of the sub-kernels and exchanging the convolution operations with nested pooling methods.


## Disclaimer

HexagDLy is built as an easy-to-use prototyping tool to design convolutional neural networks for hexagonally sampled data. The implemented methods rather aim for flexibility then for performance.
Once a model is optimized, it is possible to hard-code the desired parameters like kernel size, stride and input dimensions to make the implementation faster.
Furthermore, the [General Concept](#general-concept) is not specific to PyTorch but can be adapted to other deep learning frameworks.


## Authors

* **Tim Lukas Holch**
* **Constantin Steppa**

See also the list of [contributors](https://github.com/ai4iacts/hexagdly/contributors) who participated in this project.


## License

This project is licensed under the MIT license - please consult the [LICENSE](LICENSE) file for details.


## Citation

We have published an open access paper about HexagDLy in [SoftwareX](https://www.sciencedirect.com/science/article/pii/S2352711018302723). If this work has helped your research, please cite us via:

```
@article{hexagdly_paper,
    title = "HexagDLy—Processing hexagonally sampled data with CNNs in PyTorch",
    author = "Constantin Steppa and Tim L. Holch",
    journal = "SoftwareX",
    volume = "9",
    pages = "193 - 198",
    year = "2019",
    issn = "2352-7110",
    doi = "https://doi.org/10.1016/j.softx.2019.02.010",
    url = "https://www.sciencedirect.com/science/article/pii/S2352711018302723",
    keywords = "Convolutional neural networks, Hexagonal grid, PyTorch, Astroparticle physics",
    abstract = "HexagDLy is a Python-library extending the PyTorch deep learning framework with convolution and pooling operations on hexagonal grids. It aims to ease the access to convolutional neural networks for applications that rely on hexagonally sampled data as, for example, commonly found in ground-based astroparticle physics experiments."
}
```

HexagDLy was developed as part of a research study in the field of ground-based gamma-ray astronomy published in [Astroparticle Physics](https://doi.org/10.1016/j.astropartphys.2018.10.003).


## Acknowledgments

This project evolved by exploring new analysis techniques for Imaging Atmospheric Cherenkov Telescopes with the High Energy Stereoscopic System (H.E.S.S.). We would like to thank the members of the H.E.S.S. collaboration for their support.


