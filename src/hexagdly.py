"""
This file contains utilities to set up hexagonal convolution and pooling
kernels in PyTorch. The size of the input is abitrary, whereas the layout
from top to bottom (along tensor index 2) has to be of zig-zag-edge shape
and from left to right (along tensor index 3) of armchair-edge shape as
shown below.
 __    __                                 __ __ __ __
/11\__/31\__  . . .                      |11|21|31|41| . . .
\__/21\__/41\                            |__|__|__|__| 
/12\__/32\__/ . . .        _______|\     |12|22|32|42| . . .
\__/22\__/42\             |         \    |__|__|__|__| 
   \__/  \__/             |_______  /     
 .  .  .  .  .                    |/       .  .  .  .  .
 .  .  .  .    .                           .  .  .  .    .
 .  .  .  .      .                         .  .  .  .      .

For more information visit https://github.com/ai4iacts/hexagdly

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class HexBase:
    def __init__(self):
        super(HexBase, self).__init__()
        self.hexbase_size = None
        self.depth_size = None
        self.hexbase_stride = None
        self.depth_stride = None
        self.input_size_is_known = False
        self.odd_columns_slices = []
        self.odd_columns_pads = []
        self.even_columns_slices = []
        self.even_columns_pads = []
        self.dimensions = None
        self.combine = None
        self.process = None
        self.kwargs = dict()

    def shape_for_odd_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        pads[0] = kernel_number
        # right
        pads[1] = max(
            0, kernel_number - ((input_size[-1] - 1) % (2 * self.hexbase_stride))
        )
        # top
        pads[2] = self.hexbase_size - int(kernel_number / 2)
        # bottom
        constraint = (
            input_size[-2]
            - 1
            - int(
                (input_size[-2] - 1 - int(self.hexbase_stride / 2))
                / self.hexbase_stride
            )
            * self.hexbase_stride
        )
        bottom = (self.hexbase_size - int((kernel_number + 1) / 2)) - constraint
        if bottom >= 0:
            pads[3] = bottom
        else:
            slices[1] = bottom

        return slices, pads

    def shape_for_even_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        left = kernel_number - self.hexbase_stride
        if left >= 0:
            pads[0] = left
        else:
            slices[2] = -left
        # right
        pads[1] = max(
            0,
            kernel_number
            - ((input_size[-1] - 1 - self.hexbase_stride) % (2 * self.hexbase_stride)),
        )
        # top
        top_shift = -(kernel_number % 2) if (self.hexbase_stride % 2) == 1 else 0
        top = (
            (self.hexbase_size - int(kernel_number / 2))
            + top_shift
            - int(self.hexbase_stride / 2)
        )
        if top >= 0:
            pads[2] = top
        else:
            slices[0] = -top
        # bottom
        bottom_shift = 0 if (self.hexbase_stride % 2) == 1 else -(kernel_number % 2)
        pads[3] = max(
            0,
            self.hexbase_size
            - int(kernel_number / 2)
            + bottom_shift
            - (
                (input_size[-2] - int(self.hexbase_stride / 2) - 1)
                % self.hexbase_stride
            ),
        )

        return slices, pads

    def get_padded_input(self, input, pads):
        if self.dimensions == 2:
            return nn.ZeroPad2d(tuple(pads))(input)
        elif self.dimensions == 3:
            return nn.ConstantPad3d(tuple(pads + [0, 0]), 0)(input)

    def get_sliced_input(self, input, slices):
        if self.dimensions == 2:
            return input[:, :, slices[0] : slices[1], slices[2] : slices[3]]
        elif self.dimensions == 3:
            return input[:, :, :, slices[0] : slices[1], slices[2] : slices[3]]

    def get_dilation(self, dilation_2d):
        if self.dimensions == 2:
            return dilation_2d
        elif self.dimensions == 3:
            return tuple([1] + list(dilation_2d))

    def get_stride(self):
        if self.dimensions == 2:
            return (self.hexbase_stride, 2 * self.hexbase_stride)
        elif self.dimensions == 3:
            return (self.depth_stride, self.hexbase_stride, 2 * self.hexbase_stride)

    def get_ordered_output(self, input, order):
        if self.dimensions == 2:
            return input[:, :, :, order]
        elif self.dimensions == 3:
            return input[:, :, :, :, order]

    # general implementation of an operation with a hexagonal kernel
    def operation_with_arbitrary_stride(self, input):
        assert (
            input.size(-2) - (self.hexbase_stride // 2) >= 0
        ), "Too few rows to apply hex conv with the stide that is set"
        odd_columns = None
        even_columns = None

        for i in range(self.hexbase_size + 1):
            dilation_base = (1, 1) if i == 0 else (1, 2 * i)

            if not self.input_size_is_known:
                slices, pads = self.shape_for_odd_columns(input.size(), i)
                self.odd_columns_slices.append(slices)
                self.odd_columns_pads.append(pads)
                slices, pads = self.shape_for_even_columns(input.size(), i)
                self.even_columns_slices.append(slices)
                self.even_columns_pads.append(pads)
                if i == self.hexbase_size:
                    self.input_size_is_known = True

            if odd_columns is None:
                odd_columns = self.process(
                    self.get_padded_input(
                        self.get_sliced_input(input, self.odd_columns_slices[i]),
                        self.odd_columns_pads[i],
                    ),
                    getattr(self, "kernel" + str(i)),
                    dilation=self.get_dilation(dilation_base),
                    stride=self.get_stride(),
                    **self.kwargs
                )
            else:
                odd_columns = self.combine(
                    odd_columns,
                    self.process(
                        self.get_padded_input(
                            self.get_sliced_input(input, self.odd_columns_slices[i]),
                            self.odd_columns_pads[i],
                        ),
                        getattr(self, "kernel" + str(i)),
                        dilation=self.get_dilation(dilation_base),
                        stride=self.get_stride(),
                    ),
                )

            if even_columns is None:
                even_columns = self.process(
                    self.get_padded_input(
                        self.get_sliced_input(input, self.even_columns_slices[i]),
                        self.even_columns_pads[i],
                    ),
                    getattr(self, "kernel" + str(i)),
                    dilation=self.get_dilation(dilation_base),
                    stride=self.get_stride(),
                    **self.kwargs
                )
            else:
                even_columns = self.combine(
                    even_columns,
                    self.process(
                        self.get_padded_input(
                            self.get_sliced_input(input, self.even_columns_slices[i]),
                            self.even_columns_pads[i],
                        ),
                        getattr(self, "kernel" + str(i)),
                        dilation=self.get_dilation(dilation_base),
                        stride=self.get_stride(),
                    ),
                )

        concatenated_columns = torch.cat(
            (odd_columns, even_columns), 1 + self.dimensions
        )

        n_odd_columns = odd_columns.size(-1)
        n_even_columns = even_columns.size(-1)
        if n_odd_columns == n_even_columns:
            order = [
                int(i + x * n_even_columns)
                for i in range(n_even_columns)
                for x in range(2)
            ]
        else:
            order = [
                int(i + x * n_odd_columns)
                for i in range(n_even_columns)
                for x in range(2)
            ]
            order.append(n_even_columns)

        return self.get_ordered_output(concatenated_columns, order)

    # a slightly faster, case specific implementation of the hexagonal convolution
    def operation_with_single_hexbase_stride(self, input):
        columns_mod2 = input.size(-1) % 2
        odd_kernels_odd_columns = []
        odd_kernels_even_columns = []
        even_kernels_all_columns = []

        even_kernels_all_columns = self.process(
            self.get_padded_input(input, [0, 0, self.hexbase_size, self.hexbase_size]),
            self.kernel0,
            stride=(1, 1) if self.dimensions == 2 else (self.depth_stride, 1, 1),
            **self.kwargs
        )
        if self.hexbase_size >= 1:
            odd_kernels_odd_columns = self.process(
                self.get_padded_input(
                    input, [1, columns_mod2, self.hexbase_size, self.hexbase_size - 1]
                ),
                self.kernel1,
                dilation=self.get_dilation((1, 2)),
                stride=self.get_stride(),
            )
            odd_kernels_even_columns = self.process(
                self.get_padded_input(
                    input,
                    [0, 1 - columns_mod2, self.hexbase_size - 1, self.hexbase_size],
                ),
                self.kernel1,
                dilation=self.get_dilation((1, 2)),
                stride=self.get_stride(),
            )

        if self.hexbase_size > 1:
            for i in range(2, self.hexbase_size + 1):
                if i % 2 == 0:
                    even_kernels_all_columns = self.combine(
                        even_kernels_all_columns,
                        self.process(
                            self.get_padded_input(
                                input,
                                [
                                    i,
                                    i,
                                    self.hexbase_size - int(i / 2),
                                    self.hexbase_size - int(i / 2),
                                ],
                            ),
                            getattr(self, "kernel" + str(i)),
                            dilation=self.get_dilation((1, 2 * i)),
                            stride=(1, 1)
                            if self.dimensions == 2
                            else (self.depth_stride, 1, 1),
                        ),
                    )
                else:
                    x = self.hexbase_size + int((1 - i) / 2)
                    odd_kernels_odd_columns = self.combine(
                        odd_kernels_odd_columns,
                        self.process(
                            self.get_padded_input(
                                input, [i, i - 1 + columns_mod2, x, x - 1]
                            ),
                            getattr(self, "kernel" + str(i)),
                            dilation=self.get_dilation((1, 2 * i)),
                            stride=self.get_stride(),
                        ),
                    )
                    odd_kernels_even_columns = self.combine(
                        odd_kernels_even_columns,
                        self.process(
                            self.get_padded_input(
                                input, [i - 1, i - columns_mod2, x - 1, x]
                            ),
                            getattr(self, "kernel" + str(i)),
                            dilation=self.get_dilation((1, 2 * i)),
                            stride=self.get_stride(),
                        ),
                    )

        odd_kernels_concatenated_columns = torch.cat(
            (odd_kernels_odd_columns, odd_kernels_even_columns), 1 + self.dimensions
        )

        n_odd_columns = odd_kernels_odd_columns.size(-1)
        n_even_columns = odd_kernels_even_columns.size(-1)
        if n_odd_columns == n_even_columns:
            order = [
                int(i + x * n_even_columns)
                for i in range(n_even_columns)
                for x in range(2)
            ]
        else:
            order = [
                int(i + x * n_odd_columns)
                for i in range(n_even_columns)
                for x in range(2)
            ]
            order.append(n_even_columns)

        return self.combine(
            even_kernels_all_columns,
            self.get_ordered_output(odd_kernels_concatenated_columns, order),
        )


class Conv2d(HexBase, nn.Module):
    r"""Applies a 2D hexagonal convolution`
        
        Args:
            in_channels:        int: number of input channels
            out_channels:       int: number of output channels
            kernel_size:        int: number of layers with neighbouring pixels
                                     covered by the pooling kernel
            stride:             int: length of strides
            bias:               bool: add bias if True (default)
            debug:              bool: switch to debug mode
                                    False: weights are initalised with
                                           kaiming normal, bias with 0.01 (default)
                                    True: weights / bias are set to 1.
            
        Examples::
        
        >>> conv2d = hexagdly.Conv2d(1,3,2,1)
        >>> input = torch.randn(1, 1, 4, 2)
        >>> output = conv2d(input)
        >>> print(output)
        """

    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, bias=True, debug=False
    ):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hexbase_size = kernel_size
        self.hexbase_stride = stride
        self.debug = debug
        self.bias = bias
        self.dimensions = 2
        self.process = F.conv2d
        self.combine = torch.add

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                Parameter(
                    torch.Tensor(
                        out_channels,
                        in_channels,
                        1 + 2 * self.hexbase_size - i,
                        1 if i == 0 else 2,
                    )
                ),
            )
        if self.bias:
            self.bias_tensor = Parameter(torch.Tensor(out_channels))
            self.kwargs = {"bias": self.bias_tensor}
        else:
            self.kwargs = {"bias": None}
        self.init_parameters(self.debug)

    def init_parameters(self, debug):
        if debug:
            for i in range(self.hexbase_size + 1):
                nn.init.constant_(getattr(self, "kernel" + str(i)), 1)
            if self.bias:
                nn.init.constant_(getattr(self, "kwargs")["bias"], 1.0)
        else:
            for i in range(self.hexbase_size + 1):
                nn.init.kaiming_normal_(getattr(self, "kernel" + str(i)))
            if self.bias:
                nn.init.constant_(getattr(self, "kwargs")["bias"], 0.01)

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={hexbase_size}"
            ", stride={hexbase_stride}"
        )
        if self.bias is False:
            s += ", bias=False"
        if self.debug is True:
            s += ", debug=True"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d_CustomKernel(HexBase, nn.Module):
    r"""Applies a 2D hexagonal convolution with custom kernels`
        
        Args:
            sub_kernels:        list:   list containing sub-kernels as numpy arrays
            stride:             int:    length of strides
            bias:               array:  numpy array with biases (default: None)
            requires_grad:      bool:   trainable parameters if True (default: False)
            debug:              bool:   If True a kernel of size one with all values
                                        set to 1 will be applied as well as no bias
                                        (default: False)
            
        Examples::
        
        Given in the online repository https://github.com/ai4iacts/hexagdly
        """

    def __init__(
        self, sub_kernels=[], stride=1, bias=None, requires_grad=False, debug=False
    ):
        super(Conv2d_CustomKernel, self).__init__()
        self.sub_kernels = sub_kernels
        self.bias_array = bias
        self.hexbase_stride = stride
        self.requires_grad = requires_grad
        self.debug = debug
        self.dimensions = 2
        self.process = F.conv2d
        self.combine = torch.add

        self.init_parameters(self.debug)

    def init_parameters(self, debug):
        if debug or len(self.sub_kernels) == 0:
            print(
                "The debug kernel is used for {name}!".format(
                    name=self.__class__.__name__
                )
            )
            self.sub_kernels = [
                np.array([[[[1], [1], [1]]]]),
                np.array([[[[1, 1], [1, 1]]]]),
            ]
        self.hexbase_size = len(self.sub_kernels) - 1
        self.check_sub_kernels()

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                Parameter(
                    torch.from_numpy(self.sub_kernels[i]).type(torch.FloatTensor),
                    requires_grad=self.requires_grad,
                ),
            )

        if not debug and not self.bias_array is None:
            self.check_bias()
            self.bias_tensor = Parameter(
                torch.from_numpy(self.bias_array).type(torch.FloatTensor),
                requires_grad=self.requires_grad,
            )
            self.kwargs = {"bias": self.bias_tensor}
            self.bias = True
        else:
            self.bias = False
            if not self.bias_array is None:
                print(
                    "{name}: Bias is not used in debug mode!".format(
                        name=self.__class__.__name__
                    )
                )

    def check_sub_kernels(self):
        for i in range(self.hexbase_size + 1):
            assert (
                type(self.sub_kernels[i]).__module__ == np.__name__
            ), "sub-kernels must be given as numpy arrays"
            assert (
                len(self.sub_kernels[i].shape) == 4
            ), "sub-kernels must be of rank 4 for a 2d convolution"
            if i == 0:
                assert (
                    self.sub_kernels[i].shape[3] == 1
                ), "first sub-kernel must have only 1 column"
                assert (
                    self.sub_kernels[i].shape[2] == 2 * self.hexbase_size + 1
                ), "first sub-kernel must have 2* (kernel size) + 1 rows"
                self.out_channels = self.sub_kernels[i].shape[0]
                self.in_channels = self.sub_kernels[i].shape[1]
            else:
                assert (
                    self.sub_kernels[i].shape[3] == 2
                ), "sub-kernel {}: all but the first sub-kernel must have 2 columns".format(
                    i
                )
                assert (
                    self.sub_kernels[i].shape[2] == 2 * self.hexbase_size + 1 - i
                ), "{}. sub-kernel must have 2* (kernel size) + 1 - {} rows".format(
                    i, i
                )
                assert (
                    self.sub_kernels[i].shape[0] == self.out_channels
                ), "sub-kernel {}: out channels are not consistent".format(i)
                assert (
                    self.sub_kernels[i].shape[1] == self.in_channels
                ), "sub-kernel {}: in channels are not consistent".format(i)

    def check_bias(self):
        assert (
            type(self.bias_array).__module__ == np.__name__
        ), "bias must be given as a numpy array"
        assert len(self.bias_array.shape) == 1, "bias must be of rank 1"
        assert (
            self.bias_array.shape[0] == self.out_channels
        ), "bias must have length equal to number of out channels"

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={hexbase_size}"
            ", stride={hexbase_stride}"
        )
        if self.bias is False:
            s += ", bias=False"
        if self.debug is True:
            s += ", debug=True"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv3d(HexBase, nn.Module):
    r"""Applies a 3D hexagonal convolution`
         
        Args:
            in_channels:        int: number of input channels
            out_channels:       int: number of output channels
            kernel_size:        int, tuple: number of layers with neighbouring pixels
                                            covered by the pooling kernel
                                    int: same number of layers in all dimensions
                                    tuple of two ints:
                                        1st int: layers in depth
                                        2nd int: layers in hexagonal base
            stride:             int, tuple: length of strides
                                    int: same lenght of strides in each dimension
                                    tuple of two ints:
                                        1st int: length of strides in depth
                                        2nd int: length of strides in hexagonal base
            bias:               bool: add bias if True (default)
            debug:              bool: switch to debug mode
                                    False: weights are initalised with
                                           kaiming normal, bias with 0.01 (default)
                                    True: weights / bias are set to 1.
         
        Examples::
         
        >>> conv3d = hexagdly.Conv3d((1,1), (2,2))
        >>> input = torch.randn(1, 1, 6, 5, 4)
        >>> output = conv3d(input)
        >>> print(output)
        """

    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, bias=True, debug=False
    ):
        super(Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.hexbase_size = kernel_size
            self.depth_size = kernel_size
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "Need a tuple of two ints to set kernel size"
            self.hexbase_size = kernel_size[1]
            self.depth_size = kernel_size[0]
        if isinstance(stride, int):
            self.hexbase_stride = stride
            self.depth_stride = stride
        elif isinstance(stride, tuple):
            assert len(stride) == 2, "Need a tuple of two ints to set stride"
            self.hexbase_stride = stride[1]
            self.depth_stride = stride[0]
        self.debug = debug
        self.bias = bias
        self.dimensions = 3
        self.process = F.conv3d
        self.combine = torch.add

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                Parameter(
                    torch.Tensor(
                        out_channels,
                        in_channels,
                        self.depth_size,
                        1 + 2 * self.hexbase_size - i,
                        1 if i == 0 else 2,
                    )
                ),
            )
        if self.bias:
            self.bias_tensor = Parameter(torch.Tensor(out_channels))
            self.kwargs = {"bias": self.bias_tensor}
        else:
            self.kwargs = {"bias": None}

        self.init_parameters(self.debug)

    def init_parameters(self, debug):
        if debug:
            for i in range(self.hexbase_size + 1):
                nn.init.constant_(getattr(self, "kernel" + str(i)), 1)
            if self.bias:
                nn.init.constant_(getattr(self, "kwargs")["bias"], 1.0)
        else:
            for i in range(self.hexbase_size + 1):
                nn.init.kaiming_normal_(getattr(self, "kernel" + str(i)))
            if self.bias:
                nn.init.constant_(getattr(self, "kwargs")["bias"], 0.01)

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size=({depth_size}, {hexbase_size})"
            ", stride=({depth_stride}, {hexbase_stride})"
        )
        if self.bias is False:
            s += ", bias=False"
        if self.debug is True:
            s += ", debug=True"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv3d_CustomKernel(HexBase, nn.Module):
    r"""Applies a 3D hexagonal convolution with custom kernels`
        
        Args:
            sub_kernels:        list: list containing sub-kernels as numpy arrays
            stride:             stride:             int, tuple: length of strides
                                    int: same lenght of strides in each dimension
                                    tuple of two ints:
                                        1st int: length of strides in depth
                                        2nd int: length of strides in hexagonal base
            requires_grad:      bool:   trainable parameters if True (default: False)
            debug:              bool:   If True a kernel of size one with all values
                                        set to 1 will be applied as well as no bias
                                        (default: False)
            
        Examples::
        
        Given in the online repository https://github.com/ai4iacts/hexagdly
        """

    def __init__(
        self, sub_kernels=[], stride=1, bias=None, requires_grad=False, debug=False
    ):
        super(Conv3d_CustomKernel, self).__init__()
        self.sub_kernels = sub_kernels
        self.bias_array = bias
        if isinstance(stride, int):
            self.hexbase_stride = stride
            self.depth_stride = stride
        elif isinstance(stride, tuple):
            assert len(stride) == 2, "Need a tuple of two ints to set stride"
            self.hexbase_stride = stride[1]
            self.depth_stride = stride[0]
        self.requires_grad = requires_grad
        self.debug = debug
        self.dimensions = 3
        self.process = F.conv3d
        self.combine = torch.add

        self.init_parameters(self.debug)

    def init_parameters(self, debug):
        if debug or len(self.sub_kernels) == 0:
            print(
                "The debug kernel is used for {name}!".format(
                    name=self.__class__.__name__
                )
            )
            self.sub_kernels = [
                np.array([[[[[1], [1], [1]]]]]),
                np.array([[[[[1, 1], [1, 1]]]]]),
            ]
        self.hexbase_size = len(self.sub_kernels) - 1
        self.check_sub_kernels()

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                Parameter(
                    torch.from_numpy(self.sub_kernels[i]).type(torch.FloatTensor),
                    requires_grad=self.requires_grad,
                ),
            )

        if not debug and not self.bias_array is None:
            self.check_bias()
            self.bias_tensor = Parameter(
                torch.from_numpy(self.bias_array).type(torch.FloatTensor),
                requires_grad=self.requires_grad,
            )
            self.kwargs = {"bias": self.bias_tensor}
            self.bias = True
        else:
            self.bias = False
            print("No bias is used for {name}!".format(name=self.__class__.__name__))

    def check_sub_kernels(self):
        for i in range(self.hexbase_size + 1):
            assert (
                type(self.sub_kernels[i]).__module__ == np.__name__
            ), "sub-kernels must be given as numpy arrays"
            assert (
                len(self.sub_kernels[i].shape) == 5
            ), "sub-kernels must be of rank 5 for a 3d convolution"
            if i == 0:
                assert (
                    self.sub_kernels[i].shape[4] == 1
                ), "first sub-kernel must have only 1 column"
                assert (
                    self.sub_kernels[i].shape[3] == 2 * self.hexbase_size + 1
                ), "first sub-kernel must have 2* (kernel size) + 1 rows"
                self.out_channels = self.sub_kernels[i].shape[0]
                self.in_channels = self.sub_kernels[i].shape[1]
                self.depth_size = self.sub_kernels[i].shape[2]
            else:
                assert (
                    self.sub_kernels[i].shape[4] == 2
                ), "sub-kernel {}: all but the first sub-kernel must have 2 columns".format(
                    i
                )
                assert (
                    self.sub_kernels[i].shape[3] == 2 * self.hexbase_size + 1 - i
                ), "{}th sub-kernel must have 2* (kernel size) + 1 - {} rows".format(
                    i, i
                )
                assert (
                    self.sub_kernels[i].shape[0] == self.out_channels
                ), "sub-kernel {}: out channels are not consistent".format(i)
                assert (
                    self.sub_kernels[i].shape[1] == self.in_channels
                ), "sub-kernel {}: out channels are not consistent".format(i)
                assert (
                    self.sub_kernels[i].shape[2] == self.depth_size
                ), "sub-kernel {}: depths are not consistent".format(i)

    def check_bias(self):
        assert (
            type(self.bias_array).__module__ == np.__name__
        ), "bias must be given as a numpy array"
        assert len(self.bias_array.shape) == 1, "bias must be of rank 1"
        assert (
            self.bias_array.shape[0] == self.out_channels
        ), "bias must have length equal to number of out channels"

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size=({depth_size}, {hexbase_size})"
            ", stride=({depth_stride}, {hexbase_stride})"
        )
        if self.bias is False:
            s += ", bias=False"
        if self.debug is True:
            s += ", debug=True"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MaxPool2d(HexBase, nn.Module):
    r"""Applies a 2D hexagonal max pooling`
        
        Args:
            kernel_size:        int: number of layers with neighbouring pixels
                                     covered by the pooling kernel
            stride:             int: length of strides
        
        Examples::
        
            >>> maxpool2d = hexagdly.MaxPool2d(1,2)
            >>> input = torch.randn(1, 1, 4, 2)
            >>> output = maxpool2d(input)
            >>> print(output)
        """

    def __init__(self, kernel_size=1, stride=1):
        super(MaxPool2d, self).__init__()
        self.hexbase_size = kernel_size
        self.hexbase_stride = stride
        self.dimensions = 2
        self.process = F.max_pool2d
        self.combine = torch.max

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                (1 + 2 * self.hexbase_size - i, 1 if i == 0 else 2),
            )

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = "{name}(kernel_size={hexbase_size}" ", stride={hexbase_stride})"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MaxPool3d(HexBase, nn.Module):
    r"""Applies a 3D hexagonal max pooling`
         
        Args:
            kernel_size:        int, tuple: number of layers with neighbouring pixels
                                            covered by the pooling kernel
                                    int: same number of layers in all dimensions
                                    tuple of two ints:
                                        1st int: layers in depth
                                        2nd int: layers in hexagonal base
            stride:             int, tuple: length of strides
                                    int: same lenght of strides in each dimension
                                    tuple of two ints:
                                        1st int: length of strides in depth
                                        2nd int: length of strides in hexagonal base
         
        Examples::
         
            >>> maxpool3d = hexagdly.MaxPool3d((1,1), (2,2))
            >>> input = torch.randn(1, 1, 6, 5, 4)
            >>> output = maxpool3d(input)
            >>> print(output)
        """

    def __init__(self, kernel_size=1, stride=1):
        super(MaxPool3d, self).__init__()
        if isinstance(kernel_size, int):
            self.hexbase_size = kernel_size
            self.depth_size = kernel_size
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "Too many parameters"
            self.hexbase_size = kernel_size[1]
            self.depth_size = kernel_size[0]
        if isinstance(stride, int):
            self.hexbase_stride = stride
            self.depth_stride = stride
        elif isinstance(stride, tuple):
            assert len(stride) == 2, "Too many parameters"
            self.hexbase_stride = stride[1]
            self.depth_stride = stride[0]
        self.dimensions = 3
        self.process = F.max_pool3d
        self.combine = torch.max

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                (self.depth_size, 1 + 2 * self.hexbase_size - i, 1 if i == 0 else 2),
            )

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}(kernel_size=({depth_size}, {hexbase_size})"
            ", stride=({depth_stride}, {hexbase_stride}))"
        )
        return s.format(name=self.__class__.__name__, **self.__dict__)
