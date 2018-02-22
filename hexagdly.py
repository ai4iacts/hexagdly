"""
This file contains utilities to set up hexagonal convolution and pooling
kernels in PyTorch. The size of the input is abitrary, whereas the layout
from top to bottom (along tensor index 2) has to be of zig-zag-edge shape
and from left to right (along tensor index 3) of armchair-edge shape as
shown below:
 _   _
/ \_/ \_ ...
\_/ \_/ \
/ \_/ \_/...
\_/ \_/ \
  \_/ \_/
.   .   .
.   .    .
.   .     .

If the hexagonal input is aligned as above, every second column of pixels
in x has to be shifted up by 1/2 pixel in order to get a valid input.

For more information visit https://github.com/ai4iacts/hexagdly

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class conv2d(nn.Module):
    r"""Applies a 2D hexagonal convolution`
        
        Args:
        in_channels:        number of input channels
        out_channels:       number of output channels
        size:               radius of the hexagonal convolution kernel
        stride:             stride along symmetry axes of convolved pixels
        debug:              False: weights are initalised with
                                   kaiming normal (default)
                            True: weights are set to 1
        usebias:            True: add bias (default)
        
        Attributes:
        kernel_(i):         subkernels of size
                            (out_channels, in_channels,
                             1 + 2 * size - i, 1 if i==0 else 2)
        
        
        Examples::
        
        >>> m = hexagdly.conv2d(1,3,2,1)
        >>> input = torch.randn(1, 1, 4, 2)
        >>> output = m(input)
        >>> print(output)
        """
    
    def __init__(self, in_channels, out_channels, size=1, stride=1, debug=False, usebias=True):
        super(conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.stride = stride
        self.debug = debug
        self.usebias = usebias
        self.input_size_is_known = False
        self.odd_columns_slices = []
        self.odd_columns_pads = []
        self.even_columns_slices = []
        self.even_columns_pads = []
        
        for i in range(self.size + 1):
            setattr(self, 'kernel' + str(i),
                    Parameter(torch.Tensor(out_channels, in_channels, 1 + 2 * self.size - i, 1 if i==0 else 2)))
        if self.usebias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        
        self.init_parameters(self.debug)
    
    def init_parameters(self, debug):
        if debug:
            for i in range(self.size + 1):
                nn.init.constant(getattr(self, 'kernel' + str(i)), 1)
        else:
            for i in range(self.size + 1):
                nn.init.kaiming_normal(getattr(self, 'kernel' + str(i)))
        if self.usebias:
            nn.init.constant(getattr(self, 'bias'), 0.01)

    def shape_for_odd_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        pads[0] = kernel_number
        # right
        pads[1] = max(0, kernel_number - ((input_size[3] - 1) % (2 * self.stride)))
        # top
        pads[2] = self.size - int(kernel_number / 2)
        # bottom
        constraint = input_size[2] - 1 - int((input_size[2] - 1 - int(self.stride / 2)) / self.stride) * self.stride
        bottom = (self.size - int((kernel_number + 1) / 2)) - constraint
        if bottom >= 0:
            pads[3] = bottom
        else:
            slices[1] = bottom
            
        return slices, pads
    
    def shape_for_even_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        left = kernel_number - self.stride
        if left >= 0:
            pads[0] = left
        else:
            slices[2] = -left
        # right
        pads[1] = max(0, kernel_number - ((input_size[3] - 1 - self.stride) % (2 * self.stride)))
        # top
        top_shift = -(kernel_number % 2) if (self.stride % 2) == 1 else 0
        top = (self.size - int(kernel_number / 2)) + top_shift - int(self.stride / 2)
        if top >= 0:
            pads[2] = top
        else:
            slices[0] = -top
        # bottom
        bottom_shift = 0 if (self.stride % 2) == 1 else -(kernel_number % 2)
        pads[3] = max(0, self.size - int(kernel_number / 2) + bottom_shift - ((input_size[2] - int(self.stride / 2) - 1) % self.stride))
        
        return slices, pads

    # general implementation of the hexagonal convolution
    def conv_with_arbitrary_stride(self, input):
        assert (input.size(2) - (self.stride // 2) >= 0), 'Too few rows to apply hex conv with this stide'
        odd_columns = None
        even_columns = None
        
        for i in range(self.size + 1):
            dilation = (1, 1) if i == 0 else (1, 2 * i)
            
            if not self.input_size_is_known:
                slices, pads = self.shape_for_odd_columns(input.size(), i)
                self.odd_columns_slices.append(slices)
                self.odd_columns_pads.append(pads)
                slices, pads = self.shape_for_even_columns(input.size(), i)
                self.even_columns_slices.append(slices)
                self.even_columns_pads.append(pads)
                if i == self.size:
                    self.input_size_is_known = True
                    
            if odd_columns is None:
                odd_columns = F.conv2d(nn.ZeroPad2d(tuple(self.odd_columns_pads[i]))(input[:, :,
                                                                                           self.odd_columns_slices[i][0]:self.odd_columns_slices[i][1],
                                                                                           self.odd_columns_slices[i][2]:self.odd_columns_slices[i][3]]),
                                       getattr(self, 'kernel' + str(i)), dilation=dilation,
                                       stride=(self.stride, 2 * self.stride), bias=self.bias)
            else:
                odd_columns += F.conv2d(nn.ZeroPad2d(tuple(self.odd_columns_pads[i]))(input[:, :,
                                                                                            self.odd_columns_slices[i][0]:self.odd_columns_slices[i][1],
                                                                                            self.odd_columns_slices[i][2]:self.odd_columns_slices[i][3]]),
                                        getattr(self, 'kernel' + str(i)), dilation=dilation,
                                        stride=(self.stride, 2 * self.stride))
        
            if even_columns is None:
                even_columns = F.conv2d(nn.ZeroPad2d(tuple(self.even_columns_pads[i]))(input[:, :,
                                                                                             self.even_columns_slices[i][0]:self.even_columns_slices[i][1],
                                                                                             self.even_columns_slices[i][2]:self.even_columns_slices[i][3]]),
                                        getattr(self, 'kernel' + str(i)), dilation=dilation,
                                        stride=(self.stride, 2 * self.stride), bias=self.bias)
            else:
                even_columns += F.conv2d(nn.ZeroPad2d(tuple(self.even_columns_pads[i]))(input[:, :,
                                                                                              self.even_columns_slices[i][0]:self.even_columns_slices[i][1],
                                                                                              self.even_columns_slices[i][2]:self.even_columns_slices[i][3]]),
                                         getattr(self, 'kernel' + str(i)), dilation=dilation,
                                         stride=(self.stride, 2 * self.stride))
                        
        concatenated_columns = torch.cat((odd_columns, even_columns), 3)
    
        n_odd_columns = odd_columns.size(3)
        n_even_columns = even_columns.size(3)
        if n_odd_columns == n_even_columns:
            iter = [int(i + x * n_even_columns) for i in range(n_even_columns) for x in range(2)]
        else:
            iter = [int(i + x * n_odd_columns) for i in range(n_even_columns) for x in range(2)]
            iter.append(n_even_columns)
            
        return concatenated_columns[:, :, :, iter]

    # a slightly faster, case specific implementation of the hexagonal convolution
    def conv_with_single_stride(self, input):
        columns_mod2 = input.size(3) % 2
        odd_kernels_odd_columns = []
        odd_kernels_even_columns = []
        even_kernels_all_columns = []
        
        even_kernels_all_columns = F.conv2d(nn.ZeroPad2d((0, 0, self.size, self.size))(input),
                                            self.kernel0, stride=(1, 1), bias=self.bias)
        if self.size >= 1:
            x = self.size
            odd_kernels_odd_columns = F.conv2d(nn.ZeroPad2d((1, columns_mod2, self.size, self.size - 1))(input),
                                                            self.kernel1, dilation=(1, 2), stride=(1, 2))
            odd_kernels_even_columns = F.conv2d(nn.ZeroPad2d((0, 1 - columns_mod2, self.size - 1, self.size))(input),
                                                self.kernel1, dilation=(1, 2), stride=(1, 2))

        if self.size > 1:
            for i in range(2, self.size + 1):
                if i % 2 == 0:
                    even_kernels_all_columns += F.conv2d(nn.ZeroPad2d((i, i, self.size - int(i / 2), self.size - int(i / 2)))(input),
                                                         getattr(self, 'kernel' + str(i)), dilation=(1, 2 * i), stride=(1, 1))
                else:
                    x = self.size + int((1 - i) / 2)
                    odd_kernels_odd_columns += F.conv2d(nn.ZeroPad2d((i, i - 1 + columns_mod2, x, x - 1))(input),
                                                        getattr(self, 'kernel' + str(i)), dilation=(1, 2 * i), stride=(1, 2))
                    odd_kernels_even_columns += F.conv2d(nn.ZeroPad2d((i - 1, i - columns_mod2, x - 1, x))(input),
                                                         getattr(self, 'kernel' + str(i)), dilation=(1, 2 * i), stride=(1, 2))
    
        odd_kernels_concatenated_columns = torch.cat((odd_kernels_odd_columns, odd_kernels_even_columns), 3)
        
        n_odd_columns = odd_kernels_odd_columns.size(3)
        n_even_columns = odd_kernels_even_columns.size(3)
        if n_odd_columns == n_even_columns:
            iter = [int(i + x * n_even_columns) for i in range(n_even_columns) for x in range(2)]
        else:
            iter = [int(i + x * n_odd_columns) for i in range(n_even_columns) for x in range(2)]
            iter.append(n_even_columns)

        return even_kernels_all_columns + odd_kernels_concatenated_columns[:, :, :, iter]

    def forward(self, input):
        if self.stride == 1:
            return self.conv_with_single_stride(input)
        else:
            return self.conv_with_arbitrary_stride(input)
        
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={size}'
             ', stride={stride}')
        if self.usebias is False:
            s += ', bias=False'
        if self.debug is True:
            s += ', debug=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class maxpool2d(nn.Module):
    r"""Applies a 2D hexagonal max pooling`
        
        Args:
        size:               radius of the hexagonal pooling kernel
        stride:             stride along symmetry axes of pooled pixels
        
        Attributes:
        kernel_(i):         subkernels of size (1 + 2 * size - i, 1 if i==0 else 2)
        
        
        Examples::
        
        >>> m = hexagdly.maxpool2d(1,2)
        >>> input = torch.randn(1, 1, 4, 2)
        >>> output = m(input)
        >>> print(output)
        """
    
    def __init__(self, size=1, stride=1):
        super(maxpool2d, self).__init__()
        self.size = size
        self.stride = stride
        self.input_size_is_known = False
        self.odd_columns_slices = []
        self.odd_columns_pads = []
        self.even_columns_slices = []
        self.even_columns_pads = []
        
        for i in range(self.size + 1):
            setattr(self, 'kernel' + str(i), (1 + 2 * self.size - i, 1 if i==0 else 2))

    def shape_for_odd_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        pads[0] = kernel_number
        # right
        pads[1] = max(0, kernel_number - ((input_size[3] - 1) % (2 * self.stride)))
        # top
        pads[2] = self.size - int(kernel_number / 2)
        # bottom
        constraint = input_size[2] - 1 - int((input_size[2] - 1 - int(self.stride / 2)) / self.stride) * self.stride
        bottom = (self.size - int((kernel_number + 1) / 2)) - constraint
        if bottom >= 0:
            pads[3] = bottom
        else:
            slices[1] = bottom
        
        return slices, pads
    
    def shape_for_even_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        left = kernel_number - self.stride
        if left >= 0:
            pads[0] = left
        else:
            slices[2] = -left
        # right
        pads[1] = max(0, kernel_number - ((input_size[3] - 1 - self.stride) % (2 * self.stride)))
        # top
        top_shift = -(kernel_number % 2) if (self.stride % 2) == 1 else 0
        top = (self.size - int(kernel_number / 2)) + top_shift - int(self.stride / 2)
        if top >= 0:
            pads[2] = top
        else:
            slices[0] = -top
        # bottom
        bottom_shift = 0 if (self.stride % 2) == 1 else -(kernel_number % 2)
        pads[3] = max(0, self.size - int(kernel_number / 2) + bottom_shift - ((input_size[2] - int(self.stride / 2) - 1) % self.stride))
        
        return slices, pads

    def forward(self, input):
        assert (input.size(2) - (self.stride // 2) >= 0), 'Too few rows to apply hex pooling with this stide'
        odd_columns = None
        even_columns = None
        
        for i in range(self.size + 1):
            dilation = (1, 1) if i == 0 else (1, 2 * i)
            if not self.input_size_is_known:
                slices, pads = self.shape_for_odd_columns(input.size(), i)
                self.odd_columns_slices.append(slices)
                self.odd_columns_pads.append(pads)
                slices, pads = self.shape_for_even_columns(input.size(), i)
                self.even_columns_slices.append(slices)
                self.even_columns_pads.append(pads)
                if i == self.size:
                    self.input_size_is_known = True
        
            if odd_columns is None:
                odd_columns = F.max_pool2d(nn.ZeroPad2d(tuple(self.odd_columns_pads[i]))(input[:, :,
                                                                                               self.odd_columns_slices[i][0]:self.odd_columns_slices[i][1],
                                                                                               self.odd_columns_slices[i][2]:self.odd_columns_slices[i][3]]),
                                           getattr(self, 'kernel' + str(i)), dilation=dilation,
                                           stride=(self.stride, 2 * self.stride))
            else:
                odd_columns = torch.max(odd_columns,
                                        F.max_pool2d(nn.ZeroPad2d(tuple(self.odd_columns_pads[i]))(input[:, :,
                                                                                                         self.odd_columns_slices[i][0]:self.odd_columns_slices[i][1],
                                                                                                         self.odd_columns_slices[i][2]:self.odd_columns_slices[i][3]]),
                                                     getattr(self, 'kernel' + str(i)), dilation=dilation,
                                                     stride=(self.stride, 2 * self.stride)))
                                        
            if even_columns is None:
                even_columns = F.max_pool2d(nn.ZeroPad2d(tuple(self.even_columns_pads[i]))(input[:, :,
                                                                                                 self.even_columns_slices[i][0]:self.even_columns_slices[i][1],
                                                                                                 self.even_columns_slices[i][2]:self.even_columns_slices[i][3]]),
                                            getattr(self, 'kernel' + str(i)), dilation=dilation,
                                            stride=(self.stride, 2 * self.stride))
            else:
                even_columns = torch.max(even_columns,
                                         F.max_pool2d(nn.ZeroPad2d(tuple(self.even_columns_pads[i]))(input[:, :,
                                                                                                           self.even_columns_slices[i][0]:self.even_columns_slices[i][1],
                                                                                                           self.even_columns_slices[i][2]:self.even_columns_slices[i][3]]),
                                                      getattr(self, 'kernel' + str(i)), dilation=dilation,
                                                      stride=(self.stride, 2 * self.stride)))
            
        concatenated_columns = torch.cat((odd_columns, even_columns), 3)
                
        n_odd_columns = odd_columns.size(3)
        n_even_columns = even_columns.size(3)
        if n_odd_columns == n_even_columns:
            iter = [int(i + x * n_even_columns) for i in range(n_even_columns) for x in range(2)]
        else:
            iter = [int(i + x * n_odd_columns) for i in range(n_even_columns) for x in range(2)]
            iter.append(n_even_columns)

        return concatenated_columns[:, :, :, iter]
    
    def __repr__(self):
        s = ('{name}(kernel_size={size}'
             ', stride={stride})')
        return s.format(name=self.__class__.__name__, **self.__dict__)
