import numpy as np
import torch
import hexagdly as hex
import pytest


class TestConv3d(object):
    def get_array(self):
        return np.array(
            [[j * 5 + 1 + i for j in range(8)] for i in range(5)], dtype=np.float32
        )

    def get_array_conv2d_size1_stride1(self):
        return np.array(
            [
                [9, 39, 45, 99, 85, 159, 125, 136],
                [19, 51, 82, 121, 152, 191, 222, 176],
                [24, 58, 89, 128, 159, 198, 229, 181],
                [29, 65, 96, 135, 166, 205, 236, 186],
                [28, 39, 87, 79, 147, 119, 207, 114],
            ],
            dtype=np.float32,
        )

    def get_array_conv2d_size2_stride1(self):
        return np.array(
            [
                [42, 96, 128, 219, 238, 349, 265, 260],
                [67, 141, 194, 312, 354, 492, 388, 361],
                [84, 162, 243, 346, 433, 536, 494, 408],
                [90, 145, 246, 302, 426, 462, 474, 343],
                [68, 104, 184, 213, 314, 323, 355, 245],
            ],
            dtype=np.float32,
        )

    def get_array_stride_2(self, array_stride_1):
        array_stride_2 = np.zeros((2, 4), dtype=np.float32)
        stride_2_pos = [
            (0, 0, 0, 0),
            (0, 1, 1, 2),
            (0, 2, 0, 4),
            (0, 3, 1, 6),
            (1, 0, 2, 0),
            (1, 1, 3, 2),
            (1, 2, 2, 4),
            (1, 3, 3, 6),
        ]
        for pos in stride_2_pos:
            array_stride_2[pos[0], pos[1]] = array_stride_1[pos[2], pos[3]]
        return array_stride_2

    def get_array_stride_3(self, array_stride_1):
        array_stride_3 = np.zeros((2, 3), dtype=np.float32)
        stride_3_pos = [
            (0, 0, 0, 0),
            (0, 1, 1, 3),
            (0, 2, 0, 6),
            (1, 0, 3, 0),
            (1, 1, 4, 3),
            (1, 2, 3, 6),
        ]
        for pos in stride_3_pos:
            array_stride_3[pos[0], pos[1]] = array_stride_1[pos[2], pos[3]]
        return array_stride_3

    def get_n_neighbors_size1(self):
        return np.array(
            [
                [3, 6, 4, 6, 4, 6, 4, 4],
                [5, 7, 7, 7, 7, 7, 7, 5],
                [5, 7, 7, 7, 7, 7, 7, 5],
                [5, 7, 7, 7, 7, 7, 7, 5],
                [4, 4, 6, 4, 6, 4, 6, 3],
            ],
            dtype=np.float32,
        )

    def get_n_neighbors_size2(self):
        return np.array(
            [
                [7, 11, 11, 13, 11, 13, 9, 8],
                [10, 15, 16, 18, 16, 18, 13, 11],
                [12, 16, 19, 19, 19, 19, 16, 12],
                [11, 13, 18, 16, 18, 16, 15, 10],
                [8, 9, 13, 11, 13, 11, 11, 7],
            ],
            dtype=np.float32,
        )

    def get_tensors(
        self,
        in_channels,
        depth,
        kernel_size_depth,
        kernel_size_hex,
        stride_depth,
        stride_hex,
        bias_bool,
    ):
        channel_dist = 1000
        depth_dist = 40
        depth_steps = int(np.ceil((depth - kernel_size_depth + 1) / stride_depth))
        if bias_bool is False:
            bias_value = 0
            bias = None
        else:
            bias_value = 1.0
            bias = np.array([1])

        # input tensor
        array = self.get_array()
        array = np.expand_dims(
            np.stack(
                [
                    j * channel_dist
                    + np.stack([i * depth_dist + array for i in range(depth)])
                    for j in range(in_channels)
                ]
            ),
            0,
        )
        tensor = torch.FloatTensor(array)

        # expected output tensor
        if kernel_size_hex == 1:
            conv2d_array = self.get_array_conv2d_size1_stride1()
            n_neighbours = self.get_n_neighbors_size1()
        elif kernel_size_hex == 2:
            conv2d_array = self.get_array_conv2d_size2_stride1()
            n_neighbours = self.get_n_neighbors_size2()
        convolved_array = []
        for dstep in range(depth_steps):
            layer_array = np.sum(
                np.stack(
                    [
                        (
                            channel * channel_dist
                            + ((dstep * stride_depth) + dsize) * depth_dist
                        )
                        * n_neighbours
                        + conv2d_array
                        for dsize in range(kernel_size_depth)
                        for channel in range(in_channels)
                    ]
                ),
                0,
            )
            if stride_hex == 2:
                layer_array = self.get_array_stride_2(layer_array)
            elif stride_hex == 3:
                layer_array = self.get_array_stride_3(layer_array)
            convolved_array.append(layer_array)
        convolved_array = np.expand_dims(
            np.expand_dims(np.stack(convolved_array), 0), 0
        )
        convolved_tensor = torch.FloatTensor(convolved_array) + bias_value

        # output tensor of test method
        if kernel_size_hex == 1:
            kernel = [
                np.ones((1, in_channels, kernel_size_depth, 3, 1)),
                np.ones((1, in_channels, kernel_size_depth, 2, 2)),
            ]
        elif kernel_size_hex == 2:
            kernel = [
                np.ones((1, in_channels, kernel_size_depth, 5, 1)),
                np.ones((1, in_channels, kernel_size_depth, 4, 2)),
                np.ones((1, in_channels, kernel_size_depth, 3, 2)),
            ]

        conv3d = hex.Conv3d_CustomKernel(kernel, (stride_depth, stride_hex), bias)

        return conv3d(tensor), convolved_tensor

    def test_in_channels_1_depth_1_kernel_size_1_1_stride_1_1_bias_False(self):
        in_channels = 1
        depth = 1
        kernel_size_depth = 1
        kernel_size_hex = 1
        stride_depth = 1
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_1_kernel_size_1_1_stride_1_2_bias_False(self):
        in_channels = 1
        depth = 1
        kernel_size_depth = 1
        kernel_size_hex = 1
        stride_depth = 1
        stride_hex = 2
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_1_kernel_size_1_1_stride_1_3_bias_False(self):
        in_channels = 1
        depth = 1
        kernel_size_depth = 1
        kernel_size_hex = 1
        stride_depth = 1
        stride_hex = 3
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_1_kernel_size_1_2_stride_1_1_bias_False(self):
        in_channels = 1
        depth = 1
        kernel_size_depth = 1
        kernel_size_hex = 2
        stride_depth = 1
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_1_kernel_size_1_2_stride_1_2_bias_False(self):
        in_channels = 1
        depth = 1
        kernel_size_depth = 1
        kernel_size_hex = 2
        stride_depth = 1
        stride_hex = 2
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_1_kernel_size_1_2_stride_1_3_bias_False(self):
        in_channels = 1
        depth = 1
        kernel_size_depth = 1
        kernel_size_hex = 2
        stride_depth = 1
        stride_hex = 3
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_9_kernel_size_1_1_stride_1_1_bias_False(self):
        in_channels = 1
        depth = 9
        kernel_size_depth = 1
        kernel_size_hex = 1
        stride_depth = 1
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_9_kernel_size_1_1_stride_2_1_bias_False(self):
        in_channels = 1
        depth = 9
        kernel_size_depth = 1
        kernel_size_hex = 1
        stride_depth = 2
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_9_kernel_size_1_1_stride_3_1_bias_False(self):
        in_channels = 1
        depth = 9
        kernel_size_depth = 1
        kernel_size_hex = 1
        stride_depth = 3
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_9_kernel_size_2_1_stride_1_1_bias_False(self):
        in_channels = 1
        depth = 9
        kernel_size_depth = 2
        kernel_size_hex = 1
        stride_depth = 1
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_9_kernel_size_2_1_stride_2_1_bias_False(self):
        in_channels = 1
        depth = 9
        kernel_size_depth = 2
        kernel_size_hex = 1
        stride_depth = 2
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_9_kernel_size_2_1_stride_2_2_bias_False(self):
        in_channels = 1
        depth = 9
        kernel_size_depth = 2
        kernel_size_hex = 1
        stride_depth = 2
        stride_hex = 2
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_9_kernel_size_7_2_stride_1_1_bias_False(self):
        in_channels = 1
        depth = 9
        kernel_size_depth = 7
        kernel_size_hex = 2
        stride_depth = 1
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_9_kernel_size_7_2_stride_1_2_bias_False(self):
        in_channels = 1
        depth = 9
        kernel_size_depth = 7
        kernel_size_hex = 2
        stride_depth = 1
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_depth_9_kernel_size_7_2_stride_2_2_bias_False(self):
        in_channels = 1
        depth = 9
        kernel_size_depth = 7
        kernel_size_hex = 2
        stride_depth = 2
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_depth_9_kernel_size_3_2_stride_1_1_bias_False(self):
        in_channels = 5
        depth = 9
        kernel_size_depth = 7
        kernel_size_hex = 2
        stride_depth = 1
        stride_hex = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_depth_9_kernel_size_3_2_stride_1_1_bias_True(self):
        in_channels = 5
        depth = 9
        kernel_size_depth = 7
        kernel_size_hex = 2
        stride_depth = 1
        stride_hex = 1
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels,
            depth,
            kernel_size_depth,
            kernel_size_hex,
            stride_depth,
            stride_hex,
            bias,
        )

        assert torch.equal(test_ouput, expectation)
