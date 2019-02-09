import numpy as np
import torch
import hexagdly as hex
import pytest


class TestMaxPool2d(object):
    def get_array(self):
        return np.array(
            [[j * 5 + 1 + i for j in range(8)] for i in range(5)], dtype=np.float32
        )

    def get_array_maxpool2d_size1_stride1(self):
        return np.array(
            [
                [6, 12, 16, 22, 26, 32, 36, 37],
                [7, 13, 17, 23, 27, 33, 37, 38],
                [8, 14, 18, 24, 28, 34, 38, 39],
                [9, 15, 19, 25, 29, 35, 39, 40],
                [10, 15, 20, 25, 30, 35, 40, 40],
            ],
            dtype=np.float32,
        )

    def get_array_maxpool2d_size2_stride1(self):
        return np.array(
            [
                [12, 17, 22, 27, 32, 37, 37, 38],
                [13, 18, 23, 28, 33, 38, 38, 39],
                [14, 19, 24, 29, 34, 39, 39, 40],
                [15, 20, 25, 30, 35, 40, 40, 40],
                [15, 20, 25, 30, 35, 40, 40, 40],
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

    def get_tensors(self, in_channels, kernel_size, stride):
        channel_dist = 1000

        # input tensor
        array = self.get_array()
        array = np.expand_dims(
            np.stack([j * channel_dist + array for j in range(in_channels)]), 0
        )
        tensor = torch.FloatTensor(array)

        # expected output tensor
        if kernel_size == 1:
            pooled_array = self.get_array_maxpool2d_size1_stride1()
        elif kernel_size == 2:
            pooled_array = self.get_array_maxpool2d_size2_stride1()
        if stride == 2:
            pooled_array = self.get_array_stride_2(pooled_array)
        elif stride == 3:
            pooled_array = self.get_array_stride_3(pooled_array)
        pooled_array = np.expand_dims(
            np.stack(
                [
                    channel * channel_dist + pooled_array
                    for channel in range(in_channels)
                ]
            ),
            0,
        )
        pooled_tensor = torch.FloatTensor(pooled_array)

        # output tensor of test method
        maxpool2d = hex.MaxPool2d(kernel_size, stride)

        return maxpool2d(tensor), pooled_tensor

    def test_in_channels_1_kernel_size_1_stride_1(self):
        in_channels = 1
        kernel_size = 1
        stride = 1

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_1_stride_2(self):
        in_channels = 1
        kernel_size = 1
        stride = 2

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_1_stride_3(self):
        in_channels = 1
        kernel_size = 1
        stride = 3

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_2_stride_1(self):
        in_channels = 1
        kernel_size = 2
        stride = 1

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_2_stride_2(self):
        in_channels = 1
        kernel_size = 2
        stride = 2

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_2_stride_3(self):
        in_channels = 1
        kernel_size = 2
        stride = 3

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_1_stride_1(self):
        in_channels = 5
        kernel_size = 1
        stride = 1

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_1_stride_2(self):
        in_channels = 5
        kernel_size = 1
        stride = 2

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_1_stride_3(self):
        in_channels = 5
        kernel_size = 1
        stride = 3

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_2_stride_1(self):
        in_channels = 5
        kernel_size = 2
        stride = 1

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_2_stride_2(self):
        in_channels = 5
        kernel_size = 2
        stride = 2

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_2_stride_3(self):
        in_channels = 5
        kernel_size = 2
        stride = 3

        test_ouput, expectation = self.get_tensors(in_channels, kernel_size, stride)

        assert torch.equal(test_ouput, expectation)
