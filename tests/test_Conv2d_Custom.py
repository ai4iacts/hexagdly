import numpy as np
import torch
import hexagdly as hex
import pytest


class TestConv2d(object):
    def get_in_array(self):
        return np.array(
            [
                [
                    [
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 1],
                    ]
                ]
            ],
            dtype=np.float32,
        )

    def get_kernel_1_stride_1_array(self):
        return np.array(
            [
                [
                    [
                        [0, 1, 1, 2, 1, 1],
                        [0, 1, 1, 2, 2, 1],
                        [0, 1, 2, 2, 1, 2],
                        [0, 1, 1, 1, 2, 1],
                    ]
                ]
            ],
            dtype=np.float32,
        )

    def get_kernel_1_stride_2_array(self):
        return np.array([[[[0, 1, 1], [0, 1, 1]]]], dtype=np.float32)

    def get_kernel_1_stride_3_array(self):
        return np.array([[[[0, 2]]]], dtype=np.float32)

    def get_kernel_2_stride_1_array(self):
        return np.array(
            [
                [
                    [
                        [1, 1, 2, 3, 3, 2],
                        [1, 2, 4, 4, 3, 3],
                        [2, 2, 3, 4, 5, 2],
                        [1, 1, 3, 3, 3, 2],
                    ]
                ]
            ],
            dtype=np.float32,
        )

    def get_kernel_2_stride_2_array(self):
        return np.array([[[[1, 4, 3], [2, 3, 5]]]], dtype=np.float32)

    def get_kernel_2_stride_3_array(self):
        return np.array([[[[1, 4]]]], dtype=np.float32)

    def get_tensors(self, in_channels, kernel_size, stride, bias_bool):
        channel_dist = 1000
        if bias_bool is False:
            bias_value = 0
            bias = None
        else:
            bias_value = 1.0
            bias = np.array([1])

        # input tensor
        array = self.get_in_array()
        array = np.concatenate(
            [channel * channel_dist * array + array for channel in range(in_channels)],
            1,
        )
        tensor = torch.FloatTensor(array)

        # expected output tensor
        convolved_array = getattr(
            self, "get_kernel_" + str(kernel_size) + "_stride_" + str(stride) + "_array"
        )()
        convolved_array = np.sum(
            np.stack(
                [
                    (channel * channel_dist) * convolved_array + convolved_array
                    for channel in range(in_channels)
                ]
            ),
            0,
        )
        convolved_tensor = torch.FloatTensor(convolved_array) + bias_value

        # output tensor of test method
        if kernel_size == 1:
            kernel = [np.ones((1, in_channels, 3, 1)), np.ones((1, in_channels, 2, 2))]
        elif kernel_size == 2:
            kernel = [
                np.ones((1, in_channels, 5, 1)),
                np.ones((1, in_channels, 4, 2)),
                np.ones((1, in_channels, 3, 2)),
            ]
        conv2d = hex.Conv2d_CustomKernel(kernel, stride, bias)

        return conv2d(tensor), convolved_tensor

    def test_in_channels_1_kernel_size_1_stride_1_bias_False(self):
        in_channels = 1
        kernel_size = 1
        stride = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_1_stride_2_bias_False(self):
        in_channels = 1
        kernel_size = 1
        stride = 2
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_1_stride_3_bias_False(self):
        in_channels = 1
        kernel_size = 1
        stride = 3
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_2_stride_1_bias_False(self):
        in_channels = 1
        kernel_size = 2
        stride = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_2_stride_2_bias_False(self):
        in_channels = 1
        kernel_size = 2
        stride = 2
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_2_stride_3_bias_False(self):
        in_channels = 1
        kernel_size = 2
        stride = 3
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_1_stride_1_bias_False(self):
        in_channels = 5
        kernel_size = 1
        stride = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_1_stride_2_bias_False(self):
        in_channels = 5
        kernel_size = 1
        stride = 2
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_1_stride_3_bias_False(self):
        in_channels = 5
        kernel_size = 1
        stride = 3
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_2_stride_1_bias_False(self):
        in_channels = 5
        kernel_size = 2
        stride = 1
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_2_stride_2_bias_False(self):
        in_channels = 5
        kernel_size = 2
        stride = 2
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_2_stride_3_bias_False(self):
        in_channels = 5
        kernel_size = 2
        stride = 3
        bias = False

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_1_stride_1_bias_True(self):
        in_channels = 1
        kernel_size = 1
        stride = 1
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_1_stride_2_bias_True(self):
        in_channels = 1
        kernel_size = 1
        stride = 2
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_1_stride_3_bias_True(self):
        in_channels = 1
        kernel_size = 1
        stride = 3
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_2_stride_1_bias_True(self):
        in_channels = 1
        kernel_size = 2
        stride = 1
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_2_stride_2_bias_True(self):
        in_channels = 1
        kernel_size = 2
        stride = 2
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_1_kernel_size_2_stride_3_bias_True(self):
        in_channels = 1
        kernel_size = 2
        stride = 3
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_1_stride_1_bias_True(self):
        in_channels = 5
        kernel_size = 1
        stride = 1
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_1_stride_2_bias_True(self):
        in_channels = 5
        kernel_size = 1
        stride = 2
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_1_stride_3_bias_True(self):
        in_channels = 5
        kernel_size = 1
        stride = 3
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_2_stride_1_bias_True(self):
        in_channels = 5
        kernel_size = 2
        stride = 1
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_2_stride_2_bias_True(self):
        in_channels = 5
        kernel_size = 2
        stride = 2
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)

    def test_in_channels_5_kernel_size_2_stride_3_bias_True(self):
        in_channels = 5
        kernel_size = 2
        stride = 3
        bias = True

        test_ouput, expectation = self.get_tensors(
            in_channels, kernel_size, stride, bias
        )

        assert torch.equal(test_ouput, expectation)
