""" 
HexagDLy utilities for illustrative examples.

"""

import numpy as np
import numpy.linalg as LA
from scipy.interpolate import griddata
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim.lr_scheduler as scheduler
import os
import matplotlib.pyplot as plt
import time


def put_shape(nx, ny, cx, cy, params):
    d = np.zeros((nx, ny))
    i = np.indices((nx, ny))
    i[0] = i[0] - cx
    i[1] = i[1] - cy
    i = i.astype(float)
    i[0] *= 1.73205 / 2
    if np.mod(cx, 2) == 0:
        i[1][np.mod(cx + 1, 2) :: 2] += 0.5
    else:
        i[1][np.mod(cx + 1, 2) :: 2] -= 0.5
    di = i[0] ** 2 + i[1] ** 2
    for t1, t2 in params:
        di = np.where(np.logical_and(di >= t2, di <= t1), 1, di)
    di = np.where(di > 1.1, 0, di)
    return di.transpose()


class toy_data:
    r"""Object that contains a set of toy images of randomly scattered 
        hexagonal shapes of a certain kind.

        Args:
        shape:          str, choose from ...
        nx:             int, dimension in x
        ny:             int, dimension in y
        nchannels:      int, number of input channels ('colour' channels)
        nexamples:      int, number of images
        px:             int, center row for shape
        py:             int, center column for shape

    """

    def __init__(
        self,
        shape,
        nx=16,
        ny=16,
        nchannels=1,
        nexamples=1,
        noisy=None,
        px=None,
        py=None,
    ):
        self.shapes = {
            "small_hexagon": [(1, 0)],
            "medium_hexagon": [(4, 0)],
            "snowflake_1": [(3, 0)],
            "snowflake_2": [(1, 0), (4.1, 3.9)],
            "snowflake_3": [(7, 3)],
            "snowflake_4": [(7, 0)],
            "double_hex": [(10, 5)],
        }
        self.nx = nx
        self.ny = ny
        if noisy:
            self.image_data = np.random.normal(0, noisy, (nexamples, nchannels, ny, nx))
        else:
            self.image_data = np.zeros((nexamples, nchannels, ny, nx))
        for ie, example in enumerate(self.image_data):
            for ic, channel in enumerate(example):
                if not px and not py:
                    cx, cy = int(ny * np.random.random()), int(nx * np.random.random())
                else:
                    cx, cy = px, py
                face = put_shape(self.nx, self.ny, cx, cy, self.shapes[shape])
                self.image_data[ie, ic, :, :] += face

    def to_h5(self, filename):
        f = h5py.File(filename + ".h5", "w")
        f.create_dataset("image_data", data=self.image_data)

    def to_torch_tensor(self):
        return torch.Tensor(self.image_data)


###################################################################


class Shape(object):
    def __init__(self, nx, ny, scale=3, rotation=False):
        self.nx = nx
        self.ny = ny
        self.X = np.zeros(self.nx * self.ny)
        self.Y = np.zeros(self.nx * self.ny)
        i = 0
        for x in range(self.nx):
            for y in range(self.ny):
                self.X[i], self.Y[i] = x * np.sqrt(3) / 2, -(y + np.mod(x, 2) * 0.5)
                i += 1
        self.xmin = np.min(self.X)
        self.xmax = np.max(self.X)
        self.ymin = np.min(self.Y)
        self.ymax = np.max(self.Y)
        self.P = np.stack([self.X.flatten(), self.Y.flatten()], axis=1)
        self.size = 0.5
        self.scale = scale
        self.rotation = rotation

    def polar_to_cartesian(self, r, alpha):
        x = r * np.cos(alpha)
        y = r * np.sin(alpha)
        return np.array([x, y])

    def image_from_points(self, point_list_1, point_list_2):
        ind = np.full(len(self.P), False)
        for p1, p2 in zip(point_list_1, point_list_2):
            pa = p2 - p1
            alpha = np.arctan2(pa[1], pa[0])
            pb = self.P - p1
            beta = np.arctan2(pb[:, 1], pb[:, 0])
            vlen = LA.norm(pb, axis=1)
            dist = np.abs(self.polar_to_cartesian(vlen, beta - alpha)[1])

            tmp = np.where(dist < self.size, True, False)
            xmin = np.min([p1[0], p2[0]])
            xmax = np.max([p1[0], p2[0]])
            if np.abs(xmax - xmin) > 1e-12:
                xborder1 = np.where(self.P[:, 0] < xmin, False, True)
                xborder2 = np.where(self.P[:, 0] > xmax, False, True)
                xborder = np.logical_and(xborder1, xborder2)
            else:
                xborder = np.full(len(self.P), True)

            ymin = np.min([p1[1], p2[1]])
            ymax = np.max([p1[1], p2[1]])
            if np.abs(ymax - ymin) > 1e-12:
                yborder1 = np.where(self.P[:, 1] < ymin, False, True)
                yborder2 = np.where(self.P[:, 1] > ymax, False, True)
                yborder = np.logical_and(yborder1, yborder2)
            else:
                yborder = np.full(len(self.P), True)

            border = np.logical_and(xborder, yborder)
            tmp = np.logical_and(tmp, border)
            ind = np.logical_or(ind, tmp)
        return np.where(ind, 1, 0)

    def point_list_for_triangle(self, centre, rotation=0.0):
        a1, a2, a3 = -np.pi / 6, np.pi / 2, np.pi * 7 / 6
        P1 = self.polar_to_cartesian(self.scale, a1 + rotation) + centre
        P2 = self.polar_to_cartesian(self.scale, a2 + rotation) + centre
        P3 = self.polar_to_cartesian(self.scale, a3 + rotation) + centre
        return [P1, P2, P3], [P2, P3, P1]

    def point_list_for_square(self, centre, rotation=0.0):
        a1, a2, a3, a4 = np.pi / 4, np.pi * 3 / 4, -np.pi * 3 / 4, -np.pi / 4
        P1 = self.polar_to_cartesian(self.scale, a1 + rotation) + centre
        P2 = self.polar_to_cartesian(self.scale, a2 + rotation) + centre
        P3 = self.polar_to_cartesian(self.scale, a3 + rotation) + centre
        P4 = self.polar_to_cartesian(self.scale, a4 + rotation) + centre
        return [P1, P2, P3, P4], [P2, P3, P4, P1]

    def image_triangle(self, centre, rotation):
        p1, p2 = self.point_list_for_triangle(centre, rotation)
        return self.image_from_points(p1, p2)

    def image_square(self, centre, rotation):
        p1, p2 = self.point_list_for_square(centre, rotation)
        return self.image_from_points(p1, p2)

    def image_circle(self, centre):
        dist = np.abs(np.linalg.norm(self.P - centre, axis=1) - self.scale)
        return np.where(dist < self.size, 1, 0)

    def __call__(self, shape="circle"):
        x = self.xmin + (self.xmax - self.xmin) * np.random.rand()
        y = self.ymin + (self.ymax - self.ymin) * np.random.rand()
        if self.rotation:
            r = 2 * np.pi * np.random.rand()
        else:
            r = 0.0
        if shape == "circle":
            centre = np.array([[x, y]])
            return self.image_circle(centre).reshape((self.nx, self.ny)).T
        elif shape == "triangle":
            centre = np.array([x, y])
            return (
                self.image_triangle(centre, r + np.pi / 7.5)
                .reshape((self.nx, self.ny))
                .T
            )
        elif shape == "square":
            centre = np.array([x, y])
            return (
                self.image_square(centre, r + np.pi / 3).reshape((self.nx, self.ny)).T
            )
        else:
            return None


class toy_data2:
    r"""Object that contains a set of toy images of randomly scattered 
        hexagonal shapes of a certain kind.

        Args:
        shape:          str, choose from ...
        nx:             int, dimension in x
        ny:             int, dimension in y
        nchannels:      int, number of input channels ('colour' channels)
        nexamples:      int, number of images
        px:             int, center row for shape
        py:             int, center column for shape

    """

    def __init__(self, shape, nx=16, ny=16, nchannels=1, nexamples=1, noisy=None):
        self.nx = nx
        self.ny = ny
        self.shape = Shape(nx, ny, (nx + ny) / 6, True)
        if noisy:
            self.image_data = np.random.normal(0, noisy, (nexamples, nchannels, ny, nx))
        else:
            self.image_data = np.zeros((nexamples, nchannels, ny, nx))
        for ie, example in enumerate(self.image_data):
            for ic, channel in enumerate(example):
                self.image_data[ie, ic, :, :] += self.shape(shape)

    def to_h5(self, filename):
        f = h5py.File(filename + ".h5", "w")
        f.create_dataset("image_data", data=self.image_data)

    def to_torch_tensor(self):
        return torch.Tensor(self.image_data)


class toy_dataset:
    r"""Object that creates a data set containing different shapes

    Args:
    shapes:             list of strings with names of different shapes
    nperclass:          int, number of images of each shape
    nx:                 int, number of columns of pixels
    ny:                 int, number of rows of pixels
    nchannels:          int, number of channels for each image

    """

    def __init__(self, shapes, nperclass, nx=16, ny=16, nchannels=1, noisy=None):
        self.shapes = shapes
        self.image_data = np.zeros((len(shapes) * nperclass, nchannels, ny, nx))
        self.labels = np.zeros(len(shapes) * nperclass)
        self.nx = nx
        self.ny = ny
        self.nchannels = nchannels
        self.nperclass = nperclass
        self.noisy = noisy
        self.square_image_data = None
        self.square_benchmark = None

    def create(self):
        d = [
            toy_data(
                shape, self.nx, self.ny, self.nchannels, self.nperclass, self.noisy
            )
            for shape in self.shapes
        ]
        indices = np.arange(len(self.shapes) * self.nperclass)
        np.random.shuffle(indices)
        icount = 0
        for s, label in zip(d, np.arange(len(self.shapes), dtype=np.int)):
            for image in s.image_data:
                for ic, c in enumerate(image):
                    self.image_data[indices[icount], ic] = c
                self.labels[indices[icount]] = int(label)
                icount += 1

    def convert_to_square(self, scale=1, method="linear"):
        t0 = time.time()

        X = np.zeros(self.nx * self.ny)
        Y = np.zeros(self.nx * self.ny)
        i = 0
        for x in range(self.nx):
            for y in range(self.ny):
                X[i], Y[i] = x * np.sqrt(3) / 2, -(y + np.mod(x, 2) * 0.5)
                i += 1

        grid_x, grid_y = np.meshgrid(
            np.linspace(0, max(X), scale * self.nx),
            np.linspace(0, min(Y), scale * self.ny),
        )

        self.square_image_data = np.zeros(
            (
                len(self.shapes) * self.nperclass,
                self.nchannels,
                scale * self.ny,
                scale * self.nx,
            )
        )
        for ie, example in enumerate(self.image_data):
            for ic, image in enumerate(example):
                Z = image[:].flatten("F")
                tmp = griddata((X, Y), Z, (grid_x, grid_y), method=method)
                tmp -= np.nan_to_num(tmp).min()
                tmp /= np.nan_to_num(tmp).max()
                tmp = np.nan_to_num(tmp)
                self.square_image_data[ie, ic, :, :] += tmp
        self.square_benchmark = time.time() - t0

    def to_torch_tensor(self, sampling="hexagon"):
        if sampling == "square":
            return torch.Tensor(self.square_image_data)
        else:
            return torch.Tensor(self.image_data)

    def to_dataloader(self, batchsize=8, shuffle=True, sampling="hexagon"):
        if sampling == "square":
            assert (
                self.square_image_data is not None
            ), "No square images, please convert first!"
            image_data = self.square_image_data
        else:
            image_data = self.image_data
        data, label = torch.from_numpy(image_data), torch.from_numpy(self.labels)
        tensor_dataset = torch.utils.data.TensorDataset(data, label)
        dataloader = torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=max(1, os.sysconf("SC_NPROCESSORS_ONLN") // 2),
        )
        return dataloader


class model:
    r"""A toy model CNN

    Args:
    train_dataloader:   pytorch dataloader with training data
    val_dataloader:     pytorch dataloader with validation data
    net:                CNN model
    epochs:             int, number of epochs to train

    """

    def __init__(self, train_dataloader, val_dataloader, net, epochs=10):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.net = net
        self.epochs = epochs

    def train(self, lr=0.005):
        nbts = 16
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.net.parameters(), lr=lr, momentum=0.9, weight_decay=0.004
        )
        self.tepoch = []
        self.tloss = []
        self.taccu = []
        self.tlr = []
        self.vepoch = []
        self.vloss = []
        self.vaccu = []
        self.train_time = 0
        self.scheduler = scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            verbose=False,
            threshold=1,
            threshold_mode="abs",
            min_lr=1e-10,
        )
        for epoch in range(self.epochs):
            print("Epoch %d" % (epoch + 1))
            if torch.cuda.is_available():
                self.net = self.net.cuda()
            for dataloader, net_phase, phase in zip(
                [self.train_dataloader, self.train_dataloader, self.val_dataloader],
                ["train", "eval", "eval"],
                ["training", "train_lc", "val_lc"],
            ):
                if net_phase == "train":
                    t0 = time.time()
                num_batches = len(dataloader)
                running_loss = 0.0
                total = 0.0
                correct = 0.0
                batch_counter = 0.0
                getattr(self.net, net_phase)()
                for i, data in enumerate(dataloader, 0):
                    inputs, labels = data
                    inputs, labels = Variable(inputs).float(), Variable(labels).long()
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()
                    optimizer.zero_grad()
                    outputs = self.net(inputs)
                    tloss = criterion(outputs, labels)
                    tloss.backward()
                    optimizer.step()
                    running_loss += tloss.item()
                    total += outputs.data.size()[0]
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels.data).sum()
                    if i % nbts == nbts - 1:
                        current_epoch = epoch + (batch_counter + 1) / num_batches
                        current_lr = optimizer.param_groups[0]["lr"]
                        mean_loss = running_loss / nbts
                        mean_accuracy = 100 * correct.float() / total
                        print(
                            "epoch: %d (%.3f) %s - %5d batches -> mean loss: %.3f, lr: %.3f, mean acc.: %.2f %%"
                            % (
                                epoch + 1,
                                current_epoch,
                                phase,
                                i + 1,
                                mean_loss,
                                current_lr,
                                mean_accuracy,
                            )
                        )
                        running_loss = 0.0
                        total = 0.0
                        correct = 0.0
                        if phase == "train_lc":
                            self.tepoch.append(current_epoch)
                            self.tloss.append(mean_loss)
                            self.taccu.append(mean_accuracy)
                            self.tlr.append(current_lr)
                        elif phase == "val_lc":
                            self.vepoch.append(current_epoch)
                            self.vloss.append(mean_loss)
                            self.vaccu.append(mean_accuracy)
                            self.scheduler.step(mean_accuracy)
                    batch_counter += 1.0
                batch_counter = 0.0
                if net_phase == "train":
                    self.train_time += time.time() - t0
        self.train_time /= self.epochs

    def save_current(self):
        torch.save(
            self.net.state_dict(),
            str(self.net.__class__.__name__) + "_" + str(self.epochs) + ".ptmodel",
        )

    def load(self, filename):
        self.net.load_state_dict(torch.load(filename))

    def get_lc(self):
        return (
            np.array(self.tepoch),
            np.array(self.tloss),
            np.array(self.taccu),
            np.array(self.vepoch),
            np.array(self.vloss),
            np.array(self.vaccu),
            np.array(self.train_time),
        )

    def plot_lc(self, scale_to_time=False):
        fig = plt.figure("learning_curves", (7, 7))
        axa = fig.add_subplot(311)
        axb = fig.add_subplot(312)
        axc = fig.add_subplot(313)
        tx_axis = np.array(self.tepoch)
        vx_axis = np.array(self.vepoch)
        if scale_to_time:
            tx_axis *= self.train_time
            vx_axis *= self.train_time
        axa.plot(vx_axis, self.vaccu, "-", lw=1)
        axa.set_ylabel("accuracy [%]", size=15)
        axa.tick_params(
            axis="both",
            which="both",
            labelsize=10,
            bottom=False,
            top=False,
            labelbottom=False,
        )

        axb.plot(vx_axis, self.vloss, "-", label=self.net.name, lw=1)
        axb.legend()
        axb.set_ylabel("loss", size=15)
        axb.tick_params(
            axis="both",
            which="both",
            labelsize=10,
            bottom=False,
            top=False,
            labelbottom=False,
        )

        axc.plot(tx_axis, self.tlr, lw=1)
        axc.set_yscale("log")
        axc.set_ylabel("learning rate", size=15)
        if scale_to_time:
            axc.set_xlabel("train time [s]", size=15)
        else:
            axc.set_xlabel("# Epochs", size=15)
        axc.tick_params(
            axis="both",
            which="both",
            labelsize=10,
            bottom=True,
            top=True,
            labelbottom=True,
        )
        fig.canvas.draw()
        plt.show()
