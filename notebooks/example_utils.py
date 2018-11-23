""" 
HexagDLy utilities for illustrative examples.

"""

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import h5py
import os
import matplotlib.pyplot as plt
import logging
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def put_shape(nx, ny, cx, cy, params):
    d=np.zeros((nx, ny))
    i = np.indices((nx, ny))
    i[0]=i[0]-cx
    i[1]=i[1]-cy
    i = i.astype(float)
    i[0]*=1.73205/2
    if np.mod(cx,2)==0:
        i[1][np.mod(cx+1,2)::2] += 0.5
    else:
        i[1][np.mod(cx+1,2)::2] -= 0.5
    di = i[0]**2+i[1]**2
    for t1, t2 in params:
        di = np.where(np.logical_and(di>=t2, di<=t1), 1, di)
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

    def __init__(self, shape, nx=16, ny=16, nchannels=1, nexamples=1, px=None, py=None):
        self.shapes = {'small_hexagon':[(1, 0)], 'medium_hexagon':[(4, 0)], 'snowflake_1':[(3, 0)], 'snowflake_2': [(1,0), (4.1, 3.9)], 'snowflake_3':[(7, 3)], 'snowflake_4':[(7, 0)], 'double_hex':[(10, 5)]}
        self.nx = nx
        self.ny = ny
        self.image_data = np.zeros((nexamples, nchannels, ny, nx))
        for ie, example in enumerate(self.image_data):
            for ic, channel in enumerate(example):
                if not px and not py:
                    cx, cy = int(ny*np.random.random()), int(nx*np.random.random())
                else:
                    cx, cy = px, py
                face = put_shape(self.nx, self.ny, cx, cy, self.shapes[shape])
                self.image_data[ie, ic, :, :] += face

    def to_h5(self, filename):
        f = h5py.File(filename+".h5", "w")
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
    def __init__(self, shapes, nperclass, nx=16, ny=16, nchannels=1):
        self.shapes = shapes
        self.image_data = np.zeros((len(shapes)*nperclass, nchannels, ny, nx))
        self.labels = np.zeros(len(shapes)*nperclass)
        self.nx = nx
        self.ny = ny
        self.nchannels = nchannels
        self.nperclass = nperclass

    def create(self):
        d = [toy_data(shape, self.nx, self.ny, self.nchannels, self.nperclass) for shape in self.shapes]
        indices = np.arange(len(self.shapes)*self.nperclass)
        np.random.shuffle(indices)
        icount = 0
        for s, label in zip(d, np.arange(len(self.shapes), dtype=np.int)):
            for image in s.image_data:
                for ic, c in enumerate(image):
                    self.image_data[indices[icount], ic] = c
                self.labels[indices[icount]] = int(label)
                icount += 1

    def to_h5(self, filename):
        with h5py.File(filename+".h5", "w") as f:
            f.create_dataset("data", data=self.image_data)
            f.create_dataset("label", data=self.labels)

    def to_torch_tensor(self):
        return torch.Tensor(self.image_data)

    def to_dataloader(self, batchsize=8, shuffle=True):
        data, label = torch.from_numpy(self.image_data), torch.from_numpy(self.labels)
        tensor_dataset = torch.utils.data.TensorDataset(data, label)
        dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batchsize, shuffle=shuffle, num_workers=max(1, os.sysconf("SC_NPROCESSORS_ONLN")//2))
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
        logfilename = self.net.__class__.__name__+'_progress.log'
        self.logger = logging.getLogger(self.net.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        filehandler = logging.FileHandler(logfilename, 'w')
        filehandler.setFormatter(formatter)
        filehandler.setLevel(logging.INFO)
        self.logger.addHandler(filehandler)
        self.logger.addHandler(logging.StreamHandler())

    def train(self):
        nbts = 16
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.004)
        self.tepoch = []
        self.tloss = []
        self.taccu = []
        self.tlr = []
        self.vepoch = []
        self.vloss = []
        self.vaccu = []
        for epoch in range(self.epochs):
            self.logger.info('Epoch %d' %(epoch+1))
            if torch.cuda.is_available():
                self.net = self.net.cuda()
            for dataloader, net_phase, phase in zip([self.train_dataloader, self.train_dataloader, self.val_dataloader], \
                                         ['train', 'eval', 'eval'], ['training', 'train_lc', 'val_lc']):
                num_batches = len(dataloader)
                running_loss = 0.
                total = 0.
                correct = 0.
                batch_counter = 0.
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
                    running_loss += tloss.data[0]
                    total += outputs.data.size()[0]
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels.data).sum()
                    if i % nbts == nbts-1:
                        current_epoch = epoch+(batch_counter+1)/num_batches
                        current_lr = optimizer.param_groups[0]['lr']
                        mean_loss = running_loss / nbts
                        mean_accuracy = 100 * correct.float() / total
                        self.logger.info('epoch: %d (%.3f) %s - %5d batches -> mean loss: %.3f, lr: %.3f, mean acc.: %.2f %%' %(epoch + 1, current_epoch, phase, i+1, mean_loss, current_lr, mean_accuracy))
                        running_loss = 0.
                        total = 0.
                        correct = 0.
                        if phase == 'train_lc':
                            self.tepoch.append(current_epoch)
                            self.tloss.append(mean_loss)
                            self.taccu.append(mean_accuracy)
                            self.tlr.append(current_lr)
                        elif phase == 'val_lc':
                            self.vepoch.append(current_epoch)
                            self.vloss.append(mean_loss)
                            self.vaccu.append(mean_accuracy)
                    batch_counter += 1.
                batch_counter = 0.
    
    def save_current(self):
        torch.save(self.net.state_dict(), str(self.net.__class__.__name__)+'_'+str(self.epochs)+'.ptmodel')

    def load(self, filename):
        self.net.load_state_dict(torch.load(filename))

    def plot_lc(self):
        fig = plt.figure('learning_curves', (7,7))
        axa = fig.add_subplot(311)
        axb = fig.add_subplot(312)
        axc = fig.add_subplot(313)
        axa.plot(self.tepoch, self.taccu, '--', lw=1)
        axa.plot(self.vepoch, self.vaccu, '-', lw=1)
        axa.set_ylabel('accuracy [%]', size=15)
        axa.tick_params(axis='both', which='both', labelsize=10, bottom='off', top='off', labelbottom='off')

        axb.plot(self.tepoch, self.tloss, '--', label='train', lw=1)
        axb.plot(self.vepoch, self.vloss, '-', label='val', lw=1)
        axb.legend()
        axb.set_ylabel('loss', size=15)
        axb.tick_params(axis='both', which='both', labelsize=10, bottom='off', top='off', labelbottom='off')

        axc.plot(self.tepoch, self.tlr, lw=1)
        axc.set_yscale('log')
        axc.set_ylabel('learning rate', size=15)
        axc.set_xlabel('# Epochs', size=15)
        axc.tick_params(axis='both', which='both', labelsize=10, bottom='on', top='on', labelbottom='on')
        fig.canvas.draw()
        plt.show()

