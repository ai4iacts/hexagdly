import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

class Detector:
    def __init__(self):
        # hexagon size
        self.a = 1
        self.r = .5*3**(.5)*self.a
        # half the number of rows and columns
        ny = 5
        nx = 5
        # grid of odd columns
        y1 = np.linspace(0,2*ny*self.r,ny+1)
        x1 = np.linspace(0,2*nx*1.5*self.a,nx+1)
        X1,Y1 = np.meshgrid(x1,y1)
        # grid of even columns
        y2 = np.linspace(self.r,(2*ny+1)*self.r,ny+1)
        x2 = np.linspace(1.5*self.a,(2*nx+1)*1.5*self.a,nx+1)
        X2,Y2 = np.meshgrid(x2,y2)
        # join grids
        x = np.concatenate([X1,X2])
        y = np.concatenate([Y1,Y2])
        # select pixels within a circular region
        xc = .5*(x.max()-x.min())
        yc = .5*(y.max()-y.min())
        x,y = x-xc, y-yc
        r = np.sqrt(x**2 + y**2)
        ind = np.where(r<yc, True, False)
        x,y = x[ind],y[ind]
        # rotate pixels
        r = r[ind]
        t = np.arctan2(y,x)
        rot = np.pi/12
        self.rotation_angle = rot + np.pi/3
        self.x, self.y = r*np.cos(t+rot), r*np.sin(t+rot)



def make_colormap(seq):
    """Return a LinearSegmentedColormap
        seq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
        """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)




def plot(ax,x,y,grid=False,hexagons=False,lines=False):
    if hexagons:
        hexagons = []
        for x_,y_ in zip(x,y):
            hexagons.append(RegularPolygon((x_,y_), 6, 1, orientation=-np.pi/12))
        collection = PatchCollection(hexagons, cmap='gist_yarg', alpha=1, edgecolors='grey', linewidth=.5)
        colors = np.zeros(len(hexagons))
        collection.set_array(np.array(colors))
        ax.add_collection(collection)
    ax.scatter(x, y, color='k')
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')])
    ax.scatter(x, y, c=x, cmap=rvb)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_aspect('equal')
    if grid:
        ax.grid(which='both')

    if lines:
        xmin,xmax = -.5, x.max()+.5
        xbins = np.linspace(xmin,xmax,int(x.max())+2)
        ymin,ymax = -.5, y.max()+.5
        ybins = np.linspace(ymin,ymax,int(y.max())+2)
        ax.vlines(xbins,ymin,ymax, linewidths=.5)
        ax.hlines(ybins,xmin,xmax, linewidths=.5)
        ax.set_xlabel('i')
        ax.set_ylabel('j')
    return None
