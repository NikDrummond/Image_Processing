### Image PLotting function/functions
import matplotlib.pyplot as plt
import numpy as np

from .Image_Class import Image, ImageList
from .Image_Processing import add_hull, get_points


def plot(N, hull = False, size = (10,10), **kwargs):

    if not isinstance(N,Image):

        if isinstance(N,ImageList):
            raise AttributeError('Please specifiy a specific Neuron in your image list')
        if isinstance(N,np.ndarray):
            im = N
        if isinstance(N,list):
            if len(N) > 2:
                raise AttributeError(' More than two elements have been passed ')
        else:
            raise AttributeError('Object not recognised')
    else:
        im = N.array

    if hull == True:
        if N.hull is None:
            raise AttributeError('Please use get hull function to add a convex hull to the image')

    if not isinstance(N,list):

        fig, ax = plt.subplots(figsize = size,**kwargs)
        ax.matshow(im)

        if hull == True:
            for simplex in N.hull.simplicies:
                ax.plot(N.points[simplex,0], N.points[simplex,1],'w--')

        ax.set_ylabel(N.d_labels[0])
        ax.set_xlabel(N.d_labels[1])
        ax.set_title(N.name)

        return fig, ax

    elif isinstance(N,list):
        
        fig, axes = plt.subplots(1,2,figsize = (size[0]*2,size[1]))

        for i in range(len(N)):
            if isinstance(N[i], np.ndarray):
                axes[i].matshow(N[i])
            elif isinstance(N[i],Image):
                axes[i].matshow(N[i].array)

        return fig, axes
