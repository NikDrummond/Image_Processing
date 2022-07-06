### Image PLotting function/functions
from .Image_Class import Image, ImageList
from .Image_Processing import add_hull, get_points
import matplotlib.pyplot as plt

def plot(N, hull = False, **kwargs):

    if not isinstance(N,Image):

        if isinstance(N,ImageList):
            raise AttributeError('Please specifiy a specific Neuron in your image list')
        else:
            raise AttributeError('Object not recognised')

    if hull == True:
        if N.hull is None:
            raise AttributeError('Please use get hull function to add a convex hull to the image')


    fig, ax = plt.subplots(**kwargs)


    ax.matshow(N.array)

    if hull == True:
        for simplex in N.hull.simplicies:
            ax.plot(N.points[simplex,0], N.points[simplex,1],'w--')

    ax.set_ylabel(N.d_labels[0])
    ax.set_xlabel(N.d_labels[1])
    ax.set_title(N.name)

    return fig, ax
