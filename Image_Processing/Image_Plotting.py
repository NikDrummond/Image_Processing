### Image PLotting function/functions
from .Image_Class import Image, ImageList
import matplotlib.pyplot as plt

def plot(N, **kwargs):

    if not isinstance(N,Image):

        if isinstance(N,ImageList):
            raise AttributeError('Please specifiy a specific Neuron in your image list')
        else:
            raise AttributeError('Object not recognised')

    fig, ax = plt.subplots(**kwargs)

    ax.matshow(N.array)
    ax.set_ylabel(N.d_labels[0])
    ax.set_xlabel(N.d_labels[1])
    ax.set_title(N.name)

    return fig, ax
