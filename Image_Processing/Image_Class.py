### Image Class

import tifffile as tiff
import os
from tqdm import tqdm

def update_defaults(defaults, kwargs):
    """ Update default class arguments"""
    
    # check if anything else was passed...
    diff = set(kwargs.keys() - set(defaults.keys()))
    if diff:
        print('Invalid Input arguments:',tuple(diff), '. These have been ignored')
        for i in diff:
            del kwargs[i]

    #update defaults (just the name)
    defaults.update(kwargs)

    return defaults

class Image():
    """
    Simple Class for holding Neuron images (np.arrays)
    """

    def __init__(self, **kwargs):

        defaults = {"name": None,
                    "array": None,
                    "d_labels": None,
                    "point_size": None,
                    "points": None,
                    "hull": None}
        
        defaults = update_defaults(defaults, kwargs)

        for k, v in defaults.items():
            setattr(self,k,v)

    def save_tiff(self,f_name = None,meta = None):
        """ save a Neuron as a tiff using the ImageJ format

        Parameters
        ----------
        f_name:     str
                String with the file name

        meta:       None | dict
                If None ( default) axes labels and origin is added tot he meta data. if a dictionary is passed, this dictionary of meta data you wish to add is used
        """

       
        # save array as tiff
        if meta is None:
            meta = {'axes': self.d_labels}
        
        if f_name is None:
            f_name = self.name

        tiff.imwrite(f_name, self.array, imagej = True, metadata = meta)


class ImageList():
    """
    Simple list of multiple Neuron images
    
    """

    def __init__(self, N_list):
        self.Neurons = N_list

    def save_tiffs(self,f_names = None, meta = None):
        """
        
        """
        if meta is None:
            meta = [{'axes':N.d_labels} for N in self.Neurons]
        if f_names is None:
            f_names = [N.name for N in self.Neurons]

        for i in range(len(self.Neurons)):
            tiff.imwrite(f_names[i],
                        self.Neurons[i].array,
                        imagej = True,
                        metadata = meta[i])

def read_Image(path):
    """
    read .tiff image/images from a file or folder
    """
    if os.path.isfile(path):

        if (path.endswith('.tif')) or (path.endswith('.tiff')):
            with tiff.TiffFile(path) as tif:
                volume = tif.asarray()
                axes = tif.series[0].axes
                volume = volume.astype('float32')
                name = os.path.splitext(os.path.basename(path))[0]
                N = Image(name = name, array = volume, d_labels = axes)
                return N        
        else:
            raise AttributeError('File type not recognised')

    elif os.path.isdir(path):

        N_all = []

        for root, dirs, files in os.walk(path):
            for file in tqdm(files, desc = 'reading Neurons: '):
                if (file.endswith('.tif')) or (file.endswith('.tiff')):

                    f_path = os.path.join(root,file)

                    with tiff.TiffFile(f_path) as tif:
                        volume = tif.asarray()
                        axes = tif.series[0].axes
                        volume = volume.astype('float32')
                        name = os.path.splitext(os.path.basename(path))[0]
                        N_all.append(Image(name = name, array = volume, d_labels = axes))
        N_all = ImageList(N_all)
        return N_all
    else:
        raise TypeError('input is not a file or directory')
